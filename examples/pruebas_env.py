"""
Imports a custom enviroment from a XML file.
Produces a Cartesian motion using the Cartesian actuation mode.
"""
import math
import os

import mujoco as mj
import dm_env
import numpy as np
from dm_control import composer, mjcf
from dm_env import specs

from dm_robotics.panda import environment, arm_constants
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils
from dm_robotics.moma import entity_initializer, prop
from dm_control.composer.variation import distributions, rotations
from dm_robotics.agentflow import spec_utils
from dm_robotics.geometry import pose_distribution
from dm_control.composer import Entity

class MyoArm(Entity):
  """An entity class that wraps an MJCF model without any additional logic."""
  
  def _build(self, xml_path):
    self._mjcf_model = mjcf.from_path(xml_path)

  @property
  def mjcf_model(self):
      """Devuelve el modelo MJCF asociado a esta entidad."""
      return self._mjcf_model

class Agent:
  """Agents are used to control a robot's actions given
  current observations and rewards. This agent does nothing.
  """

  def __init__(self, spec: specs.BoundedArray) -> None:
    self._spec = spec
    self.state = 0
    self.time_state = 0.1
    self.elapsed = 3.5

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """
    Computes velocities in the x/y plane parameterized in time.
    """
    time = timestep.observation['time'][0]
    # r = 0.1
    # vel_x = r * math.cos(time)  # Derivative of x = sin(t)
    # vel_y = r * ((math.cos(time) * math.cos(time)) -
    #              (math.sin(time) * math.sin(time)))
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    # The action space of the Cartesian 6D effector corresponds to the
    # linear and angular velocities in x, y and z directions respectively 
    # Demo State Machine:
    if self.state == 0: # Move through Y-Z axis
      action[0] = 0.0 # Vel X
      action[1] = -0.075 # Vel Y
      action[2] = 0.1 # Vel Z
      if (time - self.time_state) > self.elapsed:
        self.state = 1
        self.time_state = time
    elif self.state == 1: # Move through X axis
      action[0] = 0.075 # Vel X
      action[1] = 0.0 # Vel Y
      action[2] = 0.0 # Vel Z
      if (time - self.time_state) > self.elapsed:
        self.state = 2
        self.time_state = time
    elif self.state == 2: # Backwards through X axis
      action[0] = -0.075 # Vel X
      action[1] = 0.0 # Vel Y
      action[2] = 0.0 # Vel Z
      if (time - self.time_state) > self.elapsed:
        self.state = 3
        self.time_state = time
    elif self.state == 3: # Backwards throught Y-Z axis
      action[0] = 0.0 # Vel X
      action[1] = 0.075 # Vel Y
      action[2] = -0.075 # Vel Z
      if (time - self.time_state) > self.elapsed:
        self.state = 4
        self.time_state = time
    else:
      action[0] = 0.0 # Vel X
      action[1] = 0.0 # Vel Y
      action[2] = 0.0 # Vel Z
    return action


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Load environment from an MJCF file.
  XML_ARENA_PATH = "/home/oscar/TFM/models/myo_sim/arm/myoPandaEnv.xml"
  arena = composer.Arena(xml_path=XML_ARENA_PATH)

  # Buscar posición del efector final
  robot_param = params.RobotParams(name='panda',
                                  pose=(0.0, 0.04, 0.66, 0.0, 0.0, np.pi/2),
                                  actuation=arm_constants.Actuation.CARTESIAN_VELOCITY)
  panda_env = environment.PandaEnvironment(robot_param, arena=arena)

  # to set the end-effector pose from the distribution above.
  pose0 = pose_distribution.ConstantPoseDistribution(
      np.array([0.0, 0.44, 1.05, np.pi, 0.0, np.pi/2]))
  initialize_arm = entity_initializer.PoseInitializer(
      panda_env.robots[robot_param.name].position_gripper,
      pose_sampler=pose0.sample_pose)
  panda_env.add_entity_initializers([
      initialize_arm])

  # physics = mjcf.Physics.from_mjcf_model(panda_env._arena.mjcf_model)
  # panda_env.robots['panda'].position_gripper(position=np.array([-0.3, 0.25, 0.4]),
  #                                                     quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
  #                                                     physics=physics)

  # TODO: emplear native attachmet para el myoarm y sus includes en vez de un XML kilométrico
  XML_ARM_PATH = "/home/oscar/TFM/models/myo_sim/arm/myoarmPanda.xml"
  myoarm = MyoArm(xml_path=XML_ARM_PATH)
  panda_env._arena.attach(myoarm)

  bodies = panda_env._arena.mjcf_model.worldbody.find_all('body')
  print(f"  ********\n\n {bodies} \n\n  ********")
  print(bodies[-1].name)
  wrist = None
  ee = None
  for i, body in enumerate(bodies):
    if hasattr(body, "name") and body.name == "panda_rightfinger":
      ee = body
      print(f"[{ee.name}] was found")
      break 
  for i, body in enumerate(bodies):
    if hasattr(body, "name") and body.name == "scaphoid":
      wrist = body
      print(f"[{wrist.name}] was found")
      break
  if ee == None or wrist == None:
    print("Error finding wrist or end effector !!")
    exit()
  # Establecer restricción entre la muñeca y el efector final para que se muevan
  equality_constraint = panda_env._arena.mjcf_model.equality.add(
    'weld',
    body1=ee,  
    body2=wrist, 
    relpose=[0.025, -0.05, 0.075, # Posición de la muñeca desde el efector del robot
            0.0, 0.87, -0.50, 0.0],  # Rotación de la muñeca respecto del efector delo robot (180º,0º,60º)
  )




  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification
    utils.full_spec(env)
    # Initialize the agent
    agent = Agent(env.action_spec())
    # Run the environment and agent inside the GUI.
    # app = utils.ApplicationWithPlot()
    app = utils.ApplicationWithPlot(width=1440, height=860)
    app.launch(env, policy=agent.step)