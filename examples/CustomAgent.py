"""
Imports a custom enviroment from a XML file.
Produces a Cartesian motion using the Cartesian actuation mode.
"""
import math
import os
import sys

import mujoco as mj
import dm_env
import numpy as np
from collections import defaultdict

from dm_control import composer, mjcf
from dm_env import specs

from dm_robotics.panda import environment, arm_constants
from dm_robotics.panda import parameters as params
from dm_robotics.panda import run_loop, utils
from dm_robotics.moma import entity_initializer, prop
from dm_control.composer.variation import distributions, rotations
from dm_control.rl.control import Environment
from dm_robotics.agentflow import spec_utils
from dm_robotics.geometry import pose_distribution
from dm_control.composer import Entity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from rl_spin_decoupler import AgentSide

class MyoArm(Entity):
  """An entity class that wraps an MJCF model without any additional logic."""
  
  def _build(self, xml_path):
    self._mjcf_model = mjcf.from_path(xml_path)

  @property
  def mjcf_model(self):
      """Returns the MJCF model associated with this entity."""
      return self._mjcf_model

class Agent:
  """Agents are used to control a robot's actions given
  current observations and rewards.
  """

  def __init__(self, spec: specs.BoundedArray, ipbaselinespart:str, portbaselinespart:int) -> None:
    self._spec = spec
    self.state = 0
    self.time_state = 0.1
    self.elapsed = 4.0
    self.env_reset = True
    # Create an instance of the communication object and start communication
    self.agent_side = AgentSide(ipbaselinespart, portbaselinespart)
    self.agent_side.startComms()

  def pass_args(self, env: Environment, joint_names):
    self.env = env
    self.joint_names = joint_names
  
  def format_obs(self, qpos, time, forces):
    # Add time to observations
    obs = {'time': time}
    # Add franka and myoarm joints values to observations
    for i, name in enumerate(self.joint_names):
        obs.update({name : qpos[i]})
    # Add franka's wrist forces to observations
    obs.update({'F_wristX': forces[0],
                'F_wristY': forces[1],
                'F_wristZ': forces[2]})
    return obs

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """
    Computes velocities in the x/y plane parameterized in time.
    """
    # keys = timestep.observation.keys()
    # print(keys) -> Result:
    # ['panda_joint_pos', 'panda_joint_vel', 'panda_joint_torques', 'panda_tcp_pos', 
    # 'panda_tcp_quat', 'panda_tcp_rmat', 'panda_tcp_pose', 'panda_tcp_vel_world', 
    # 'panda_tcp_vel_relative', 'panda_tcp_pos_control', 'panda_tcp_quat_control', 
    # 'panda_tcp_rmat_control', 'panda_tcp_pose_control', 'panda_tcp_vel_control', 
    # 'panda_force', 'panda_torque', 'panda_gripper_width', 'panda_gripper_state', 
    # 'panda_twist_previous_action', 'time']
    time = timestep.observation['time'][0]
    forces = timestep.observation['panda_force']
    qpos_values = env.physics.data.qpos

    # for i, name in enumerate(self.joint_names):
    #     print(f"{name}: {qpos_values[i]}")
    # print("\n")
    # The action space of the Cartesian 6D effector corresponds to the
    # linear and angular velocities in x, y and z directions respectively 
    # Demo State Machine:
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    if self.state == 0: # Move through Y-Z axis
      action[0] = 0.0 # Vel X
      action[1] = -0.065 # Vel Y
      action[2] = 0.065 # Vel Z
      if (time - self.time_state) > self.elapsed:
        self.state = 1
        self.time_state = time
    elif self.state == 1: # Move through X axis
      action[0] = 0.065 # Vel X
      action[1] = 0.0 # Vel Y
      action[2] = 0.0 # Vel Z
      if (time - self.time_state) > self.elapsed:
        self.state = 2
        self.time_state = time
    elif self.state == 2: # Backwards through X axis
      action[0] = -0.065 # Vel X
      action[1] = 0.0 # Vel Y
      action[2] = 0.0 # Vel Z
      if (time - self.time_state) > self.elapsed:
        self.state = 3
        self.time_state = time
    elif self.state == 3: # Backwards throught Y-Z axis
      action[0] = 0.0 # Vel X
      action[1] = 0.065 # Vel Y
      action[2] = -0.065 # Vel Z
      if (time - self.time_state) > self.elapsed:
        self.state = 4
        self.time_state = time
    else:
      action[0] = 0.0 # Vel X
      action[1] = 0.0 # Vel Y
      action[2] = 0.0 # Vel Z

    ### COMMUNICATE WITH STABLE-BASELINES ###
    # print(f"Agent-Side:\tGoing on a new step...\n")
    try:
      # Receive the indicator of what to do
      action_type = self.agent_side.readWhatToDo()
      # Select the case
      if action_type == AgentSide.WhatToDo.SEND_OBS_REC_ACTION:
          # Create observation dict
          obs = self.format_obs(qpos_values, time, forces)
          # Send observation dict and receive actions dict
          action_sb = self.agent_side.sendObsRecAct(obs)
          print(f"Agent-Side:\tReceived STEP action:  {action_sb}\n")

      elif action_type == AgentSide.WhatToDo.RESET_SEND_OBS:
          print("\nRESETTING ENV TO START NEW EPISODE...\n")
          if self.env_reset:
            obs = self.format_obs(qpos_values, time, forces)
            self.agent_side.resetSendObs(obs)
            self.env_reset = False
          else:
            self.reset()
            return np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)

      elif action_type == AgentSide.WhatToDo.FINISH:
          # Finish training
          print("Experiment finished.")
          self.agent_side.stopComms()
          sys.exit()
    except Exception as e:
      print(f"Error in communication: {e}")
      print("Finishing experiment ...")
      self.agent_side.stopComms()
      sys.exit()

    self.agent_side.stopComms()
    self.agent_side.startComms()
    return action

  def reset(self):
    timestep = self.env.reset()
    time = timestep.observation['time'][0]
    forces = timestep.observation['panda_force']
    qpos_values = env.physics.data.qpos
    # reset time-state (for state machine)
    self.time_state = time
    # Create observation dict
    obs = self.format_obs(qpos_values, time, forces)
    # Just send observation dict
    self.agent_side.resetSendObs(obs)
    self.agent_side.stopComms()
    self.agent_side.startComms()


if __name__ == '__main__':
  # We initialize the default configuration for logging
  # and argument parsing. These steps are optional.
  utils.init_logging()
  parser = utils.default_arg_parser()
  args = parser.parse_args()

  # Load environment from an MJCF file.
  # Get python script path
  script_dir = os.path.dirname(os.path.abspath(__file__))
  print(script_dir)
  # Form absolute path to the XML file
  XML_ARENA_PATH = os.path.join(script_dir, "../../models/myo_sim/arm/myoPandaEnv.xml")
  print(XML_ARENA_PATH)
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
  # Form absolute path to the XML file
  XML_ARM_PATH = os.path.join(script_dir, "../../models/myo_sim/arm/myoarmPanda.xml")
  
  myoarm = MyoArm(xml_path=XML_ARM_PATH)
  panda_env._arena.attach(myoarm)

  bodies = panda_env._arena.mjcf_model.worldbody.find_all('body')
  # print(f"  ********\n\n {bodies} \n\n  ********")
  wrist = None
  ee = None
  for i, body in enumerate(bodies):
    if hasattr(body, "name") and body.name == "panda_hand":
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
    relpose=[0.025, 0.0, 0.115, # Posición de la muñeca desde el efector del robot
            0.0, 0.87, -0.50, 0.0],  # Rotación de la muñeca respecto del efector delo robot (180º,0º,60º)
  )

  # Obtain joint names of the resulting enviroment
  jnts = panda_env._arena.mjcf_model.worldbody.find_all('joint')
  joint_names = []
  for i, jnt in enumerate(jnts):
    if hasattr(jnt, "name"): 
      print(f"joint[{i}] : {jnt.name}")
      joint_names.append(jnt.name)

  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification
    utils.full_spec(env)
    # Initialize the agent
    agent = Agent(env.action_spec(), '192.168.0.18', 49053)
    agent.pass_args(env, joint_names)
    # Run the environment and agent inside the GUI.
    # app = utils.ApplicationWithPlot(width=1440, height=860)
    # app.launch(env, policy=agent.step)
    run_loop.run(env, agent, [], max_steps=1000, real_time=True)