"""
Imports a custom enviroment from a XML file.
Produces a Cartesian motion using the Cartesian actuation mode.
"""
import math
import os

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

  def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
    """
    Computes velocities in the x/y plane parameterized in time.
    """
    time = timestep.observation['time'][0]
    r = 0.1
    vel_x = r * math.cos(time)  # Derivative of x = sin(t)
    vel_y = r * ((math.cos(time) * math.cos(time)) -
                 (math.sin(time) * math.sin(time)))
    action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
    # The action space of the Cartesian 6D effector corresponds
    # to the linear and angular velocities in x, y and z directions
    # respectively
    action[0] = vel_x
    action[1] = vel_y
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

  robot_param = params.RobotParams(name='panda',
                                  pose=(-0.7, 0.55, 0.5, 0.0, 0.0, 0), # base position
                                  actuation=arm_constants.Actuation.CARTESIAN_VELOCITY)
  panda_env = environment.PandaEnvironment(robot_param, arena=arena)
  # TODO: emplear native attachmet para el myoarm y sus includes en vez de un XML kilométrico
  XML_ARM_PATH = "/home/oscar/TFM/models/myo_sim/arm/myoarmPanda.xml"
  myoarm = MyoArm(xml_path=XML_ARM_PATH)
  arena.attach(myoarm)
  # print(arena.mjcf_model.find_all('site'))
  mjcf_model = arena.mjcf_model

  # Buscar posición de la muñeca
  wrist_site = myoarm.mjcf_model.find('site', 'FDS_ellipsoid_site_FDS4_side')
  # Buscar posición del efector final
  robot_ee = panda_env._arena.mjcf_model.worldbody.find_all('body')
  # panda_link1 = panda_env._arena.mjcf_model.find('body', 'panda_gripper/')
  # print(panda_link1)

    # Verificar los cuerpos del brazo humano
  # print("\n\nBodies en el modelo del brazo humano (MyoArm):")
  # for body in myoarm.mjcf_model.find_all('body'):
  #     print(body.name)

  # # Verificar los cuerpos del brazo robótico
  # print("\n\nBodies en el modelo del Panda:")
  # for body in panda_env._arena.mjcf_model.find_all('body'):
  #     print(body.name)
  
  # print(robot_ee)
  print(wrist_site)
  # Crear un joint entre la muñeca y el efector final
  joint = myoarm.mjcf_model.add(
      'joint',
      type='ball',  # 3 grados de libertad
      name='human_robot_joint',
      pos=wrist_site.pos,  # Posición del joint en la muñeca
  )

  # Agregar una restricción de igualdad entre ambos
  myoarm.mjcf_model.equality.add(
      'joint',
      joint1=joint.full_identifier,
      joint2=robot_ee.full_identifier,
      name='wrist_to_ee_constraint',
  )


  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification
    utils.full_spec(env)
    # Initialize the agent
    agent = Agent(env.action_spec())
    # Run the environment and agent inside the GUI.
    app = utils.ApplicationWithPlot()
    app.launch(env, policy=agent.step)
