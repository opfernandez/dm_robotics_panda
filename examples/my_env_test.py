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
  XML_ARM_PATH = "/home/oscar/TFM/models/myo_sim/arm/myoarmPanda.xml"
  myoarm = MyoArm(xml_path=XML_ARM_PATH)
  arena.attach(myoarm)

  robot_param = params.RobotParams(name='panda',
                                  pose=(1.3, 0.6, 0.6, 0.0, 0.0, np.pi),
                                  actuation=arm_constants.Actuation.CARTESIAN_VELOCITY)
  panda_env = environment.PandaEnvironment(robot_param, arena=arena)

  with panda_env.build_task_environment() as env:
    # Print the full action, observation and reward specification
    utils.full_spec(env)
    # Initialize the agent
    agent = Agent(env.action_spec())
    # Run the environment and agent either in headless mode or inside the GUI.
    app = utils.ApplicationWithPlot()
    app.launch(env, policy=agent.step)
