"""Example of an agent that communicates with a stable-baselines agent
through sockets in a Panda environment with a MyoArm attached.
This agent does not implement any RL logic, it just sends the
actions received from the stable-baselines agent to the environment.
It also receives observations from the environment and sends them
to the stable-baselines agent.
It is used to control a MyoArm attached to a Panda robot in a simulated environment.
The main method is `step`, which is called at each control timestep.
Other methods are used to format the observations, save data to a CSV file,
and reset the environment. This example should help you to build your own
agent that communicates with a stable-baselines agent through sockets.
"""

import os
import sys
import argparse
import random
import csv
import dm_env
import numpy as np

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
from rl_spin_decoupler.spindecoupler import AgentSide
from rl_spin_decoupler.socketcomms.comms import BaseCommPoint

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
    current observations and rewards. This agent communicates with
    a stable-baselines agent through sockets.
    This agent does not implement any RL logic, it just sends the
    actions received from the stable-baselines agent to the environment.
    It also receives observations from the environment and sends them
    to the stable-baselines agent.
    """

    def __init__(self, spec: specs.BoundedArray, portbaselinespart:int) -> None:
        self._spec = spec
        # Initialize variables
        self.time_state = 0.1
        self.env_reset = True
        self.init = True
        # Flag to indicate if the agent is waiting for RL commands
        self._waitingforrlcommands = True
        # Time in seconds that the agent will wait before receiving the next action
        self.step_time = 0.35
        self.action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        # Create an instance of the communication object and start communication
        self.agent_side = AgentSide(BaseCommPoint.get_ip(), portbaselinespart)
        self.points_traj = 5000

    def pass_args(self, env: Environment, joint_names):
        self.env = env
        self.joint_names = joint_names

    def format_obs(self, ..., ..., ..., ..., ..., ...):
        """
        Format the observations to be sent to the stable-baselines agent.
        Args:
            ...: The observations to be formatted.
        Returns:
            obs: A dictionary containing the formatted observations.
            invalid_value: A boolean indicating if the observations are valid.
        """
        return obs, invalid_value

    def save_data(self, file_name, data, mode):
        """Save data to a CSV file.
        Args:
            file_name: The name of the file to save the data.
            data: The data to save in the file.
            mode: The mode to open the file. It can be 'w' for write or 'a' for append.
        """
        with open(file_name, mode=mode, newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """
        Provides robot actions every control timestep.
        This method is called by the environment at each control timestep.
        It also communicates with the stable-baselines agent to receive the action to be executed 
        by the robot and to send observations.
        It returns the action to be executed by the robot.
        The action is a numpy array with the same shape as the action specification of the environment.
        Args:
            timestep: The current timestep of the environment.
        Returns:
            action: The action to be executed by the robot.
        """
        # Get the current timestep observation
        time_t = timestep.observation['time'][0]

        ### COMMUNICATE WITH STABLE-BASELINES ###
        if not self._waitingforrlcommands:
            if self.env_reset or ((time_t - self.time_state) >= self.step_time):
                # Create observation dict
                obs, invalid_value = self.format_obs(..., ..., ..., ..., ..., ...)
                if not invalid_value:
                    self.agent_side.stepSendObs(obs) # RL was waiting for this; no reward is actually needed here
                    self._waitingforrlcommands = True
        else:
            # Receive the indicator of what to do
            whattodo = self.agent_side.readWhatToDo()
            if whattodo is not None:
            # Select the case
                if  whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                    sb_action = whattodo[1]
                    lat = time_t - self.time_state
                    self._waitingforrlcommands = False # from now on, we are waiting to execute the action
                    self.agent_side.stepSendLastActDur(lat)
                    self.action = np.array(list(sb_action.values()), dtype=np.float32)
                    self.env_reset = False
                    self.time_state = time_t
                    print("--"*30)
                    print(f"Received STEP [{time_t:.2f}] action: {sb_action}") 
                    print(f"After [{lat:.2f}] time")
                    print("--"*30)
                elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
                    print("\nRESETTING ENV TO START NEW EPISODE...\n")
                    if self.env_reset:
                        obs, invalid_value = self.format_obs(..., ..., ..., ..., ..., ...)
                        self.agent_side.resetSendObs(obs)
                        self.time_state = time_t
                    else:
                        self.reset()
                elif whattodo[0] == AgentSide.WhatToDo.FINISH:
                    # Finish training
                    print("Experiment finished.")
                    sys.exit()
                else:
                    raise(ValueError("Unknown indicator data"))
        return self.action

    def reset(self):
        """Reset the environment to start a new episode.
        This method is called by the environment when a new episode starts.
        It resets the episode counter, sets the waiting flag to True and resets the environment.
        """
        # Reset the episode counter and waiting fla
        self._waitingforrlcommands = True
        self.env_reset = True
        # Get the current timestep observation
        timestep = self.env.reset()
        time_t = timestep.observation['time'][0]
        # Recalculate trajectory
        # reset time-state (for state machine)
        self.time_state = time_t
        # Create observation dict
        obs, invalid_value = self.format_obs(..., ..., ..., ..., ..., ...)
        # Send observation dict
        self.agent_side.resetSendObs(obs)
        print(f"Agent-Side:\tRESET obs send\n")


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser(description="Sockets port communication is required")
    parser.add_argument("-p", "--port", type=int, help="Sockets port communication")
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

    # Form absolute path to the XML file
    XML_ARM_PATH = os.path.join(script_dir, "../../models/myo_sim/arm/myoarmPanda.xml")
    
    myoarm = MyoArm(xml_path=XML_ARM_PATH)
    panda_env._arena.attach(myoarm)

    # Print the MJCF model of the environment
    # This piece of code is only used to print the MJCF model of the environment to the console.
    bodies = panda_env._arena.mjcf_model.worldbody.find_all('body')
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

    # Add a weld constraint to the end-effector and the wrist
    # This will make the wrist follow the end-effector position and orientation
    equality_constraint = panda_env._arena.mjcf_model.equality.add(
        'weld',
        body1=ee,  
        body2=wrist, 
        relpose=[0.025, 0.0, 0.115, # Position of the wrist relative to the end-effector
                0.0, 0.87, -0.50, 0.0],  # Orientation of the wrist relative to the end-effector (180º,0º,60º)
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
        agent = Agent(env.action_spec(), args.port)
        agent.pass_args(env, joint_names)
        # Run the environment and agent inside the GUI.
        # app = utils.ApplicationWithPlot(width=1440, height=860)
        # app.launch(env, policy=agent.step)
        run_loop.run(env, agent, [], max_steps=1e10, real_time=False)