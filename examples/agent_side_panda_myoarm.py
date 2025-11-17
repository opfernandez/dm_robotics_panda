import os
import sys
import argparse
import dm_env
import numpy as np
import time
from dm_env import specs

from dm_robotics.panda import run_loop, utils
from dm_control.rl.control import Environment

from agent_side_base import *


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from rl_spin_decoupler.spindecoupler import AgentSide
from rl_spin_decoupler.socketcomms.comms import BaseCommPoint

class Agent(BaseAgent):
    """Agents are used to control a robot's actions given
    current observations and rewards. This agent communicates with
    a stable-baselines agent through sockets.
    This agent does not implement any RL logic, it just sends the
    actions received from the stable-baselines agent to the environment.
    It also receives observations from the environment and sends them
    to the stable-baselines agent.
    """

    def __init__(self, spec: specs.BoundedArray, 
                 portbaselinespart:int, 
                 time_step: float,
                 env: Environment,
                 joint_names: list) -> None:
        """Initializes the agent.
        """
        super().__init__(spec, portbaselinespart, time_step, env, joint_names)

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """
        Provides robot actions every control timestep.
        This method is called by the environment at each control timestep.
        It calculates the trajectory to follow, the distance to the trajectory,
        the euclidean distance to the trajectory, and the follow vector.
        It also communicates with the stable-baselines agent to receive the action to be executed 
        by the robot and to send observations.
        It returns the action to be executed by the robot.
        The action is a numpy array with the same shape as the action specification of the environment.
        Args:
            timestep: The current timestep of the environment.
        Returns:
            action: The action to be executed by the robot.
        """
        whattodo = None
        # Get the current timestep observation
        time_t = timestep.observation['time'][0]
        force = timestep.observation['panda_force']
        torque = timestep.observation['panda_torque']
        vel_ef = timestep.observation['panda_tcp_vel_world']
        ef_position = [timestep.observation['panda_tcp_pose'][0], # X
                        timestep.observation['panda_tcp_pose'][1], # Y
                        timestep.observation['panda_tcp_pose'][2]] # Z
        print("--"*30)
        # Calculate trajectory on first step:
        if self.init:
            index_traj = ( self.episode_cont // 100) % len(self.traj_dict)
            print(f"Current trajectory is [{self.traj_dict[index_traj]}]")
            self.calculate_trajectory(ef_position, self.traj_dict[index_traj])
            self.init = False
        # Calculate distance and follow vector
        eu_dist = self.calculate_eu_dist(time_t, ef_position)
        dist = self.calculate_dist(time_t, ef_position)
        follow_vector = self.calculate_follow_vector(time_t)
        print(f"\tStep Time: [{time_t:.2f}] | Last Action Time: [{self.time_state:.2f}]")

        ### COMMUNICATE WITH STABLE-BASELINES ###
        if not self._waitingforrlcommands:
            print(f"Not waiting for RL commands...")
            if ((time_t - self.time_state) >= self.step_time - 0.09):
                print(f"Sending step obs to Agent-Side...")
                # Create observation dict
                obs, invalid_value = self.format_obs(force, torque, vel_ef, dist, eu_dist, follow_vector)
                if not invalid_value:
                    self.agent_side.stepSendObs(obs) # RL was waiting for this; no reward is actually needed here
                    self._waitingforrlcommands = True
                    print(f"Obs sent, now waiting for RL commands\n")
        else:
            # Receive the indicator of what to do
            # measure ms wwaiting for data
            start = time.perf_counter()
            while whattodo is None:
                whattodo = self.agent_side.readWhatToDo()
            end = time.perf_counter()
            print(f"Time waiting for RL data: {(end - start)*1000:.2f} ms")
            if whattodo is not None:
            # Select the case
                if  whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                    sb_action = whattodo[1]
                    lat = time_t - self.time_state
                    self._waitingforrlcommands = False # from now on, we are waiting to execute the action
                    self.agent_side.stepSendLastActDur(lat)
                    self.action = np.array(list(sb_action.values()), dtype=np.float32)
                    self.time_state = time_t
                    print("**"*30)
                    print(f"Received action: {sb_action}") 
                    print(f"After [{lat:.2f}] time")
                    print("**"*30)
                elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
                    print("\nRESETTING ENV TO START NEW EPISODE...\n")
                    self.reset()
                elif whattodo[0] == AgentSide.WhatToDo.FINISH:
                    # Finish training
                    print("Experiment finished.")
                    sys.exit()
                else:
                    raise(ValueError("Unknown indicator data"))
            else:
                print("No data received from stable-baselines agent.")
        print("--"*30)
        return self.action

    def reset(self):
        """Reset the environment to start a new episode.
        This method is called by the environment when a new episode starts.
        It resets the episode counter, sets the waiting flag to True,
        resets the environment, recalculates the trajectory, and sends the observation to the agent side.
        It also calculates the distance to the trajectory, the euclidean distance, and the follow vector.
        """
        # Reset the episode counter and waiting fla
        self.episode_cont += 1
        self._waitingforrlcommands = True
        # Get the current timestep observation
        timestep = self.env.reset()
        time_t = timestep.observation['time'][0]
        force = timestep.observation['panda_force']
        torque = timestep.observation['panda_torque']
        vel_ef = timestep.observation['panda_tcp_vel_world']
        ef_position = [timestep.observation['panda_tcp_pose'][0], # X
                        timestep.observation['panda_tcp_pose'][1], # Y
                        timestep.observation['panda_tcp_pose'][2]] # Z
        # Recalculate trajectory
        index_traj = (self.episode_cont // 100) % len(self.traj_dict)
        print(f"Current trajectory is [{self.traj_dict[index_traj]}]")
        self.calculate_trajectory(ef_position, self.traj_dict[index_traj])
        # Calculate distance and follow vector
        eu_dist = self.calculate_eu_dist(time_t, ef_position)
        dist = self.calculate_dist(time_t, ef_position)
        follow_vector = self.calculate_follow_vector(time_t)
        # reset time-state (for state machine)
        self.time_state = time_t
        # Create observation dict
        obs, invalid_value = self.format_obs(force, torque, vel_ef, dist, eu_dist, follow_vector)
        # Send observation dict
        self.agent_side.resetSendObs(obs)
        print(f"Agent-Side:\tRESET obs send\n")


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, help="Sockets port communication")
    parser.add_argument("-t", "--time_step", type=float, default=0.2, help="Time step between actions (default: 0.2s)")
    args = parser.parse_args()

    if args.port is None:
        print("Error: Port number is required. Use -p or --port to specify the port.")
        sys.exit(1)

    # Build the Panda MyoArm environment.
    panda_env, joint_names = init_environment()

    with panda_env.build_task_environment() as env:
        # Print the full action, observation and reward specification
        utils.full_spec(env)
        # Initialize the agent
        agent = Agent(
            spec=env.action_spec(),
            portbaselinespart=args.port,
            time_step=args.time_step,
            env=env,
            joint_names=joint_names
        )
        # Run the environment and agent inside the GUI.
        # app = utils.ApplicationWithPlot(width=1440, height=860)
        # app.launch(env, policy=agent.step)
        run_loop.run(env, agent, [], max_steps=1e10, real_time=False)