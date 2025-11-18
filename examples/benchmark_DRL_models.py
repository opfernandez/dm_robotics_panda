import os
import sys
import argparse

import dm_env
import numpy as np

from dm_env import specs

from dm_robotics.panda import run_loop, utils
from dm_control.rl.control import Environment

from stable_baselines3 import SAC, TD3


from agent_side_base import *

class Agent(BaseAgent):
    """Agents are used to control a robot's actions given
    current observations and rewards.
    """

    def __init__(self, spec: specs.BoundedArray, 
                 model_path: str, 
                 home_path: str, 
                 selected_trajectory: str,
                 time_step: float,
                 env: Environment,
                 joint_names: list) -> None:
        # Initialize variables
        self._spec = spec
        self.time_state = 0.1
        self.step_time = time_step
        self.env_reset = True
        self.init = True
        # Flag to indicate if the agent is waiting for RL commands
        self._waitingforrlcommands = True
        self.episode_cont = 0
        self.action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        # Load trained model
        self.model = SAC.load(model_path)
        self.rewrite = True
        self.home_path = home_path
        self.data_path = os.path.join(self.home_path, "data")
        self.points_traj = 5000
        self.selected_trajectory = selected_trajectory
        # Magnitude thresholds for normalization
        self.max_vel = 0.25
        self.max_force_threshold = 20
        self.max_torque_threshold = 55
        self.max_dist = 0.15
        # Time step of the last action inferred
        self.last_action_step = 0.0
        # Environment and joint names
        self.env = env
        self.joint_names = joint_names

    def calculate_reward(self, obs: dict) -> float:
        """
        Calculate the reward based on the observations received from the agent side.
        Reward is based on the distance to the ideal trajectory, penalties for
        exceeding force and torque thresholds, and penalties for linear and angular velocities.
        The reward is shaped to encourage the agent to reach the target position while
        minimizing excessive velocities and forces.
        Args:
            obs (np.ndarray): Observations received from the agent side.
        Returns:
            total_rw (float): Calculated reward based on the observations.
        """
        # Configurations for reward calculation
        gain_pos = 2.5
        gain_vel = 0.75
        force_threshold = 20
        torque_threshold = 40

        # Position: positive reward with exponential shaping
        # Value between 0 and 10, higher the nearest to the point
        eud = obs[15]
        rw_dist = 10.0 * np.exp(-gain_pos * eud)

        # Penalty for lineal vel
        max_vel = np.array([1.0, 1.0, 1.0])
        norm_lineal_vel = np.linalg.norm(obs[6:9])
        norm_max_vel = np.linalg.norm(max_vel)
        rw_lineal_vel = gain_vel * (norm_lineal_vel/norm_max_vel)

        # Penalty for angular vel
        norm_angular_vel = np.linalg.norm(obs[9:12])
        rw_angular_vel = gain_vel * (norm_angular_vel/norm_max_vel)

        # High penalty for force-torque threshold exceed
        rw_safety = 0.0
        force_check = np.any(np.abs(obs[0:3]) >= (force_threshold/self.max_force_threshold))
        torque_check = np.any(np.abs(obs[3:6]) >= (torque_threshold/self.max_torque_threshold))
        if force_check or torque_check:
            rw_safety = 10.0
        
        # Calculate total reward
        total_rw = rw_dist - rw_lineal_vel - rw_angular_vel - rw_safety

        # SAC works better with normalization
        total_rw = np.clip(total_rw, -10.0, 10.0)

        print(f"\n\nrw_dist={rw_dist:.2f}\n" 
            f"rw_safety={-rw_safety:.2f}\n"
            f"rw_lineal_vel={-rw_lineal_vel:.2f}\n"
            f"rw_angular_vel={-rw_angular_vel:.2f}\n")
        print(f"Generated Reward: [{total_rw:.2f}]")
        return total_rw

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """
        Provides robot actions every control timestep. 
        To infer the action, it uses the trained DRL model
        and the current observations of the environment.
        Args:
            timestep: The current timestep of the environment.
        Returns:
            self.action: The action to be performed by the robot.
        """
        time_t = timestep.observation['time'][0]
        force = timestep.observation['panda_force']
        torque = timestep.observation['panda_torque']
        vel_ef = timestep.observation['panda_tcp_vel_world']
        ef_position = [timestep.observation['panda_tcp_pose'][0], # X
                        timestep.observation['panda_tcp_pose'][1], # Y
                        timestep.observation['panda_tcp_pose'][2]] # Z
        print("--"*30)
        print(f"Time [{time_t}]")        
        # Force and torque in local coordinates
        force_local = np.array(timestep.observation['panda_force'])  # [fx, fy, fz]
        torque_local = np.array(timestep.observation['panda_torque'])
        # Rotation matrix from local to base frame coordinates
        rmat = timestep.observation['panda_tcp_rmat']
        rotation_matrix = np.array(rmat).reshape(3, 3)
        # print(f"panda_tcp_rmat: {timestep.observation['panda_tcp_rmat']}")
        # Transform force and torque to base frame coordinates
        force_base = rotation_matrix @ force_local
        torque_base = rotation_matrix @ torque_local
        # Transform to world coordinates by permuting the X and Y axes
        force_world = np.array([force_base[1], force_base[0], force_base[2]])
        torque_world = np.array([torque_base[1], torque_base[0], torque_base[2]])
        # Calculate trajectory on first step:
        if self.init:
            self.calculate_trajectory(ef_position, self.selected_trajectory) # square // triangle // circle // pentagon
            self.save_data(os.path.join(self.data_path, "panda_ideal_traj_model.csv"), self.trajectory, 'w')
        eu_dist = self.calculate_eu_dist(time_t, ef_position)
        dist = self.calculate_dist(time_t, ef_position)
        follow_vector = self.calculate_follow_vector(time_t)
        ### INFERENCE ###
        obs, invalid_value = self.format_obs(force, torque, vel_ef, dist, eu_dist, follow_vector)
        obs_array = np.array(list(obs.values()), dtype=np.float32)
        reward = self.calculate_reward(obs_array)
        if self.init or (time_t - self.last_action_step >= (self.step_time - 5e-3)):
            print(f"Generating action after [{time_t - self.last_action_step:.2f}] seconds")
            self.last_action_step = time_t
            self.init = False
            # Predict action using the trained model
            self.action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
            act, _states = self.model.predict(obs_array, deterministic=True)
            self.action[0:6] = act
        # print(ef_position)
        # Save data
        mode = 'a'
        if self.rewrite: # reset files at first iteration
            mode = 'w'
            self.rewrite = False
        self.save_data(os.path.join(self.data_path, f"panda_traj_model_ts_{self.step_time:.1f}.csv"), [ef_position[0], ef_position[1], ef_position[2]], mode)
        self.save_data(os.path.join(self.data_path, f"panda_forces_model_ts_{self.step_time:.1f}.csv"), timestep.observation['panda_force'], mode)
        self.save_data(os.path.join(self.data_path, f"panda_torques_model_ts_{self.step_time:.1f}.csv"), timestep.observation['panda_torque'], mode)
        self.save_data(os.path.join(self.data_path, f"panda_joint_torques_model_ts_{self.step_time:.1f}.csv"), timestep.observation['panda_joint_torques'], mode)
        self.save_data(os.path.join(self.data_path, f"panda_vel_ef_model_ts_{self.step_time:.1f}.csv"), timestep.observation['panda_tcp_vel_world'], mode)
        self.save_data(os.path.join(self.data_path, f"panda_forces_model_world_ts_{self.step_time:.1f}.csv"), force_world, mode)
        self.save_data(os.path.join(self.data_path, f"panda_torques_model_world_ts_{self.step_time:.1f}.csv"), torque_world, mode)
        self.save_data(os.path.join(self.data_path, f"panda_reward_model_ts_{self.step_time:.1f}.csv"), [reward], mode)
        
        return self.action
    
    def reset(self):
        pass


if __name__ == '__main__':
    # Argument parsing.
    parser = argparse.ArgumentParser(description="Trained model for inference")
    parser.add_argument("-m", "--model", type=str, help="name of the .zip file resulting from training")
    parser.add_argument("-t", "--trajectory", type=str, default="h-square", help="type of trajectory to follow: ah-square, h-square, ah-triangle, h-triangle, ah-circle, h-circle")
    parser.add_argument("-s", "--time_step", type=float, help="time step for the agent's action update")
    args = parser.parse_args()

    # Build the Panda MyoArm environment.
    panda_env, joint_names = init_environment()

    # Get python script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(script_dir)
    model_path = os.path.join(script_dir, "..", "..", "checkpoints", args.model)
    home_path = os.path.join(script_dir, "..", "..")
    with panda_env.build_task_environment() as env:
        # Print the full action, observation and reward specification
        utils.full_spec(env)
        # Initialize the agent
        agent = Agent(spec=env.action_spec(), 
                      model_path=model_path,
                      home_path=home_path, 
                      selected_trajectory=args.trajectory, 
                      time_step=args.time_step,
                      env=env, 
                      joint_names=joint_names)
        # Run the environment and agent inside the GUI.
        # app = utils.ApplicationWithPlot(width=800, height=800)
        # app.launch(env, policy=agent.step)
        run_loop.run(env, agent, [], max_steps=1e10, real_time=False)
