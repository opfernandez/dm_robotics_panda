import os
import sys
import numpy as np
import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, ProgressBarCallback, CallbackList, CheckpointCallback, BaseCallback
import gymnasium as gym
from gymnasium.spaces import Box
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from rl_spin_decoupler.spindecoupler import RLSide

# Custom callback to save the replay buffer
class SaveReplayBufferCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, name_prefix: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            filename = f"{self.name_prefix}_latest_replay_buffer.pkl"
            full_path = os.path.join(self.save_path, filename)
            self.model.save_replay_buffer(full_path)
            if self.verbose:
                print(f"[Replay Buffer] Saved to: {full_path}")
        return True

class CustomEnv(gym.Env):
    def __init__(self, port: int):
        super(CustomEnv, self).__init__()
        # Create an instance of the communication object
        self.baselines_side = RLSide(port)
        """
        Observation space is conformed by:
        - force: End effector measured force (axis X,Y,Z). [0, 1, 2]
        - torque: End effector measured torque (axis X,Y,Z). [3, 4, 5]
        - ef_vel: End effector measured Cartesian velocity (axis X,Y,Z, roll, pitch, yaw). [6, 7, 8, 9, 10, 11]
        - End effector distance to ideal position: X [12], Y[13], Z [14]
        - Euclidean distance between end effector position and ideal trajectory position. [15]
        - Vector formed by step t and t+1 trajectory points (axis X,Y,Z). [16, 17, 18]
        The observation space is a Box with shape (19,) and dtype float32.
        """
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)
        # The action space are the Cartesian Velocities (3-translations, 3-rotations)
        self.lineal_vel = 0.25
        self.angular_vel = 0.25
        low_ap_limits = np.array([-self.lineal_vel, -self.lineal_vel, -self.lineal_vel, -self.angular_vel, -self.angular_vel, -self.angular_vel], dtype=np.float32)
        up_ap_limits = np.array([self.lineal_vel, self.lineal_vel, self.lineal_vel, self.angular_vel, self.angular_vel, self.angular_vel], dtype=np.float32)
        self.action_space = Box(low=low_ap_limits, high=up_ap_limits, shape=(6,), dtype=np.float32)
        # Set the maximum number of steps per episode
        self.max_step = 400
        self.end_ep_cont = 0
        # If robot exceeds any of these threshold the episode ends
        self.max_force_threshold = 25
        self.max_torque_threshold = 55
        self.prev_eud = None

    def reset(self, seed=None, options=None):
        """
        Reset the environment and prepare it for a new episode.
        Args:
            seed (int, optional): Random seed for reproducibility.
            options (dict, optional): Additional options for the reset.
        Returns:
            obs (np.ndarray): Initial observation after reset.
        """
        super().reset(seed=seed, options=options)
        print("SB-Side:\tGoing on a new reset...\n")
        # Communicate SB3 loop with Agent loop using sockets
        reseting = True
        self.prev_eud = None
        while reseting:
            try:
                obs = self.baselines_side.resetGetObs(timeout=360)
                reseting = False
            except Exception as e:
                print(f"Error in communication descrption: {e}")
            
        # print(f"SB-Side:\tObservation received after reset:{obs}\n")
        return np.array(list(obs.values()), dtype=np.float32), {}

    def format_actions(self, actions):
        """
        Format the actions to be sent to the agent side.
        Args:
            actions (np.ndarray): Actions to be sent to the agent side.
        Returns:
            act (dict): Formatted actions dictionary to be sent to the agent side.
        """
        # Actions are Cartesian velocities (3-translations, 3-rotations)
        vel_cart = ["Vx", "Vy", "Vz", "roll", "pitch", "yaw"]
        act = {}
        for i, v in enumerate(vel_cart):
            act.update({v : actions[i]})
        act.update({"noused" : 0.0})
        return act

    def calculate_reward(self, obs, done):
        """
        Calculate the reward based on the observations received from the agent side.
        Reward is based on the distance to the ideal trajectory, penalties for
        exceeding force and torque thresholds, and penalties for linear and angular velocities.
        The reward is shaped to encourage the agent to reach the target position while
        minimizing excessive velocities and forces.
        Args:
            obs (np.ndarray): Observations received from the agent side.
            done (bool): Whether the episode is done.
        Returns:
            total_rw (float): Calculated reward based on the observations.
        """
        # Configuraciones
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

        print(f"\nrw_dist={rw_dist:.2f}\n" 
            f"rw_safety={-rw_safety:.2f}\n"
            f"rw_lineal_vel={-rw_lineal_vel:.2f}\n"
            f"rw_angular_vel={-rw_angular_vel:.2f}\n")
        print(f"Generated Reward: [{total_rw:.2f}]")
        return total_rw

    def end_episode(self, obs):
        """
        Check if the episode should end based on the observations received.
        The episode ends if the force or torque thresholds are exceeded,
        or if the maximum number of steps is reached.
        Args:
            obs (np.ndarray): Observations received from the agent side.
        Returns:
            truncated (bool): Whether the episode was truncated.
            done (bool): Whether the episode is done.
        """
        # Increment the episode counter
        self.end_ep_cont += 1
        done = False
        truncated = False
        # Check force and torque thresholds
        force_check = np.any(np.abs(obs[0:3]) >= (self.max_force_threshold/self.max_force_threshold))
        torque_check = np.any(np.abs(obs[3:6]) >= (self.max_torque_threshold/self.max_torque_threshold))
        if force_check or torque_check:
            done = True
            self.end_ep_cont = 0
        # Check trajectory end
        if self.end_ep_cont >= self.max_step:
            truncated = True
            self.end_ep_cont = 0
        return truncated, done

    def step(self, action):
        """
        Step function to send actions to the agent side and receive observations.
        Reward and done flags are calculated based on the observations received.
        Args:
            action (np.ndarray): Action to be sent to the agent side.
        Returns:
            obs (np.ndarray): Observations received from the agent side.
            reward (float): Reward calculated based on the observations.
            done (bool): Whether the episode is done.
            truncated (bool): Whether the episode was truncated.
        """
        # Generate actions dict for transmission
        action_dict = self.format_actions(action)
        # print(action_dict)
        # Get observations dict from agent and send actions dict
        _, obs, _ = self.baselines_side.stepSendActGetObs(action_dict, timeout=300)
        print("--"*30)
        print(f"Received obs:")
        print_obs = list(obs.items())[:]
        for key, val in print_obs:
            print(f"{key}: {val:.3f}", end=", ")
        # Convertfrom dict to np.parray (used by sb3)
        obs_array = np.array(list(obs.values()), dtype=np.float32)
        # Check end episode conditions
        truncated, done = self.end_episode(obs_array)
        # Calculate reward based on observations 
        reward = self.calculate_reward(obs_array, done)
        print("--"*30)
        if done:
            print("\nSending DONE to Agent Side\n")
        return obs_array, reward, done, truncated, {}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port", type=int, help="Sockets port communication")
    parser.add_argument("-n", "--name", type=str, help="Checkpoint name")
    args = parser.parse_args()

    if args.port is None or args.name is None:
        import sys
        print("Error: Port number and checkpoint name are required.\n Use -p and -n flags to provide them.")
        sys.exit(1)
    
    # Get script dir for relative path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_path = os.path.join(script_dir, "../../checkpoints")
    tensorboard_log_path = os.path.join(script_dir, "../../train_logs")
    
    # Create custom enviroment
    env = CustomEnv(port=args.port)

    # Uncomment the next line to use action noise
    # Some Gaussian noise on acctions for safer sim-world transfer
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = SAC("MlpPolicy", env,
                learning_rate=0.0005, #0.0003 0.001
                learning_starts=1000,
                batch_size = 256,
                gamma=0.9999,
                train_freq=(1, "step"),
                gradient_steps=1,
                #action_noise=action_noise,  
                verbose=1,
                tensorboard_log=tensorboard_log_path)

    name_prefix = args.name
    print(f"Training with checkpoint name: {name_prefix}, results will be saved in: {checkpoints_path}")
    # Train the model
    callback_max_ep = StopTrainingOnMaxEpisodes(max_episodes=5e6, verbose=1)
    pb_callback = ProgressBarCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # save model every 50k steps
        save_path=checkpoints_path,
        name_prefix=name_prefix
    )
    replay_buffer_callback = SaveReplayBufferCallback(
        save_freq=250_000,
        save_path=checkpoints_path,  
        name_prefix=name_prefix,
        verbose=1
    )
    callbacks = CallbackList([pb_callback, callback_max_ep, checkpoint_callback, replay_buffer_callback])
    model.learn(int(1e10), callback=callbacks)

    # Save the resulting model
    model.save(name_prefix + "_final_model")

if __name__ == "__main__":
    main()