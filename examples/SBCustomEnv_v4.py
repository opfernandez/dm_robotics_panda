import os
import sys
import numpy as np
import argparse
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, ProgressBarCallback, CallbackList, CheckpointCallback
import gymnasium as gym
from gymnasium.spaces import Box, Dict, Discrete
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from rl_spin_decoupler.spindecoupler import RLSide

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
        """
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        # The action space are the Cartesian Velocities (3-translations, 3-rotations)
        self.lineal_vel = 0.25
        self.angular_vel = 0.25
        low_ap_limits = np.array([-self.lineal_vel, -self.lineal_vel, -self.lineal_vel, -self.angular_vel, -self.angular_vel, -self.angular_vel], dtype=np.float32)
        up_ap_limits = np.array([self.lineal_vel, self.lineal_vel, self.lineal_vel, self.angular_vel, self.angular_vel, self.angular_vel], dtype=np.float32)
        self.action_space = Box(low=low_ap_limits, high=up_ap_limits, shape=(6,), dtype=np.float32)
        self.max_step = 400
        self.end_ep_cont = 0
        # If robot exceeds any of these threshold the episode ends
        self.max_force_threshold = 48
        self.max_torque_threshold = 60
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        print("SB-Side:\tGoing on a new reset...\n")
        # Communicate SB3 loop with Agent loop using sockets
        reseting = True
        while reseting:
            try:
                obs = self.baselines_side.resetGetObs(timeout=360)
                reseting = False
            except Exception as e:
                print(f"Error in communication descrption: {e}")
            
        # print(f"SB-Side:\tObservation received after reset:{obs}\n")
        return np.array(list(obs.values()), dtype=np.float32), {}

    def format_actions(self, actions):
        vel_cart = ["Vx", "Vy", "Vz", "roll", "pitch", "yaw"]
        act = {}
        for i, v in enumerate(vel_cart):
            act.update({v : actions[i]})
        act.update({"noused" : 0.0})
        return act

    def calculate_reward(self, obs, done):
        """
        Reward = +Reward(eu_dist) +Reward(continuity) -Reward(esfuerzo)
            Reward(eu_dist) = 200 - np.clip((gain*obs[-1]), 0, 200)
            Reward(effort) -> If torque or force threshold is exceeded the agent is punished with -200.
            Reward(continuity) = (100/max_steps)*step_cont -> rewarding the agent for providing continuous and non-charging cinematics in causes that end the episode
            Reward(energy) = 100*energy / max_energy -> reward used to minimize using high ef cartesian velocities
        """
        # Configuraciones
        gain_pos = 10.0
        gain_vel = 0.75
        gain_force = 0.75
        gain_torque = 0.75
        force_threshold = 15
        torque_threshold = 35

        # Position: positive reward with exponential shaping
        # Value between 0 and 10, higher the nearest to the point
        eud = obs[15]
        rw_dist = 10.0 * np.exp(-gain_pos * eud) 

        # Penalty for lineal vel
        max_lineal_vel = np.array([self.lineal_vel, self.lineal_vel, self.lineal_vel])
        norm_lineal_vel = np.linalg.norm(obs[6:9])
        norm_max_lineal_vel = np.linalg.norm(max_lineal_vel)
        rw_lineal_vel = gain_vel * (norm_lineal_vel/norm_max_lineal_vel)

        # Penalty for angular vel
        max_angular_vel =np.array([self.angular_vel, self.angular_vel, self.angular_vel])
        norm_angular_vel = np.linalg.norm(obs[9:12])
        norm_max_angular_vel = np.linalg.norm(max_angular_vel)
        rw_angular_vel = gain_vel * (norm_angular_vel/norm_max_angular_vel)

        # Penalty for force
        max_force =np.array([self.max_force_threshold, self.max_force_threshold, self.max_force_threshold])
        norm_force = np.linalg.norm(obs[0:3])
        norm_max_force = np.linalg.norm(max_force)
        rw_force = gain_force * (norm_force/norm_max_force)

        # Penalty for torques
        max_torque =np.array([self.max_torque_threshold, self.max_torque_threshold, self.max_torque_threshold])
        norm_torque = np.linalg.norm(obs[3:6])
        norm_max_torque = np.linalg.norm(max_torque)
        rw_torque = gain_torque * (norm_torque/norm_max_torque)

        # High penalty for force-torque threshold exceed
        rw_safety = 0.0
        force_check = np.any(np.abs(obs[0:3]) > force_threshold)
        torque_check = np.any(np.abs(obs[3:6]) > torque_threshold)
        if force_check or torque_check:
            rw_safety = 10.0

        # Total reward sum
        total_rw = rw_dist - rw_lineal_vel - rw_angular_vel - rw_force - rw_torque - rw_safety

        # SAC works better with normalization
        total_rw = np.clip(total_rw, -10.0, 10.0)
        
        print(f"\nrw_dist={rw_dist:.2f}\n" 
            f"rw_safety={-rw_safety:.2f}\n"
            f"rw_lineal_vel={-rw_lineal_vel:.2f}\n"
            f"rw_angular_vel={-rw_angular_vel:.2f}\n"
            f"rw_force={-rw_force:.2f}\n"
            f"rw_torque={-rw_torque:.2f}\n")
        print(f"Generated Reward: [{total_rw:.2f}]")
        return total_rw

    def end_episode(self, obs):
        """
        Episode must end if the robot has completed the trajectory or the
        force or torque threshold has been exceeded.
        """
        self.end_ep_cont += 1
        done = False
        truncated = False
        
        # Check force and torque thresholds
        force_check = np.any(np.abs(obs[0:3]) > self.max_force_threshold)
        torque_check = np.any(np.abs(obs[3:6]) > self.max_torque_threshold)
        if force_check or torque_check:
            done = True
            self.end_ep_cont = 0
        # Check trajectory end
        if self.end_ep_cont >= self.max_step:
            truncated = True
            self.end_ep_cont = 0
        return truncated, done

    def step(self, action):
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
        #print(f"SB-Side:\tObservation received after step:{obs_array[:10]}\n")
        # Check end episode conditions
        truncated, done = self.end_episode(obs_array)
        # Calculate reward based on observations 
        reward = self.calculate_reward(obs_array, done)
        print("--"*30)
        if done:
            print("\nSending DONE\n")
        return obs_array, reward, done, truncated, {}

def main():
    parser = argparse.ArgumentParser(description="Sockets port communication is required")
    parser.add_argument("-p", "--port", type=int, help="Sockets port communication")
    args = parser.parse_args()
    
    # Get script dir for relative path settings
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoits_path = os.path.join(script_dir, "../../checkpoints")
    tensorboard_log_path = os.path.join(script_dir, "../../train_logs")

    # Create custom enviroment
    env = CustomEnv(port=args.port)

    # Some Gaussian noise on acctions for safer sim-world transfer
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = SAC("MlpPolicy", env,
                learning_rate=0.0008, #0.0003 0.001
                learning_starts=15000,
                batch_size = 256,
                gamma=0.9999,
                train_freq=(1, "step"),
                gradient_steps=1,
                #action_noise=action_noise,  
                verbose=1, 
                tensorboard_log=tensorboard_log_path)

    # Train the model
    callback_max_ep = StopTrainingOnMaxEpisodes(max_episodes=5e6, verbose=1)
    pb_callback = ProgressBarCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # save model every 50k steps
        save_path=checkpoits_path,
        name_prefix="sac_panda_home_v20"
    )

    callbacks = CallbackList([pb_callback, callback_max_ep, checkpoint_callback])
    model.learn(int(1e10), callback=callbacks)

    # Save the resulting model
    model.save("SAC-SB3")

if __name__ == "__main__":
    main()


########################
#         TD3          #
########################

# # # Create custom enviroment
# env = CustomEnv(port=49055)

# # Some Gaussian noise on acctions for safer sim-world transfer
# n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# # Initilize de model
# tensorboard_log_path = "/home/oscar/TFM/train_logs"
# model = TD3("MlpPolicy", env,
#             learning_rate=0.002, 
#             learning_starts=1000,
#             batch_size = 256,
#             train_freq=1,
#             gradient_steps=1,
#             action_noise=action_noise, 
#             target_noise_clip=0.1, 
#             verbose=1, 
#             tensorboard_log=tensorboard_log_path)

# # Train the model
# callback_max_ep = StopTrainingOnMaxEpisodes(max_episodes=5e6, verbose=1)
# pb_callback = ProgressBarCallback()
# checkpoint_callback = CheckpointCallback(
#     save_freq=25000,  # Guardar cada 100 pasos, NO episodios
#     save_path="/home/oscar/TFM",
#     name_prefix="td3_panda_v3"
# )


# callbacks = CallbackList([pb_callback, callback_max_ep, checkpoint_callback])
# model.learn(int(1e10), callback=callbacks)

# # Save the resulting model
# model.save("TD3-SB3_v2")