import os
import sys
import time
import numpy as np
from stable_baselines3 import TD3, SAC
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, ProgressBarCallback, CallbackList, CheckpointCallback
from gym import Env
from gym.spaces import Box, Dict, Discrete
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from rl_spin_decoupler.spindecoupler import RLSide

class CustomEnv(Env):
    def __init__(self, port: int):
        super(CustomEnv, self).__init__()
        # Create an instance of the communication object
        self.baselines_side = RLSide(port)
        """
        Observation space is conformed by:
        - force: End effector measured force (axis X,Y,Z). [0, 1, 2]
        - torque: End effector measured torque (axis X,Y,Z). [3, 4, 5]
        - ef_vel: End effector measured Cartesian velocity (axis X,Y,Z, roll, pitch, yaw). [6, 7, 8, 9, 10, 11]
        - eu_dist: Euclidean distance between end effector position and trajectory step. [13]
        """
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)
        # The action space are the Cartesian Velocities (3-translations, 3-rotations)
        lineal_vel = 0.75
        angular_vel = 1.0
        low_ap_limits = np.array([-lineal_vel, -lineal_vel, -lineal_vel, -angular_vel, -angular_vel, -angular_vel], dtype=np.float16)
        up_ap_limits = np.array([lineal_vel, lineal_vel, lineal_vel, angular_vel, angular_vel, angular_vel], dtype=np.float16)
        self.action_space = Box(low=low_ap_limits, high=up_ap_limits, shape=(6,), dtype=np.float16)
        self.end_ep_cont = 0
    
    def reset(self):
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
        return np.array(list(obs.values()), dtype=np.float32)

    def format_actions(self, actions):
        vel_cart = ["Vx", "Vy", "Vz", "roll", "pitch", "yaw"]
        act = {}
        for i, v in enumerate(vel_cart):
            act.update({v : actions[i]})
        act.update({"noused" : 0.0})
        return act

    def calculate_reward(self, obs, step_cont):
        """
        Reward = +Reward(eu_dist) +Reward(continuity) -Reward(esfuerzo)
            Reward(eu_dist) = 100 - np.clip((gain*obs[-1]), 0, 100)
            Reward(effort) -> If torque or force threshold is exceeded the agent is punished with -100.
            Reward(continuity) = 5*step_cont -> rewarding the agent for providing continuous and non-charging cinematics in causes that end the episode
        """
        rw_effort = 0.0
        rw_dist = 0.0
        gain = 500.0
        force_threshold = 20
        torque_threshold = 40
        # Check force and torque thresholds
        force_check = np.any(np.abs(obs[0:4]) > force_threshold)
        torque_check = np.any(np.abs(obs[4:7]) > torque_threshold)
        if force_check or torque_check:
            rw_effort = 100.0
        rw_dist = 100 - np.clip((gain*obs[-1]), 0, 100)
        rw_continuity = 2 * step_cont
        total_rw = rw_dist+rw_continuity-rw_effort
        # total_rw = rw_dist-rw_effort
        print(f"\nGenerated Reward: [{total_rw}]\n")
        return total_rw

    def end_episode(self, obs):
        """
        Episode must end if the robot has completed the trajectory or the
        force or torque threshold has been exceeded.
        """
        self.end_ep_cont += 1
        step_cont = self.end_ep_cont
        done = False
        force_threshold = 48
        torque_threshold = 60
        # Check force and torque thresholds
        force_check = np.any(np.abs(obs[0:4]) > force_threshold)
        torque_check = np.any(np.abs(obs[4:7]) > torque_threshold)
        if force_check or torque_check:
            done = True
            self.end_ep_cont = 0
            print("\nSending DONE\n")
        # Check trajectory end
        if self.end_ep_cont >= 24:
            done = True
            self.end_ep_cont = 0
            print("\nSending DONE\n")
        return step_cont, done

    def step(self, action):
        print("SB-Side:\tGoing on a new step...\n")
        ###########################################################
        # actionn = np.zeros(shape=(7,), dtype=np.float32)
        # if self.end_ep_cont < 6: # Move through Y-Z axis
        #     actionn[0] = 0.05 # Vel X
        #     actionn[1] = 0.0 # Vel Y
        #     actionn[2] = 0.0 # Vel Z
        # elif (self.end_ep_cont >= 6 and self.end_ep_cont < 12): # Move through Y-Z axis
        #     actionn[0] = 0.0 # Vel X
        #     actionn[1] = -0.05 # Vel Y
        #     actionn[2] = 0.0 # Vel Z
        # elif (self.end_ep_cont >= 12 and self.end_ep_cont < 18): # Move through X axis
        #     actionn[0] = -0.05 # Vel X
        #     actionn[1] = 0.0 # Vel Y
        #     actionn[2] = 0.0 # Vel Z
        # elif (self.end_ep_cont >= 18 and self.end_ep_cont < 24): # Backwards through X axis
        #     actionn[0] = 0.0 # Vel X
        #     actionn[1] = 0.05 # Vel Y
        #     actionn[2] = 0.0 # Vel Z
        # else:
        #     actionn[0] = 0.0 # Vel X
        #     actionn[1] = 0.0 # Vel Y
        #     actionn[2] = 0.0 # Vel Z
        ###########################################################
        # Generate actions dict for transmission
        action_dict = self.format_actions(action)
        # print(action_dict)
        # Get observations dict from agent and send actions dict
        _, obs, _ = self.baselines_side.stepSendActGetObs(action_dict, timeout=300)
        #print(obs)
        # Convertfrom dict to np.parray (used by sb3)
        obs_array = np.array(list(obs.values()), dtype=np.float32)
        #print(f"SB-Side:\tObservation received after step:{obs_array[:10]}\n")
        # Check end episode conditions
        step_cont, done = self.end_episode(obs_array)
        # Calculate reward based on observations 
        reward = self.calculate_reward(obs_array, step_cont)
        return obs_array, reward, done, {}

# Create custom enviroment
env = CustomEnv(port=49055)

# Some Gaussian noise on acctions for safer sim-world transfer
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# Initilize de model
tensorboard_log_path = "/home/oscar/TFM/train_logs"
model = TD3("MlpPolicy", env,
            learning_rate=0.002, 
            learning_starts=1000,
            batch_size = 256,
            train_freq=1,
            gradient_steps=1,
            action_noise=action_noise, 
            target_noise_clip=0.1, 
            verbose=1, 
            tensorboard_log=tensorboard_log_path)

# Train the model
callback_max_ep = StopTrainingOnMaxEpisodes(max_episodes=5e6, verbose=1)
pb_callback = ProgressBarCallback()
checkpoint_callback = CheckpointCallback(
    save_freq=25000,  # Guardar cada 100 pasos, NO episodios
    save_path="/home/oscar/TFM",
    name_prefix="td3_panda_v3"
)


callbacks = CallbackList([pb_callback, callback_max_ep, checkpoint_callback])
model.learn(int(1e10), callback=callbacks)

# Save the resulting model
model.save("TD3-SB3_v2")


########################
#         SAC          #
########################


# tensorboard_log_path = "/home/oscar/TFM/train_logs"
# model = SAC("MlpPolicy", env,
#             learning_rate=0.0003, 
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
#     name_prefix="sac_panda_v3"
# )


# callbacks = CallbackList([pb_callback, callback_max_ep, checkpoint_callback])
# model.learn(int(1e10), callback=callbacks)

# # Save the resulting model
# model.save("SAC-SB3_v2")

