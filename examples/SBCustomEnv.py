import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes
from gym import Env
from gym.spaces import Box, Dict, Discrete

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from rl_spin_decoupler import BaselinesSide

class CustomEnv(Env):
    def __init__(self, port: int):
        super(CustomEnv, self).__init__()
        # Create an instance of the communication object
        self.baselines_side = BaselinesSide(port)
        # Observation space is a 51-position np.array formed by:
        # obs[0] -> timestep
        # obs[1:10] -> franka robot joint configuration
        # obs[10:48] -> myoarm joint configuration
        # obs[48:51] -> franka robot end effector forces
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(51,), dtype=np.float32)
        # The action space are the Cartesian Velocities (3-translations, 3-rotations)
        self.action_space = Box(low=-1.0, high=-1.0, shape=(6,), dtype=np.float32)
    
    def reset(self):
        print("SB-Side:\tGoing on a new reset...\n")
        # Communicate SB3 loop with Agent loop using sockets 
        obs = self.baselines_side.resetGetObs()
        print(f"SB-Side:\tObservation received after reset:{obs}\n")
        return np.array(list(obs.values()), dtype=np.float32)

    def format_actions(self, actions):
        vel_cart = ["Vx", "Vy", "Vz", "Rotx", "Roty", "Rotz"]
        act = {}
        for i, v in enumerate(vel_cart):
            act.update({v : actions[i]})
        return act

    def step(self, action):
        print("SB-Side:\tGoing on a new step...\n")
        # Generate actions dict for transmission
        action_dict = self.format_actions(action)
        # Get observations dict from agent and send actions dict
        obs = self.baselines_side.stepGetObsSendAct(action_dict)
        # Convertfrom dict to np.parray (used by sb3)
        obs_array = np.array(list(obs.values()), dtype=np.float32)
        print(f"SB-Side:\tObservation received after step:{obs_array[:10]}\n")
        # Set a random reward 
        reward = np.random.rand() # Value between 0-1
        # Get timestep and set done flag after 5 seconds
        timestep = obs_array[0]
        done = False
        if timestep > 4.99:
            done = True
        return obs_array, reward, done, {}

# Create custom enviroment
env = CustomEnv(port=49053)

# Initilize de model
model = PPO("MlpPolicy", env, learning_rate=0.0005, n_steps=2048, batch_size=128, 
                n_epochs=25, gamma=0.99, gae_lambda=0.95, clip_range=0.2, 
                clip_range_vf=None, ent_coef=0.025, vf_coef=0.5, use_sde=True, sde_sample_freq=-1,  
                verbose=1, device='cuda:0')

# Train the model
callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=3, verbose=1)
model.learn(int(1e10), callback=callback_max_episodes)

# Save the resulting model
model.save("PPO-SB3")
