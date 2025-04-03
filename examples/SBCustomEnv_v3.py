import os
import sys
import numpy as np
import argparse
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
        - End effector distance to ideal position: X [12], Y[13], Z [14]
        - time step [15]
        """
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        # The action space are the Cartesian Velocities (3-translations, 3-rotations)
        self.lineal_vel = 0.25
        self.angular_vel = 0.25
        low_ap_limits = np.array([-self.lineal_vel, -self.lineal_vel, -self.lineal_vel, -self.angular_vel, -self.angular_vel, -self.angular_vel], dtype=np.float16)
        up_ap_limits = np.array([self.lineal_vel, self.lineal_vel, self.lineal_vel, self.angular_vel, self.angular_vel, self.angular_vel], dtype=np.float16)
        self.action_space = Box(low=low_ap_limits, high=up_ap_limits, shape=(6,), dtype=np.float16)
        self.end_ep_cont = 0
        self.max_step = 36
    
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
            Reward(eu_dist) = 200 - np.clip((gain*obs[-1]), 0, 200)
            Reward(effort) -> If torque or force threshold is exceeded the agent is punished with -200.
            Reward(continuity) = (100/max_steps)*step_cont -> rewarding the agent for providing continuous and non-charging cinematics in causes that end the episode
            Reward(energy) = 100*energy / max_energy -> reward used to minimize using high ef cartesian velocities
        """
        eud = np.linalg.norm(obs[12:15])
        rw_effort = 0.0
        rw_dist = 0.0
        gain = 2000.0
        force_threshold = 15
        torque_threshold = 30
        max_energy = 3*self.lineal_vel + 3*self.angular_vel
        energy = np.sum(np.abs(obs[6:12]))
        rw_energy = 100*energy / max_energy
        rw_energy = np.clip(rw_energy, 0, 100)
        # Check force and torque thresholds
        force_check = np.any(np.abs(obs[0:3]) > force_threshold)
        torque_check = np.any(np.abs(obs[3:6]) > torque_threshold)
        if force_check or torque_check:
            rw_effort = 200.0
        rw_dist = 200 - np.clip((gain*eud), 0, 200)
        rw_continuity = (100/self.max_step) * step_cont
        total_rw = rw_dist+rw_continuity-rw_effort-rw_energy
        # SAC has better results with normalizer reward range between +-10 (rw/30)
        total_rw /= 30.0
        # total_rw = rw_dist-rw_effort
        print(f"rw_dist={rw_dist:.2f}, rw_continuity={rw_continuity:.2f}, rw_effort={-rw_effort:.2f}, rw_energy={-rw_energy:.2f}")
        print(f"Generated Reward: [{total_rw:.2f}]")
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
        force_check = np.any(np.abs(obs[0:3]) > force_threshold)
        torque_check = np.any(np.abs(obs[3:6]) > torque_threshold)
        if force_check or torque_check:
            done = True
            self.end_ep_cont = 0
        # Check trajectory end
        if self.end_ep_cont >= self.max_step:
            done = True
            self.end_ep_cont = 0
        return step_cont, done

    def step(self, action):
        # Generate actions dict for transmission
        action_dict = self.format_actions(action)
        # print(action_dict)
        # Get observations dict from agent and send actions dict
        _, obs, _ = self.baselines_side.stepSendActGetObs(action_dict, timeout=300)
        print("--"*30)
        print(f"Received obs:", end=" ")
        print_obs = list(obs.items())[6:12]
        for key, val in print_obs:
            print(f"{key}: {val:.3f}", end=", ")
        # Convertfrom dict to np.parray (used by sb3)
        obs_array = np.array(list(obs.values()), dtype=np.float32)
        #print(f"SB-Side:\tObservation received after step:{obs_array[:10]}\n")
        # Check end episode conditions
        step_cont, done = self.end_episode(obs_array)
        # Calculate reward based on observations 
        reward = self.calculate_reward(obs_array, step_cont)
        print("--"*30)
        if done:
            print("\nSending DONE\n")
        return obs_array, reward, done, {}

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
                learning_rate=0.0003, 
                learning_starts=1000,
                batch_size = 256,
                gamma=0.999,
                train_freq=1,
                gradient_steps=1,
                action_noise=action_noise,  
                verbose=1, 
                tensorboard_log=tensorboard_log_path)

    # Train the model
    callback_max_ep = StopTrainingOnMaxEpisodes(max_episodes=5e6, verbose=1)
    pb_callback = ProgressBarCallback()
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # save model every 50k steps
        save_path=checkpoits_path,
        name_prefix="sac_panda_v6"
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