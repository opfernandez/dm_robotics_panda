import os
import sys
import time
import signal
import numpy as np
from gym import Env
from gym.spaces import Box
import argparse
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes, ProgressBarCallback, CallbackList
from stable_baselines3.common.noise import NormalActionNoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from rl_spin_decoupler.spindecoupler import RLSide, AgentSide

class CustomEnv(Env):
    def __init__(self, port: int):
        super(CustomEnv, self).__init__()
        # Create an instance of the communication objectBaselinesSide
        self.baselines_side = RLSide(port)
        self.observation_space = Box(low=-1000, high=1000, shape=(1,), dtype=np.int16)
        self.action_space = Box(low=-1e4, high=1e4, shape=(1,), dtype=np.float16)
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self):
        print("\n[INFO] Ctrl+C detectado. Saliendo limpiamente...")
        sys.exit(0)  # Termina el programa de forma controlada
    
    def reset(self):
        obs = self.baselines_side.resetGetObs(timeout=360)
        print(f"SB-Side:\tObservation received after RESET:{obs}\n")
        return np.random.rand()

    def step(self, action):
        t = time.time()
        action_dict = {"SB-Env Hello time:": t}
        lat,obs,_ = self.baselines_side.stepSendActGetObs(action_dict, timeout=300.0)
        if not obs:  # Verifica si obs es None o vacío
            print("\n\nhola 2\n\n")
            raise RuntimeError("Error: No se recibió ninguna observación en step()")
        print(f"SB-Side:\tObservation received after STEP: {obs}")
        obs_array = np.array(list(obs.values()), dtype=np.int8)
        reward = np.random.rand()
        done = False
        if obs_array[0] >= 10:
            done = True
        return obs_array, reward, done, {}


class Agent():
    def __init__(self, ipbaselinespart:str, portbaselinespart:int):
        # Create an instance of the communication object and start communication
        self.agent_side = AgentSide(ipbaselinespart, portbaselinespart)
        self.count = 0
        self.start_ep = time.time()
        self.elapsed = 1
        self.init = True
        self._waitingforrlcommands = True
        signal.signal(signal.SIGINT, self.handle_sigint)

    def handle_sigint(self):
        print("\n[INFO] Ctrl+C detectado. Saliendo limpiamente...")
        self.agent_side.stopComms()
        sys.exit(0)  # Termina el programa de forma controlada

    def reset(self):
        self._waitingforrlcommands = True
        self.count = 0
        self.start_ep = time.time()
        obs = {"Agent count:": self.count}
        self.agent_side.resetSendObs(obs)

    def step(self):
        while True:
            t = time.time()
            if not self._waitingforrlcommands:
                if (t - self.start_ep) > self.elapsed or self.init:
                    self.count += 1
                    obs = {"Agent count:": self.count}
                    self.agent_side.stepSendObs(obs) # RL was waiting for this; no reward is actually needed here
                    self._waitingforrlcommands = True
            else:
                # Receive the indicator of what to do
                whattodo = self.agent_side.readWhatToDo()
                if whattodo is not None:
                    # Select the case
                    if whattodo[0] == AgentSide.WhatToDo.REC_ACTION_SEND_OBS:
                        self.init = False
                        sb_action = whattodo[1]
                        lat = t - self.start_ep
                        self.start_ep = time.time()
                        self._waitingforrlcommands = False # from now on, we are waiting to execute the action
                        self.agent_side.stepSendLastActDur(lat)
                        print(f"\nAgent-Side:\tReceived STEP action:  {sb_action}\n")
                    elif whattodo[0] == AgentSide.WhatToDo.RESET_SEND_OBS:
                        self.init = False
                        print("\nRESETTING ENV TO START NEW EPISODE...\n")
                        self.reset()
                    elif whattodo[0] == AgentSide.WhatToDo.FINISH:
                        # Finish training
                        print("Experiment finished.")
                        sys.exit()
                    else:
                        raise(ValueError("Unknown indicator data"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para comprobar parámetros de entrada.")
    parser.add_argument("type", choices=["agent", "env"], help="Debe ser 'agent' o 'env'.")
    args = parser.parse_args()

    port = 49053
    if args.type == "agent":
        agent = Agent('192.168.0.18', port)
        agent.step()
    elif args.type == "env":
        env = CustomEnv(port=port)
        # Some Gaussian noise on acctions for safer sim-world transfer
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        # Initilize de model
        tensorboard_log_path = "/home/oscar/TFM/train_logs"
        model = TD3("MlpPolicy", env, action_noise=action_noise, target_noise_clip=0.1, verbose=1, tensorboard_log=tensorboard_log_path)

        # Train the model
        callback_max_ep = StopTrainingOnMaxEpisodes(max_episodes=10000, verbose=1)
        pb_callback = ProgressBarCallback()
        callbacks = CallbackList([pb_callback, callback_max_ep])
        model.learn(int(1e10), callback=callbacks)
    else:
        print("Wrong input, must be [agent] or [env]")
