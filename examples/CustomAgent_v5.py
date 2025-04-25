"""
Imports a custom enviroment from a XML file.
Produces a Cartesian motion using the Cartesian actuation mode.
"""
import os
import sys
import argparse
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
    current observations and rewards.
    """

    def __init__(self, spec: specs.BoundedArray, portbaselinespart:int) -> None:
        self._spec = spec
        self.state = 0
        self.time_state = 0.1
        self.env_reset = True
        self.init = True
        self._waitingforrlcommands = True
        self.step_time = 0.35 # segundos que va a pasar el agente realizando un mismo movimiento
        # self.traj_dict = ["square", "triangle", "circle"]
        self.traj_dict = ["square", "triangle"]
        self.episode_cont = 0
        self.action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        # Create an instance of the communication object and start communication
        self.agent_side = AgentSide(BaseCommPoint.get_ip(), portbaselinespart)
        self.points_traj = 5000

    def pass_args(self, env: Environment, joint_names):
        self.env = env
        self.joint_names = joint_names

    # def calculate_trajectory(self, ef_position):
    #     state = 1
    #     cont = 1
    #     cycles = 30
    #     posX = ef_position[0]; posY = ef_position[1]; posZ = ef_position[2]
    #     constant_vel = 0.05
    #     incT = 0.1
    #     self.trajectory = np.zeros((2000, 3), dtype=np.float32)
    #     self.trajectory[0] = [posX, posY, posZ]
    #     for step in range(1, 2000): # Ideal square trajectory
    #         if state == 0: # Move through Z axis
    #             posZ += incT * constant_vel
    #             cont += 1
    #         if cont >= cycles:
    #             state = 1
    #             cont = 0
    #         elif state == 1: # Move through X axis
    #             posX += incT * constant_vel
    #             cont += 1
    #         if cont >= cycles:
    #             state = 2
    #             cont = 0
    #         elif state == 2: # Move through Y axis
    #             posY += incT * (-1*constant_vel)
    #             cont += 1
    #         if cont >= cycles:
    #             state = 3
    #             cont = 0
    #         elif state == 3: # Move through X axis
    #             posX += incT * (-1*constant_vel)
    #             cont += 1
    #         if cont >= cycles:
    #             state = 4
    #             cont = 0
    #         elif state == 4: # Move through Y axis
    #             posY += incT * constant_vel
    #             cont += 1
    #         if cont >= cycles:
    #             state = 1
    #             cont = 0
    #         self.trajectory[step] = [posX, posY, posZ]
    #     #return trajectory

    def calculate_trajectory(self, ef_position, uc="square"):
        state = 1
        cont = 1
        cycles = 30
        posX = ef_position[0]; posY = ef_position[1]; posZ = ef_position[2]
        constant_vel = 0.05
        incT = 0.1
        n_points = self.points_traj
        #######################################################
        if uc == "square":
            self.trajectory = np.zeros((n_points, 3), dtype=np.float32)
            self.trajectory[0] = [posX, posY, posZ]
            for step in range(1, n_points): # Ideal square trajectory
                if state == 0: # Move through Z axis
                    posZ += incT * constant_vel 
                    cont += 1
                    if cont >= cycles:
                        state = 1
                        cont = 0
                elif state == 1: # Move through X axis
                    posX += incT * constant_vel
                    cont += 1
                    if cont >= cycles:
                        state = 2
                        cont = 0
                elif state == 2: # Move through Y axis
                    posY += incT * (-1*constant_vel)
                    cont += 1
                    if cont >= cycles:
                        state = 3
                        cont = 0
                elif state == 3: # Move through X axis
                    posX += incT * (-1*constant_vel)
                    cont += 1
                    if cont >= cycles:
                        state = 4
                        cont = 0
                elif state == 4: # Move through Y axis
                    posY += incT * constant_vel
                    cont += 1
                    if cont >= cycles:
                        state = 1
                        cont = 0
                self.trajectory[step] = [posX, posY, posZ]
        #######################################################
        elif uc == "triangle":
            self.trajectory = np.zeros((n_points, 3), dtype=np.float32)
            self.trajectory[0] = [posX, posY, posZ]
            for step in range(1, n_points): # Ideal square trajectory
                if state == 0: # Move through Z axis
                    posZ += incT * constant_vel
                    cont += 1
                    if cont >= cycles:
                        state = 1
                        cont = 0
                elif state == 1: # Move through X axis
                    posX += incT * constant_vel
                    cont += 1
                    if cont >= cycles:
                        state = 2
                        cont = 0
                elif state == 2: # Move through Y axis
                    posY += incT * (-1*constant_vel)
                    posX += incT * (-0.5*constant_vel)
                    cont += 1
                    if cont >= cycles:
                        state = 3
                        cont = 0
                elif state == 3: # Move through X axis
                    posY += incT * constant_vel
                    posX += incT * (-0.5*constant_vel)
                    cont += 1
                    if cont >= cycles:
                        state = 1
                        cont = 0
                self.trajectory[step] = [posX, posY, posZ]
        #######################################################
        # elif uc == "circle":
        #     r = 0.075
        #     self.trajectory = []
        #     # Uniformely distributed angles
        #     thetas = np.linspace(0, 2*np.pi, 120)

        #     laps = n_points // 120
        #     if  (n_points % 120) != 0:
        #         laps += 1
        #     for i in range(laps):
        #         # Circular coordinates on XY plane
        #         x = posX + r * np.cos(thetas)
        #         y = posY - r * np.sin(thetas) - r
        #         z = np.full_like(x, posZ)  # Z constant

        #         lap_traj = np.stack((x, y, z), axis=1)
        #         self.trajectory.append(lap_traj)

        #     self.trajectory = np.vstack(self.trajectory)
        else:
            print("\n\tERROR DURING TRAJECTORY CALCULATION!!!\n")

    def format_obs(self, force, torque, vel_ef, dist, eu_dist):
        """
        Observation space is conformed by:
        - force: End effector measured force (axis X,Y,Z).
        - torque: End effector measured torque (axis X,Y,Z).
        - ef_vel: End effector measured Cartesian velocity (axis X,Y,Z, roll, pitch, yaw).
        - dist: Distance between end effector position and ideal trajectory position.
        - eu_dist: euclidean distance between end effector position and ideal trajectory position.
        """
        # Create observation dict
        obs = {'F_wristX': force[0],
                    'F_wristY': force[1],
                    'F_wristZ': force[2],
                    'T_wristX': torque[0],
                    'T_wristY': torque[1],
                    'T_wristZ': torque[2],
                    'Vx': vel_ef[0],
                    'Vy': vel_ef[1],
                    'Vz': vel_ef[2],
                    'roll': vel_ef[3],
                    'pitch': vel_ef[4],
                    'yaw': vel_ef[5],
                    'X_dif': dist[0],
                    'Y_dif': dist[1],
                    'Z_dif': dist[2],
                    'euclidean_dist': eu_dist}
        # Check Inf or NaN posible values
        invalid_value = False
        for key, value in obs.items():
            if np.isnan(value) or np.isinf(value):
                invalid_value = True
        return obs, invalid_value

    def save_data(self, file_name, data, mode):
        with open(file_name, mode=mode, newline="") as file:
            writer = csv.writer(file)
            writer.writerow(data)

    def calculate_dist(self, step, ef_position):
        # Timestep advance 0.1 at a time, to get index is mandatory multiply the timestep by 10
        # % 480 is just a safety trick, it should never be effective, because step should not exceed 48
        ideal_position = self.trajectory[int(((step*10)-1)%self.points_traj)]
        # Calculate distance 
        dist = ideal_position - ef_position
        return dist

    def calculate_eu_dist(self, step, ef_position):
        # Timestep advance 0.1 at a time, to get index is mandatory multiply the timestep by 10
        # % 480 is just a safety trick, it should never be effective, because step should not exceed 48
        ideal_position = self.trajectory[int(((step*10)-1)%self.points_traj)]
        # Calculate euclidean distance 
        eud = np.linalg.norm(ideal_position - ef_position)
        return eud

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """
        Computes velocities in the x/y plane parameterized in time.
        """
        # keys = timestep.observation.keys()
        # print(keys) -> Result:
        # ['panda_joint_pos', 'panda_joint_vel', 'panda_joint_torques', 'panda_tcp_pos', 
        # 'panda_tcp_quat', 'panda_tcp_rmat', 'panda_tcp_pose', 'panda_tcp_vel_world', 
        # 'panda_tcp_vel_relative', 'panda_tcp_pos_control', 'panda_tcp_quat_control', 
        # 'panda_tcp_rmat_control', 'panda_tcp_pose_control', 'panda_tcp_vel_control', 
        # 'panda_force', 'panda_torque', 'panda_gripper_width', 'panda_gripper_state', 
        # 'panda_twist_previous_action', 'time']
        time_t = timestep.observation['time'][0]
        force = timestep.observation['panda_force']
        torque = timestep.observation['panda_torque']
        vel_ef = timestep.observation['panda_tcp_vel_world']
        ef_position = [timestep.observation['panda_tcp_pose'][0], # X
                        timestep.observation['panda_tcp_pose'][1], # Y
                        timestep.observation['panda_tcp_pose'][2]] # Z
        # Calculate trajectory on first step:
        if self.init:
            index_traj = ( self.episode_cont // 100) % len(self.traj_dict)
            print(f"Current trajectory is [{self.traj_dict[index_traj]}]")
            self.calculate_trajectory(ef_position, self.traj_dict[index_traj])
            self.init = False
        eu_dist = self.calculate_eu_dist(time_t, ef_position)
        dist = self.calculate_dist(time_t, ef_position)

        ### COMMUNICATE WITH STABLE-BASELINES ###
        if not self._waitingforrlcommands:
            if self.env_reset or ((time_t - self.time_state) >= self.step_time):
                # Create observation dict
                obs, invalid_value = self.format_obs(force, torque, vel_ef, dist, eu_dist)
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
                        obs, invalid_value = self.format_obs(force, torque, vel_ef, dist, eu_dist)
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
                # self.time_state = time_t
        return self.action

    def reset(self):
        self.episode_cont += 1
        self._waitingforrlcommands = True
        self.env_reset = True
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
        self.calculate_trajectory(ef_position)
        # calculate euclidean dist and exis-dist
        eu_dist = self.calculate_eu_dist(time_t, ef_position)
        dist = self.calculate_dist(time_t, ef_position)
        # reset time-state (for state machine)
        self.time_state = time_t
        # Create observation dict
        obs, invalid_value = self.format_obs(force, torque, vel_ef, dist, eu_dist)
        # Just send observation dict
        self.agent_side.resetSendObs(obs)
        #time.sleep(0.1)
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

    # physics = mjcf.Physics.from_mjcf_model(panda_env._arena.mjcf_model)
    # panda_env.robots['panda'].position_gripper(position=np.array([-0.3, 0.25, 0.4]),
    #                                                     quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
    #                                                     physics=physics)

    # TODO: emplear native attachmet para el myoarm y sus includes en vez de un XML kilométrico
    # Form absolute path to the XML file
    XML_ARM_PATH = os.path.join(script_dir, "../../models/myo_sim/arm/myoarmPanda.xml")
    
    myoarm = MyoArm(xml_path=XML_ARM_PATH)
    panda_env._arena.attach(myoarm)

    bodies = panda_env._arena.mjcf_model.worldbody.find_all('body')
    # print(f"  ********\n\n {bodies} \n\n  ********")
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
    # Establecer restricción entre la muñeca y el efector final para que se muevan
    equality_constraint = panda_env._arena.mjcf_model.equality.add(
        'weld',
        body1=ee,  
        body2=wrist, 
        relpose=[0.025, 0.0, 0.115, # Posición de la muñeca desde el efector del robot
                0.0, 0.87, -0.50, 0.0],  # Rotación de la muñeca respecto del efector del robot (180º,0º,60º)
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