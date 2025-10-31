import os
import sys
import argparse
import random
import csv
import dm_env
import numpy as np
import time
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

    def __init__(self, spec: specs.BoundedArray, portbaselinespart:int, time_step: float) -> None:
        self._spec = spec
        # Initialize variables
        self.time_state = 0.1
        self.init = True
        # Flag to indicate if the agent is waiting for RL commands
        self._waitingforrlcommands = True
        # Time in seconds that the agent will wait before receiving the next action
        self.step_time = time_step
        # Type of trajectory to follow
        # self.traj_dict = ["ah-square", "ah-circle", "h-square", "h-triangle", "h-circle"]
        self.traj_dict = ["ah-square"]
        self.episode_cont = 0
        self.action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        # Create an instance of the communication object and start communication
        self.agent_side = AgentSide(BaseCommPoint.get_ip(), portbaselinespart)
        self.points_traj = 5000
        # Magnitude thresholds for normalization
        self.max_vel = 0.25
        self.max_force_threshold = 25
        self.max_torque_threshold = 55
        self.max_dist = 0.15

    def pass_args(self, env: Environment, joint_names):
        self.env = env
        self.joint_names = joint_names

    def calculate_trajectory(self, ef_position, uc="h-square"):
        """
        Calculate the trajectory to follow based on the end effector position and the trajectory type.
        The trajectory is calculated in a 3D space, with the Z axis being the vertical axis.
        The trajectory is calculated in a way that the end effector will follow the trajectory
        with a constant velocity.
        Args:
            ef_position: The end effector position in the form of a list [x, y, z].
            uc: The trajectory type to follow. It can be "ah-square", "h-square", "ah-triangle", "h-triangle", "h-circle".
        """
        state = 1
        cont = 1
        cycles = 30
        posX = ef_position[0]; posY = ef_position[1] 
        posZ = ef_position[2] # + random.uniform(-0.1, 0.1)
        constant_vel = 0.05 # + random.uniform(-0.01, 0.01)
        incT = 0.1
        n_points = self.points_traj
        self.trajectory = np.zeros((n_points, 3), dtype=np.float32)
        self.trajectory[0] = [posX, posY, posZ]
        print(f"Computing trajectory {uc} with constant velocity: {constant_vel:.3f} at [{posX:.2f}, {posY:.2f}, {posZ:.2f}]")
        #######################################################
        if uc == "ah-square":
            for step in range(1, n_points): # Ideal square trajectory
                if state == 0: # Move through Z axis
                    posZ += incT * constant_vel
                    cont += 1
                    if cont >= cycles:
                        state = 1
                        cont = 0
                elif state == 1: # Move through X axis
                    posX += -incT * constant_vel
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
                    posX += incT * (constant_vel)
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
        elif uc == "h-square":
            for step in range(1, n_points): # Ideal square trajectory
                if state == 0: # Move through Z axis
                    posZ += -incT * constant_vel
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
                    posX += -incT * constant_vel
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
        elif uc == "ah-triangle":
            for step in range(1, n_points): # Ideal square trajectory
                if state == 0: # Move through Z axis
                    posZ += incT * constant_vel
                    cont += 1
                    if cont >= cycles:
                        state = 1
                        cont = 0
                elif state == 1: # Move through X axis
                    posX += -incT * constant_vel
                    cont += 1
                    if cont >= cycles:
                        state = 2
                        cont = 0
                elif state == 2: # Move through Y axis
                    posY += incT * (-1*constant_vel)
                    posX += incT * (0.5*constant_vel)
                    cont += 1
                    if cont >= cycles:
                        state = 3
                        cont = 0
                elif state == 3: # Move through X axis
                    posY += incT * constant_vel
                    posX += incT * (0.5*constant_vel)
                    cont += 1
                    if cont >= cycles:
                        state = 1
                        cont = 0
                self.trajectory[step] = [posX, posY, posZ]
        #######################################################
        elif uc == "h-triangle":
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
                    posX += -incT * (0.5*constant_vel)
                    cont += 1
                    if cont >= cycles:
                        state = 3
                        cont = 0
                elif state == 3: # Move through X axis
                    posY += incT * constant_vel
                    posX += -incT * (0.5*constant_vel)
                    cont += 1
                    if cont >= cycles:
                        state = 1
                        cont = 0
                self.trajectory[step] = [posX, posY, posZ]
        #######################################################
        elif uc == "h-circle":
            r = 0.07
            self.trajectory = []
            # Uniformely distributed angles
            thetas = np.linspace(-np.pi/2, 1.5*np.pi, 120)  # h

            laps = n_points // 120
            if (n_points % 120) != 0:
                laps += 1
            for i in range(laps):
                # Circular coordinates on XY plane
                x = posX + r * np.cos(thetas)
                y = posY - r * np.sin(thetas) - r
                z = np.full_like(x, posZ)  # Z constant

                lap_traj = np.stack((x, y, z), axis=1)
                self.trajectory.append(lap_traj)

            self.trajectory = np.vstack(self.trajectory)
        #######################################################
        elif uc == "ah-circle":
            r = 0.07
            self.trajectory = []
            # Uniformely distributed angles
            thetas = np.linspace(-np.pi/2, (-5/2)*np.pi, 120)  # ah

            laps = n_points // 120
            if (n_points % 120) != 0:
                laps += 1
            for i in range(laps):
                # Circular coordinates on XY plane
                x = posX + r * np.cos(thetas)
                y = posY - r * np.sin(thetas) - r
                z = np.full_like(x, posZ)  # Z constant

                lap_traj = np.stack((x, y, z), axis=1)
                self.trajectory.append(lap_traj)

            self.trajectory = np.vstack(self.trajectory)
        else:
            print("\n\tERROR DURING TRAJECTORY CALCULATION!!!\n")

    def format_obs(self, force, torque, vel_ef, dist, eu_dist, follow_vector):
        """
        Observation space is conformed by:
        - force: End effector measured force (axis X,Y,Z).
        - torque: End effector measured torque (axis X,Y,Z).
        - ef_vel: End effector measured Cartesian velocity (axis X,Y,Z, roll, pitch, yaw).
        - dist: Distance between end effector position and ideal trajectory position.
        - eu_dist: euclidean distance between end effector position and ideal trajectory position.
        - follow_vector: Vector that indicates the direction of the trajectory at the current step.
        Args:
            force: End effector measured force (axis X,Y,Z).
            torque: End effector measured torque (axis X,Y,Z).
            vel_ef: End effector measured Cartesian velocity (axis X,Y,Z, roll, pitch, yaw).
            dist: Distance between end effector position and ideal trajectory position.
            eu_dist: Euclidean distance between end effector position and ideal trajectory position.
            follow_vector: Vector that indicates the direction of the trajectory at the current step.
        Returns:
            obs: Observation dictionary with the normalized values.
            invalid_value: Boolean indicating if there is an invalid value in the observation.
        """
        # Normalize values
        force = np.clip((force/self.max_force_threshold), -1.0, 1.0)
        torque = np.clip((torque/self.max_torque_threshold), -1.0, 1.0)
        vel_ef = np.clip((vel_ef/self.max_vel), -1.0, 1.0)
        dist = np.clip((dist/self.max_dist), -1.0, 1.0)
        eu_dist = np.clip((eu_dist/self.max_dist), -1.0, 1.0)
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
                    'euclidean_dist': eu_dist,
                    'X_follow_vector': follow_vector[0], # Already normalized
                    'Y_follow_vector': follow_vector[1],
                    'Z_follow_vector': follow_vector[2]}
        # Check Inf or NaN posible values
        invalid_value = False
        for key, value in obs.items():
            if np.isnan(value) or np.isinf(value):
                invalid_value = True
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

    def calculate_dist(self, step, ef_position):
        """Calculate the distance between the end effector position and the ideal trajectory position.
        Args:
            step: The current step in the trajectory.
            ef_position: The end effector position in the form of a list [x, y, z].
        Returns:
            dist: The distance between the end effector position and the ideal trajectory position.
        """
        # Timestep advance 0.1 at a time, to get index is mandatory multiply the timestep by 10
        ideal_position = self.trajectory[int(((step*10)-1)%self.points_traj)]
        # Calculate distance 
        dist = ideal_position - ef_position
        return dist

    def calculate_eu_dist(self, step, ef_position):
        """Calculate the euclidean distance between the end effector position and the ideal trajectory position.
        Args:
            step: The current step in the trajectory.
            ef_position: The end effector position in the form of a list [x, y, z].
        Returns:
            eud: The euclidean distance between the end effector position and the ideal trajectory position.
        """
        # Timestep advance 0.1 at a time, to get index is mandatory multiply the timestep by 10
        ideal_position = self.trajectory[int(((step*10)-1)%self.points_traj)]
        # Calculate euclidean distance 
        eud = np.linalg.norm(ideal_position - ef_position)
        return eud
    
    def calculate_follow_vector(self, step):
        """Calculate the follow vector that indicates the direction of the trajectory at the current step.
        Args:
            step: The current step in the trajectory.
        Returns:
            follow_vector: The vector that indicates the direction of the trajectory at the current step.
        """
        # Timestep advance 0.1 at a time, to get index is mandatory multiply the timestep by 10
        current_ideal_position = self.trajectory[int(((step*10)-1)%self.points_traj)]
        next_ideal_position = self.trajectory[int(((step*10))%self.points_traj)]
        follow_vector = next_ideal_position - current_ideal_position
        follow_vector = follow_vector / np.linalg.norm(follow_vector)
        return follow_vector

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
            while whattodo is None:
                whattodo = self.agent_side.readWhatToDo()
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
        agent = Agent(env.action_spec(), args.port, args.time_step)
        agent.pass_args(env, joint_names)
        # Run the environment and agent inside the GUI.
        # app = utils.ApplicationWithPlot(width=1440, height=860)
        # app.launch(env, policy=agent.step)
        run_loop.run(env, agent, [], max_steps=1e10, real_time=True)