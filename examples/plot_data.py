import argparse
import csv
import os
import re
import matplotlib.pyplot as plt
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
files_path = os.path.join(script_dir, "../../data/")

def plot_trajectory(ts="0.1"):
    """ Reads trajectory data from a CSV and plots it over the XY plane. """
    # Read trained model followed trajectory
    posX, posY = [], []
    file_model = files_path + f"panda_traj_model_ts_{ts}.csv"
    with open(file_model, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            posX.append(float(row[0]))  # X value
            posY.append(float(row[1]))  # Y value

    # Read ideal trajectory
    with open(files_path + "panda_ideal_traj_model.csv", "r") as f:
        line = f.read()
    # Extract all matches of the form [x y z]
    matches = re.findall(r"\[([^\]]+)\]", line)
    # Convert matches to a numpy array
    xy_array = np.array([
        list(map(float, m.split()[:2]))
        for m in matches
    ])

    x_ideal = xy_array[:, 0]
    y_ideal = xy_array[:, 1]

    # Plotting the trajectory over the XY plane
    plt.figure(figsize=(8, 6))
    plt.plot(posX, posY, marker='o', linestyle='-', color='r', label="Model Trajectory")
    plt.plot(x_ideal, y_ideal, marker='o', linestyle='-', color='b', label="Ideal Trajectory")

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trajectory over XY Plane")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_forces_from_csv(filename="panda_forces_model.csv", timestep=0.1, title=None):
    """ Reads forces Fx, Fy, Fz from a CSV and plots them over time. """
    
    # Lists to store the forces
    fx_vals, fy_vals, fz_vals = [], [], []
    
    # Read data from the CSV file
    csvpath = files_path + filename
    with open(csvpath, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            fx_vals.append(float(row[0]))  # Fx
            fy_vals.append(float(row[1]))  # Fy
            fz_vals.append(float(row[2]))  # Fz

    # Create the timesteps with a step of 0.1s
    timesteps = np.arange(0, len(fx_vals) * timestep, timestep)

    # Plot the forces
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, fx_vals, label="Fx", color="r")
    plt.plot(timesteps, fy_vals, label="Fy", color="g")
    plt.plot(timesteps, fz_vals, label="Fz", color="b")

    # Labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Force (N)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_torques_from_csv(filename="panda_torques_model.csv", timestep=0.1, title=None):
    """ Reads torques Tx, Ty, Tz from a CSV and plots them over time. """
    
    # Lists to store the torques
    tx_vals, ty_vals, tz_vals = [], [], []
    
    # Read data from the CSV file
    csvpath = files_path + filename
    with open(csvpath, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            tx_vals.append(float(row[0]))  # Tx
            ty_vals.append(float(row[1]))  # Ty
            tz_vals.append(float(row[2]))  # Tz

    # Create the timesteps with a step of 0.1s
    timesteps = np.arange(0, len(tx_vals) * timestep, timestep)

    # Plot the torques
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, tx_vals, label="Tx", color="r")
    plt.plot(timesteps, ty_vals, label="Ty", color="g")
    plt.plot(timesteps, tz_vals, label="Tz", color="b")

    # Labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Torque (Nm)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_joint_torques_from_csv(filename="panda_joint_torques_model.csv", timestep=0.1):
    """ Reads joint torques from a CSV and plots them over time. """
    
    # Lists to store the joint torques
    jt1_vals, jt2_vals, jt3_vals, jt4_vals, jt5_vals, jt6_vals, jt7_vals = [], [], [], [], [], [], []
    
    # Read data from the CSV file
    csvpath = files_path + filename
    with open(csvpath, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            jt1_vals.append(float(row[0]))  
            jt2_vals.append(float(row[1]))  
            jt3_vals.append(float(row[2]))  
            jt4_vals.append(float(row[3]))  
            jt5_vals.append(float(row[4]))  
            jt6_vals.append(float(row[5]))  
            jt7_vals.append(float(row[6]))  

    # Create timesteps
    timesteps = np.arange(0, len(jt1_vals) * timestep, timestep)

    # Plot the joint torques
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, jt1_vals, label="Joint 1 Torque", color="r")
    plt.plot(timesteps, jt2_vals, label="Joint 2 Torque", color="g")
    plt.plot(timesteps, jt3_vals, label="Joint 3 Torque", color="b")
    plt.plot(timesteps, jt4_vals, label="Joint 4 Torque", color="y")
    plt.plot(timesteps, jt5_vals, label="Joint 5 Torque", color="m")
    plt.plot(timesteps, jt6_vals, label="Joint 6 Torque", color="c")
    plt.plot(timesteps, jt7_vals, label="Joint 7 Torque", color="k")

    # Labels and title
    plt.xlabel("Time (s)")
    plt.ylabel("Articular Torque (Nm)")
    plt.title("Evolution of Joint Torques Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_vel_ef_from_csv(filename="panda_vel_ef_model.csv", timestep=0.1):
    """ Reads end effector velocities Vx, Vy, Vz from a CSV and plots them over time. """
    
    # Lists to store the velocities
    vx_vals, vy_vals, vz_vals = [], [], []
    
    # Read data from the CSV file
    csvpath = files_path + filename
    with open(csvpath, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            vx_vals.append(float(row[0]))  # Fx
            vy_vals.append(float(row[1]))  # Fy
            vz_vals.append(float(row[2]))  # Fz

    # Create the timesteps with a step of 0.1s
    timesteps = np.arange(0, len(vx_vals) * timestep, timestep)

    # Plot the end effector velocities
    plt.figure(figsize=(8, 5))

    plt.plot(timesteps, vx_vals, label="Vx", color="r")
    plt.plot(timesteps, vy_vals, label="Vy", color="g")
    plt.plot(timesteps, vz_vals, label="Vz", color="b")

    # Labels and title
    plt.ylim(-0.12, 0.12)
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("End Effector Velocities Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_force_over_trajectory(ts="0.1"):
    """ Reads forces from a CSV and plots them over the trajectory. """
    # Read trajectory data
    posX, posY, posZ = [], [], []
    file_model = files_path + f"panda_traj_model_ts_{ts}.csv"
    with open(file_model, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            posX.append(float(row[0]))  # X value
            posY.append(float(row[1]))  # Y value
            posZ.append(float(row[2]))  # Z value
    
    # Read ideal trajectory
    with open(files_path + "panda_ideal_traj_model.csv", "r") as f:
        line = f.read()
    # Extract all matches of the form [x y z]
    matches = re.findall(r"\[([^\]]+)\]", line)
    # Convert matches to a numpy array
    xyz_array = np.array([
        list(map(float, m.split()[:3]))
        for m in matches
    ])

    x_ideal = xyz_array[:, 0]
    y_ideal = xyz_array[:, 1]
    z_ideal = xyz_array[:, 2]
    
    # Read forces data
    fx_vals, fy_vals, fz_vals = [], [], []
    file_forces = files_path + f"panda_forces_model_world_ts_{ts}.csv"
    with open(file_forces, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            fx_vals.append(float(row[0]))  # Fx
            fy_vals.append(float(row[1]))  # Fy
            fz_vals.append(float(row[2]))  # Fz
    
    # normalice forces between -1 and 1
    max_force = 55 # Assuming the maximum force is 55N
    scale = 1.0  # Scale factor for the force vectors plotting
    fx_vals = np.array(fx_vals)
    fy_vals = np.array(fy_vals)
    fz_vals = np.array(fz_vals)
    fx_vals = np.clip(fx_vals/max_force, -1, 1) * scale
    fy_vals = np.clip(fy_vals/max_force, -1, 1)
    fz_vals = np.clip(fz_vals/max_force, -1, 1)
    # Sample forces spaced by 5
    sample_rate = 3
    sampled_fx_vals = fx_vals[::sample_rate]
    sampled_fy_vals = fy_vals[::sample_rate]
    sampled_fz_vals = fz_vals[::sample_rate]
    sampled_posX = posX[::sample_rate]
    sampled_posY = posY[::sample_rate]
    sampled_posZ = posZ[::sample_rate]
    # Plot 3D trajectory and forces
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_zlim(0, 1.15)
    ax.plot(posX, posY, posZ, label='Model Trajectory', color='r')
    ax.plot(x_ideal, y_ideal, z_ideal, label='Ideal Trajectory', color='b')
    ax.quiver(sampled_posX, sampled_posY, sampled_posZ, sampled_fx_vals, sampled_fy_vals, sampled_fz_vals,\
                length=0.1, normalize=False, color='g', label='Force Vector')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('3D Trajectory with Forces')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot data from benchmark runs")
    parser.add_argument("-s", "--time_step", type=float, default=0.1, help="time step used during benchmark")
    args = parser.parse_args()
    ts = f"{args.time_step:.1f}"

    plot_trajectory(ts=ts)
    plot_force_over_trajectory(ts=ts)
    plot_forces_from_csv(filename=f"panda_forces_model_ts_{ts}.csv", title="Forces at the End Effector Over Time")
    plot_torques_from_csv(filename=f"panda_torques_model_ts_{ts}.csv", title="Torques at the End Effector Over Time")
    plot_joint_torques_from_csv(filename=f"panda_joint_torques_model_ts_{ts}.csv")
    plot_vel_ef_from_csv(filename=f"panda_vel_ef_model_ts_{ts}.csv")
    plot_forces_from_csv(filename=f"panda_forces_model_world_ts_{ts}.csv", title="Forces at the World Frame Over Time")
    plot_torques_from_csv(filename=f"panda_torques_model_world_ts_{ts}.csv", title="Torques at the World Frame Over Time")
