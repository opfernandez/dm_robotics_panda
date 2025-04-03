import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def calculate_trajectory(ef_position):
    state = 1
    cont = 1
    cycles = 30
    posX = ef_position[0]; posY = ef_position[1]; posZ = ef_position[2]
    constant_vel = 0.05
    incT = 0.1
    trajectory = np.zeros((240, 3), dtype=np.float32)
    trajectory[0] = [posX, posY, posZ]
    for step in range(1, 240): # Ideal square trajectory
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
      trajectory[step] = [posX, posY, posZ]
    return trajectory

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_path = os.path.join(script_dir, "../../data/")
    # Compute ideal trajectory
    traj = calculate_trajectory([0.012972587,0.47250384, 0.0]) # Initial ef position

    # Read trained model followed trajectory
    posX, posY = [], []
    file_model = files_path + "panda_test_model.csv"
    with open(file_model, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Saltar el encabezado
        for row in reader:
            posX.append(float(row[0]))  # X value
            posY.append(float(row[1]))  # Y value

    # Read hard-coded followed trajectory
    pos2X, pos2Y = [], []
    file_hc = files_path + "panda_trajectory.csv"
    with open(file_hc, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Saltar el encabezado
        for row in reader:
            pos2X.append(float(row[0]))  # X value
            pos2Y.append(float(row[1]))  # Y value

    # Extract XY coordinates
    x_ideal = traj[0:120, 0]
    y_ideal = traj[0:120, 1]

    # Plotting the trajectory in the XY plane
    plt.figure(figsize=(8, 6))
    plt.plot(pos2X[0:600], pos2Y[0:600], marker='o', linestyle='-', color='g', label="Trayectoria hard-coded")
    plt.plot(posX[0:600], posY[0:600], marker='o', linestyle='-', color='r', label="Trayectoria SAC")
    plt.plot(x_ideal, y_ideal, marker='o', linestyle='-', color='b', label="Trayectoria ideal")

    plt.xlabel("Posición X")
    plt.ylabel("Posición Y")
    plt.title("Trayectoria en el plano XY (5 iteraciones)")
    plt.legend()
    plt.grid(True)
    plt.show()