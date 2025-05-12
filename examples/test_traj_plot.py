import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import re

def calculate_trajectory(ef_position, uc="square"):
    state = 1
    cont = 1
    cycles = 30
    posX = ef_position[0]; posY = ef_position[1]; posZ = ef_position[2]
    constant_vel = 0.4
    incT = 0.1
    n_points = 4800
    trajectory = np.zeros((n_points, 3), dtype=np.float32)
    trajectory[0] = [posX, posY, posZ]
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
            trajectory[step] = [posX, posY, posZ]
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
            trajectory[step] = [posX, posY, posZ]
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
            trajectory[step] = [posX, posY, posZ]
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
            trajectory[step] = [posX, posY, posZ]
    #######################################################
    elif uc == "h-circle":
        r = 0.07
        trajectory = []
        # Uniformely distributed angles
        thetas = np.linspace(-np.pi/2, 1.5*np.pi, 120) # h

        laps = n_points // 120
        if  (n_points % 120) != 0:
            laps += 1
        for i in range(laps):
            # Circular coordinates on XY plane
            x = posX + r * np.cos(thetas)
            y = posY - r * np.sin(thetas) - r
            z = np.full_like(x, posZ)  # Z constant

            lap_traj = np.stack((x, y, z), axis=1)
            trajectory.append(lap_traj)

        trajectory = np.vstack(trajectory)
        print(len(trajectory))
    elif uc == "ah-circle":
        r = 0.05
        trajectory = []
        # Uniformely distributed angles
        thetas = np.linspace(-np.pi/2, (-5/2)*np.pi, 120) # ah

        laps = n_points // 120
        if  (n_points % 120) != 0:
            laps += 1
        for i in range(laps):
            # Circular coordinates on XY plane
            x = posX + r * np.cos(thetas)
            y = posY - r * np.sin(thetas) - r
            z = np.full_like(x, posZ)  # Z constant

            lap_traj = np.stack((x, y, z), axis=1)
            trajectory.append(lap_traj)

        trajectory = np.vstack(trajectory)
        print(len(trajectory))
    #######################################################
    elif uc == "ah-pentagon":
        # Parámetros del pentágono
        r = 0.07  # Radio del círculo circunscrito
        num_sides = 5
        points_per_side = 24  # 120 puntos totales / 5 lados
        total_vertices = []
        center_x, center_y = posX, posY

        # Calcular los vértices del pentágono (vértice superior a las 12 en punto)
        for i in range(num_sides):
            angle = 2 * np.pi * i / num_sides + np.pi / 2  # vértice arriba
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            total_vertices.append((x, y))
        total_vertices.append(total_vertices[0])  # cerrar la figura

        # Crear la trayectoria con 24 puntos por lado
        trajectory = []
        laps = n_points // 120
        if (n_points % 120) != 0:
            laps += 1
        for i in range(laps):
            for i in range(num_sides):
                start = np.array(total_vertices[i])
                end = np.array(total_vertices[i + 1])
                for t in np.linspace(0, 1, points_per_side, endpoint=False):
                    point = (1 - t) * start + t * end
                    trajectory.append([point[0], point[1], posZ])

        trajectory = np.array(trajectory, dtype=np.float32)
    elif uc == "h-pentagon":
        # Parámetros del pentágono
        r = 0.07  # Radio del círculo circunscrito
        num_sides = 5
        points_per_side = 24  # 120 puntos totales / 5 lados
        total_vertices = []
        center_x, center_y = posX, posY

        # Calcular los vértices del pentágono (vértice superior a las 12 en punto)
        for i in range(num_sides):
            angle = -2 * np.pi * i / num_sides + np.pi / 2  # vértice arriba
            x = center_x + r * np.cos(angle)
            y = center_y + r * np.sin(angle)
            total_vertices.append((x, y))
        total_vertices.append(total_vertices[0])  # cerrar la figura

        # Crear la trayectoria con 24 puntos por lado
        trajectory = []
        laps = n_points // 120
        if (n_points % 120) != 0:
            laps += 1
        for i in range(laps):
            for i in range(num_sides):
                start = np.array(total_vertices[i])
                end = np.array(total_vertices[i + 1])
                for t in np.linspace(0, 1, points_per_side, endpoint=False):
                    point = (1 - t) * start + t * end
                    trajectory.append([point[0], point[1], posZ])

        trajectory = np.array(trajectory, dtype=np.float32)
    else:
        print("\n\tERROR DURING TRAJECTORY CALCULATION!!!\n")
    return trajectory

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    files_path = os.path.join(script_dir, "../../data/")

    # Read trained model followed trajectory
    posX, posY = [], []
    file_model = files_path + "panda_traj_model.csv"
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

    # Leer la línea como una sola cadena
    with open(files_path + "panda_ideal_traj_model.csv", "r") as f:
        line = f.read()
    # Extraer vectores con corchetes
    matches = re.findall(r"\[([^\]]+)\]", line)
    # Extraer x, y
    xy_array = np.array([
        list(map(float, m.split()[:2]))
        for m in matches
    ])

    print(xy_array.shape)
    print(xy_array)
    x_ideal = xy_array[:, 0]
    y_ideal = xy_array[:, 1]


    # Compute ideal trajectory
    # traj = calculate_trajectory(ef_position=[posX[0], posY[0], 0.0], uc="h-pentagon") # square // triangle // circle
    # # Extract XY coordinates
    # x_ideal = traj[0:5000, 0]
    # y_ideal = traj[0:5000, 1]

    # Plotting the trajectory in the XY plane
    plt.figure(figsize=(8, 6))
    # plt.plot(pos2X[0:600], pos2Y[0:600], marker='o', linestyle='-', color='c', label="Trayectoria hard-coded")
    plt.plot(posX[:], posY[:], marker='o', linestyle='-', color='r', label="Trayectoria SAC")
    plt.plot(x_ideal, y_ideal, marker='o', linestyle='-', color='b', label="Trayectoria ideal")
    plt.plot(x_ideal[0:80], y_ideal[0:80], marker='o', linestyle='-', color='g', label="Sentido de giro")


    plt.xlabel("Posición X")
    plt.ylabel("Posición Y")
    plt.title("Trayectoria en el plano XY (5 iteraciones)")
    plt.legend()
    plt.grid(True)
    plt.show()


# def calculate_trajectory(ef_position, uc="square"):
#     state = 1
#     cont = 1
#     cycles = 30
#     posX = ef_position[0]; posY = ef_position[1]; posZ = ef_position[2]
#     constant_vel = 0.03
#     incT = 0.1
#     n_points = 5000
#     trajectory = np.zeros((n_points, 3), dtype=np.float32)
#     trajectory[0] = [posX, posY, posZ]
#     #######################################################
#     if uc == "ah-square":
#       for step in range(1, n_points): # Ideal square trajectory
#         if state == 0: # Move through Z axis
#           posZ += incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 1
#             cont = 0
#         elif state == 1: # Move through X axis
#           posX += -incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 2
#             cont = 0
#         elif state == 2: # Move through Y axis
#           posY += incT * (-1*constant_vel)
#           cont += 1
#           if cont >= cycles:
#             state = 3
#             cont = 0
#         elif state == 3: # Move through X axis
#           posX += incT * (constant_vel)
#           cont += 1
#           if cont >= cycles:
#             state = 4
#             cont = 0
#         elif state == 4: # Move through Y axis
#           posY += incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 1
#             cont = 0
#         trajectory[step] = [posX, posY, posZ]
#     #######################################################
#     elif uc == "h-square":
#       for step in range(1, n_points): # Ideal square trajectory
#         if state == 0: # Move through Z axis
#           posZ += -incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 1
#             cont = 0
#         elif state == 1: # Move through X axis
#           posX += incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 2
#             cont = 0
#         elif state == 2: # Move through Y axis
#           posY += incT * (-1*constant_vel)
#           cont += 1
#           if cont >= cycles:
#             state = 3
#             cont = 0
#         elif state == 3: # Move through X axis
#           posX += -incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 4
#             cont = 0
#         elif state == 4: # Move through Y axis
#           posY += incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 1
#             cont = 0
#         trajectory[step] = [posX, posY, posZ]
#     #######################################################
#     elif uc == "ah-triangle":
#       for step in range(1, n_points): # Ideal square trajectory
#         if state == 0: # Move through Z axis
#           posZ += incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 1
#             cont = 0
#         elif state == 1: # Move through X axis
#           posX += -incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 2
#             cont = 0
#         elif state == 2: # Move through Y axis
#           posY += incT * (-1*constant_vel)
#           posX += incT * (0.5*constant_vel)
#           cont += 1
#           if cont >= cycles:
#             state = 3
#             cont = 0
#         elif state == 3: # Move through X axis
#           posY += incT * constant_vel
#           posX += incT * (0.5*constant_vel)
#           cont += 1
#           if cont >= cycles:
#             state = 1
#             cont = 0
#         trajectory[step] = [posX, posY, posZ]
#     #######################################################
#     elif uc == "h-triangle":
#       for step in range(1, n_points): # Ideal square trajectory
#         if state == 0: # Move through Z axis
#           posZ += incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 1
#             cont = 0
#         elif state == 1: # Move through X axis
#           posX += incT * constant_vel
#           cont += 1
#           if cont >= cycles:
#             state = 2
#             cont = 0
#         elif state == 2: # Move through Y axis
#           posY += incT * (-1*constant_vel)
#           posX += -incT * (0.5*constant_vel)
#           cont += 1
#           if cont >= cycles:
#             state = 3
#             cont = 0
#         elif state == 3: # Move through X axis
#           posY += incT * constant_vel
#           posX += -incT * (0.5*constant_vel)
#           cont += 1
#           if cont >= cycles:
#             state = 1
#             cont = 0
#         trajectory[step] = [posX, posY, posZ]
#     #######################################################
#     elif uc == "circle":
#       r = 0.075
#       trajectory = []
#       # Uniformely distributed angles
#       thetas = np.linspace(0, 2*np.pi, 120)

#       laps = n_points // 120
#       if  (n_points % 120) != 0:
#         laps += 1
#       for i in range(laps):
#         # Circular coordinates on XY plane
#         x = posX + r * np.cos(thetas)
#         y = posY - r * np.sin(thetas) - r
#         z = np.full_like(x, posZ)  # Z constant

#         lap_traj = np.stack((x, y, z), axis=1)
#         trajectory.append(lap_traj)

#       trajectory = np.vstack(trajectory)
#       print(len(trajectory))
#     else:
#       print("\n\tERROR DURING TRAJECTORY CALCULATION!!!\n")
#     return trajectory