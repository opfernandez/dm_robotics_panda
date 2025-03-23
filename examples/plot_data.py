import csv
import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(filename="panda_trajectory.csv", num_loops=3):
    # Leer los datos del CSV
    x_vals = []
    y_vals = []
    x_ideal = []
    y_ideal = []
    x_vals2 = []
    y_vals2 = []
    with open("panda_ideal_trajectory.csv", mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Saltar el encabezado
        for row in reader:
            x_ideal.append(float(row[0]))  # Leer x
            y_ideal.append(float(row[1]))  # Leer y
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Saltar el encabezado
        for row in reader:
            x_vals.append(float(row[0]))  # Leer x
            y_vals.append(float(row[1]))  # Leer y
    with open("panda_trajectory_sb.csv", mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Saltar el encabezado
        for row in reader:
            x_vals2.append(float(row[0]))  # Leer x
            y_vals2.append(float(row[1]))  # Leer y
    # Graficar la trayectoria en el plano XY
    colors = ["b", "r", "g"]
    # Determinar el tamaño de cada vuelta
    total_points = len(x_vals)
    points_per_loop = total_points // num_loops  # Suponemos que todas las vueltas tienen el mismo número de puntos
    # Dibujar cada vuelta con un color diferente
    for i in range(num_loops):
        start = i * points_per_loop
        end = (i + 1) * points_per_loop if i < num_loops - 1 else total_points  # Asegurar que la última vuelta toma todos los puntos restantes
        plt.plot(x_vals[start:end], y_vals[start:end], marker="o", linestyle="-", color=colors[i % len(colors)], label=f"Vuelta {i+1}")
    plt.scatter(x_ideal, y_ideal, color='y', marker='*', label='Ideal')
    plt.scatter(x_vals2, y_vals2, color='c', marker='*', label='SB')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Trayectoria del Efector Final en el plano XY")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_forces_from_csv(filename="panda_forces.csv", timestep=0.1):
    """Lee fuerzas Fx, Fy, Fz desde un CSV y las grafica en función del tiempo."""
    
    # Listas para almacenar las fuerzas
    fx_vals, fy_vals, fz_vals = [], [], []
    
    # Leer datos desde el archivo CSV
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            fx_vals.append(float(row[0]))  # Fx
            fy_vals.append(float(row[1]))  # Fy
            fz_vals.append(float(row[2]))  # Fz

    # Crear los timesteps con un salto de 0.1s
    timesteps = np.arange(0, len(fx_vals) * timestep, timestep)

    # Graficar las fuerzas
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, fx_vals, label="Fx", color="r")  # Rojo
    plt.plot(timesteps, fy_vals, label="Fy", color="g")  # Verde
    plt.plot(timesteps, fz_vals, label="Fz", color="b")  # Azul

    # Etiquetas y título
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Fuerza (N)")
    plt.title("Evolución de las Fuerzas del Efector Final en el Tiempo")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_torques_from_csv(filename="panda_torques.csv", timestep=0.1):
    """Lee fuerzas Fx, Fy, Fz desde un CSV y las grafica en función del tiempo."""
    
    # Listas para almacenar las fuerzas
    tx_vals, ty_vals, tz_vals = [], [], []
    
    # Leer datos desde el archivo CSV
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            tx_vals.append(float(row[0]))  # Fx
            ty_vals.append(float(row[1]))  # Fy
            tz_vals.append(float(row[2]))  # Fz

    # Crear los timesteps con un salto de 0.1s
    timesteps = np.arange(0, len(tx_vals) * timestep, timestep)

    # Graficar las fuerzas
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, tx_vals, label="Tx", color="r")  # Rojo
    plt.plot(timesteps, ty_vals, label="Ty", color="g")  # Verde
    plt.plot(timesteps, tz_vals, label="Tz", color="b")  # Azul

    # Etiquetas y título
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Torque (Nm)")
    plt.title("Evolución de los Torques del Efector Final en el Tiempo")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_joint_torques_from_csv(filename="panda_joint_torques.csv", timestep=0.1):
    """Lee fuerzas Fx, Fy, Fz desde un CSV y las grafica en función del tiempo."""
    
    # Listas para almacenar las fuerzas
    jt1_vals, jt2_vals, jt3_vals, jt4_vals, jt5_vals, jt6_vals, jt7_vals = [], [], [], [], [], [], []
    
    # Leer datos desde el archivo CSV
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            jt1_vals.append(float(row[0]))  # Fx
            jt2_vals.append(float(row[1]))  # Fy
            jt3_vals.append(float(row[2]))  # Fz
            jt4_vals.append(float(row[3]))  # Fz
            jt5_vals.append(float(row[4]))  # Fz
            jt6_vals.append(float(row[5]))  # Fz
            jt7_vals.append(float(row[6]))  # Fz

    # Crear los timesteps con un salto de 0.1s
    timesteps = np.arange(0, len(jt1_vals) * timestep, timestep)

    # Graficar las fuerzas
    plt.figure(figsize=(8, 5))
    plt.plot(timesteps, jt1_vals, label="Joint 1 Torque", color="r")  # Rojo
    plt.plot(timesteps, jt2_vals, label="Joint 2 Torque", color="g")  # Verde
    plt.plot(timesteps, jt3_vals, label="Joint 3 Torque", color="b")  # Azul
    plt.plot(timesteps, jt4_vals, label="Joint 4 Torque", color="y")  # Azul
    plt.plot(timesteps, jt5_vals, label="Joint 5 Torque", color="m")  # Azul
    plt.plot(timesteps, jt6_vals, label="Joint 6 Torque", color="c")  # Azul
    plt.plot(timesteps, jt7_vals, label="Joint 7 Torque", color="k")  # Azul

    # Etiquetas y título
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Torque Articular (Nm)")
    plt.title("Evolución de los Torques de las Articulaciones en el Tiempo")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_vel_ef_from_csv(filename="panda_vel_ef.csv", timestep=0.1, title="Evolución de las Velocidades del Efector Final en el Tiempo"):
    """Lee fuerzas Fx, Fy, Fz desde un CSV y las grafica en función del tiempo."""
    
    # Listas para almacenar las fuerzas
    vx_vals, vy_vals, vz_vals = [], [], []
    
    # Leer datos desde el archivo CSV
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        for row in reader:
            vx_vals.append(float(row[0]))  # Fx
            vy_vals.append(float(row[1]))  # Fy
            vz_vals.append(float(row[2]))  # Fz

    # Crear los timesteps con un salto de 0.1s
    timesteps = np.arange(0, len(vx_vals) * timestep, timestep)

    # Graficar las fuerzas
    plt.figure(figsize=(8, 5))

    plt.plot(timesteps, vx_vals, label="Vx", color="r",)  # Rojo
    plt.plot(timesteps, vy_vals, label="Vy", color="g",)  # Verde
    plt.plot(timesteps, vz_vals, label="Vz", color="b")  # Azul

    # Etiquetas y título
    plt.ylim(-0.12, 0.12)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Velocidad (m/s)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_eudist(timestep=0.1, filename="panda_euclidean.csv"):
    # Leer los datos del CSV
    eud = []
    eud2 = []
    with open(filename, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Saltar el encabezado
        for row in reader:
            eud.append(float(row[0]))  # Leer x
    
    with open("panda_euclidean_sb.csv", mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Saltar el encabezado
        for row in reader:
            eud2.append(float(row[0]))  # Leer x

    timesteps = np.arange(0, len(eud) * timestep, timestep)
    plt.figure(figsize=(8, 6))
    plt.plot(timesteps, eud, label="euclidean distance", color="r",)  # Rojo
    plt.plot(timesteps, eud2, label="euclidean distance sb", color="b",)  # Rojo
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Euclidea (m)")
    plt.title("Distancia Euclidea respecto a la trayectoria ideal")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    plot_trajectory()
    plot_eudist()
    plot_torques_from_csv()
    plot_forces_from_csv()
    plot_joint_torques_from_csv()
    plot_vel_ef_from_csv()
    plot_vel_ef_from_csv(filename="panda_expected_vel.csv", timestep=0.1, title="Velocidades Ideales del Efector Final")
