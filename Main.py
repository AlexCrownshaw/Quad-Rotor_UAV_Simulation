import numpy as np

from FCS.Control_System import ControlSystem
from Simulation.Dynamics_Model import DynamicsModel

# Simulation time variables
t_duration = 100
t_delta = 0.001

# PID Gain Values [Kp, Ki, Kd]
gain_x = [1, 1, 1]
gain_y = [1, 1, 1]
gain_z = [1, 1, 1]

gain_yaw = [1, 1, 1]
gain_pitch = [1, 1, 1]
gain_roll = [1, 1, 1]

# Config properties file paths
STRUCTURAL_JSON_PATH = r"Config_JSON/Structural_Properties/Structure.json"
FLIGHT_PATH_JSON_PATH = r"Config_JSON/Flight_Plan/Flight_Path.json"


def main():

    # Instantiate control system object
    control = ControlSystem(gain_x, gain_y, gain_z, gain_yaw, gain_pitch, gain_roll)

    # Instantiate dynamics Simulation Object
    dynamics = DynamicsModel(t_delta, STRUCTURAL_JSON_PATH, FLIGHT_PATH_JSON_PATH)

    # x = Initial conditions

    t_steps = np.arange(0, t_duration / t_delta + t_delta, t_delta)

    for t in t_steps:

        u = control.run_control_loop()
        x = dynamics.rk4(x, u)


if __name__ == "__main__":
    main()
