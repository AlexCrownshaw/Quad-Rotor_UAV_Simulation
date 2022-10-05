import json
import os.path

import numpy as np

from typing import Tuple
from General.UsefulFunctions import get_parent_path
from FCS.Control_System import ControlSystem
from Simulation.Dynamics_Model import DynamicsModel
from Simulation.Time_State import TimeState
from Data_Processing.Data_Processing import DynamicsData, ControlData, DataProcessor

# Simulation time variables
t_duration = 30
t_delta = 0.01

# PID Gain Values [Kp, Ki, Kd]
gain_x = [1, 1, 1]
gain_y = [1, 1, 1]
gain_z = [0.5, 0.01, 0.3]

gain_yaw = [1, 1, 1]
gain_pitch = [1, 1, 1]
gain_roll = [1, 1, 1]

# Config properties file paths
STRUCTURAL_JSON_PATH = r"Config_JSON/Structural_Properties/Structure.json"
FLIGHT_PATH_JSON_PATH = r"Config_JSON/Flight_Plan/Flight_Path.json"

# Data save path
SAVE_PATH = os.path.join(get_parent_path(os.path.abspath(__file__)), "Data")


def main():
    # Load Config properties
    properties, dimensions = load_structural_json(STRUCTURAL_JSON_PATH)
    IC, maneuvers = load_flight_path_json(FLIGHT_PATH_JSON_PATH)

    # Instantiate control system object
    control = ControlSystem(t_delta, maneuvers, gain_x, gain_y, gain_z, gain_yaw, gain_pitch, gain_roll)

    # Instantiate dynamics Simulation Object
    dynamics = DynamicsModel(properties, dimensions)

    # Set up initial conditions
    state_vector = np.array([0, 0, 0, 0, 0, 0, IC["x"], IC["y"], IC["z"], 0, 0, 0])
    X = TimeState(state_vector)

    # Initialise data collection objects
    dynamics_data = DynamicsData()
    dynamics_data.append_time_instance(t=0, X=X, U=np.array([0, 0, 0, 0]))

    for t in np.arange(t_delta, t_duration + t_delta, t_delta):
        t = round(t, 4)

        U = control.run_control_loop(X, t)
        X = dynamics.rk4(X, U, t_delta)

        dynamics_data.append_time_instance(t, X, U)

        print("Calculation Complete for t = {}s".format(t))

    control_data = ControlData(control.return_pid_data())

    dp = DataProcessor(dynamics_data, control_data, SAVE_PATH)
    dp.plot_inertial(save=True)

    # dynamics_data.plot_3d()


def load_structural_json(json_path: str) -> Tuple[dict, dict]:
    with open(json_path) as json_file:
        json_object = json.load(json_file)
        return json_object["properties"], json_object["dimensions"]


def load_flight_path_json(json_path: str) -> Tuple[dict, dict]:
    with open(json_path) as json_file:
        json_object = json.load(json_file)
        return json_object["Initial_Conditions"], json_object["Maneuvers"]


if __name__ == "__main__":
    main()
