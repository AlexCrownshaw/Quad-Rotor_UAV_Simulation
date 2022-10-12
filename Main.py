import json
import os.path

import numpy as np

from typing import Tuple
from General.UsefulFunctions import get_parent_path
from FCS.Control_System import ControlSystem
from Simulation.Thrust_Model import ThrustModel
from Simulation.Dynamics_Model import DynamicsModel
from Simulation.Time_State import TimeState
from Data_Processing.Data_Processing import DynamicsData, ControlData, DataProcessor

# Simulation time variables
t_duration = 10
t_delta = 0.01

# PID Gain Values [Kp, Ki, Kd]
# gain_x = [0.05, 0, 0.5]
gain_x = [0, 0, 0]
gain_y = [0, 0, 0]
# gain_z = [1e3, 0.7, 1e3]  # Setpoint 10
gain_z = [45e3, 17e3, -6.5e3]  # Setpoint 1

gain_yaw = [0, 0, 0]
gain_pitch = [0, 0, 0]
# gain_pitch = [0.05, 0, 0.5]
gain_roll = [0, 0, 0]

# Config properties file paths
VEHICLE_PROPERTIES_JSON_PATH = r"Config_JSON/Structural_Properties/Vehicle_Properties.json"
FLIGHT_PATH_JSON_PATH = r"Config_JSON/Flight_Plan/Flight_Path.json"

# Data save path
SAVE_PATH = os.path.join(get_parent_path(os.path.abspath(__file__)), "Data")


def main():
    # Load Config properties
    properties, dimensions, motor, propeller = load_vehicle_properties_json(VEHICLE_PROPERTIES_JSON_PATH)
    IC, maneuvers = load_flight_path_json(FLIGHT_PATH_JSON_PATH)

    # Instantiate simulation objects
    thrust = ThrustModel(t_delta, motor, propeller, dimensions)
    control = ControlSystem(t_delta, maneuvers, gain_x, gain_y, gain_z, gain_yaw, gain_pitch, gain_roll)
    dynamics = DynamicsModel(properties, dimensions)

    # Set up initial conditions
    state_vector = np.array([0, 0, 0, 0, 0, 0, IC["x"], IC["y"], IC["z"], 0, 0, 0])
    X = TimeState(state_vector)

    # Initialise data collection objects
    dynamics_data = DynamicsData()
    dynamics_data.append_time_instance(t=0, X=X, T=np.array(4 * [0]), U=np.array(4 * [0]))

    for t in np.arange(t_delta, t_duration + t_delta, t_delta):
        t = round(t, 4)

        U = control.run_control_loop(X, t)
        T = thrust.solve_thrust(X, U)
        X = dynamics.rk4(X, T, t_delta)

        dynamics_data.append_time_instance(t, X, T, U)

        print("Calculation Complete for t = {}s".format(t))

    control_data = ControlData(control.return_pid_data())
    thrust_data = thrust.return_thrust_data()

    dp = DataProcessor(dynamics_data, control_data, thrust_data, SAVE_PATH)
    dp.plot_inertial(save=True)
    dp.plot_thrust(show=False, save=True)
    dp.plot_induced_velocity(show=False, save=True)
    dp.plot_3d(show=False, save=True)


def load_vehicle_properties_json(json_path: str) -> Tuple[dict, dict, dict, dict]:
    with open(json_path) as json_file:
        json_object = json.load(json_file)
        return json_object["properties"], json_object["dimensions"], json_object["motor"], json_object["propeller"]


def load_flight_path_json(json_path: str) -> Tuple[dict, dict]:
    with open(json_path) as json_file:
        json_object = json.load(json_file)
        return json_object["Initial_Conditions"], json_object["Maneuvers"]


if __name__ == "__main__":
    main()
