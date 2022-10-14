import os.path

import numpy as np

from Config_JSON.Config_Load_Functions import load_vehicle_properties_json, load_flight_path_json
from General.UsefulFunctions import get_parent_path
from FCS.Control_System import ControlSystem
from Simulation.Thrust_Model import ThrustModel
from Simulation.Dynamics_Model import DynamicsModel
from Simulation.Time_State import TimeState
from Data_Processing.Data_Processing import DynamicsData, ControlData, DataProcessor

# Simulation time variables
t_duration = 25
t_delta = 0.01

# PID Gain Values [Kp, Ki, Kd]
gain_x = [-0.03, 0, 0.09]
gain_y = [0.07, 0, -0.18]
gain_z = [45e3, 17e3, -6.5e3]

gain_yaw = [0, 0, 0]
gain_pitch = [-400, 0, 140]
gain_roll = [115, 0, -45]

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

        print("\nTime [s]: {}\nX [m]: {}\tY [m]: {}\tZ [m]: {}\nYaw [deg]: {}\tPitch [deg]: {}\tRoll [deg]: {}"
              "\nU [RPM]: {}\n T [N]: {}".format(t, X.x, X.y, X.z, X.psi, X.theta, X.phi, U, T))

    control_data = ControlData(control.return_pid_data())
    thrust_data = thrust.return_thrust_data()

    dp = DataProcessor(dynamics_data, control_data, thrust_data, SAVE_PATH)
    dp.plot_inertial(save=True)
    dp.plot_thrust(show=False, save=True)
    dp.plot_induced_velocity(show=False, save=True)
    dp.plot_3d(show=True, save=True)


if __name__ == "__main__":
    main()
