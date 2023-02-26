import os.path

import numpy as np

from Config_JSON.Config_Load_Functions import load_vehicle_properties, load_flight_path, load_sensor_config
from General.UsefulFunctions import get_parent_path
from FCS.Control_System import ControlSystem
from Simulation.Thrust_Torque_Model import ThrustModel
from Simulation.Dynamics_Model import DynamicsModel
from Data_Handling.Data_Classes import TimeState, EstimateState
from Data_Handling.Data_Processing import DynamicsData, ControlData, DataProcessor
from Simulation.Sensor_Model import Sensors
from FCS.State_Estimation import StateEstimation

# Simulation time variables
t_duration = 30
t_delta = 0.01

# PID Gain Values [Kp, Ki, Kd]
gain_x = [-0.03, 0, 0.09]
gain_y = [0.07, 0, -0.18]
gain_z = [45e3, 17e3, -6.5e3]

gain_yaw = [10, 0, -40]
gain_pitch = [-115, 0, 45]
gain_roll = [115, 0, -45]

# Sensor Variables
state_estimation_frequency = 100  # Hz
acc_lpf_cutoff_freq = 4  # Hz
gyro_lpf_cutoff_freq = 4  # Hz

# State Estimation Variables
alpha = 0.3

# Config properties file paths
VEHICLE_PROPERTIES_JSON_PATH = r"Config_JSON/Vehicle_Properties.json"
FLIGHT_PATH_JSON_PATH = r"Config_JSON/Flight_Path.json"
SENSOR_CONFIG_JSON_PATH = r"Config_JSON/Sensors.json"

# Data save path
SAVE_PATH = os.path.join(get_parent_path(os.path.abspath(__file__)), "Data")

# Copy config JSON to save path


def main():
    # Load Config properties
    properties, dimensions, motor, propeller = load_vehicle_properties(VEHICLE_PROPERTIES_JSON_PATH)
    IC, maneuvers = load_flight_path(FLIGHT_PATH_JSON_PATH)
    acc, gyro, mag = load_sensor_config(SENSOR_CONFIG_JSON_PATH)

    # Instantiate simulation objects
    thrust = ThrustModel(t_delta, motor, propeller, dimensions)
    control = ControlSystem(t_delta, maneuvers, gain_x, gain_y, gain_z, gain_yaw, gain_pitch, gain_roll)
    dynamics = DynamicsModel(properties, dimensions)
    sensors = Sensors(acc, gyro, mag)
    estimation = StateEstimation(state_estimation_frequency, acc_lpf_cutoff_freq, gyro_lpf_cutoff_freq, alpha)

    # Set up initial conditions
    state_vector = np.array([0, 0, 0, 0, 0, 0, IC["x"], IC["y"], IC["z"], 0, 0, 0])
    X = TimeState(state_vector)

    # Create Initial State Estimate
    G = EstimateState(np.zeros(3))

    # Initialise data collection objects
    dynamics_data = DynamicsData()
    dynamics_data.append_time_instance(t=0, X=X, T=np.array(4 * [0]), U=np.array(4 * [0]))

    run_simulation(dynamics, thrust, control, sensors, estimation, X, G, dynamics_data)

    control_data = ControlData(control.return_pid_data())
    thrust_data, torque_data = thrust.return_thrust_data()
    sensor_data = sensors.return_sensor_data()
    estimation_data = estimation.return_data()

    dp = DataProcessor(dynamics_data, control_data, thrust_data, torque_data, sensor_data, estimation_data, SAVE_PATH)
    dp.plot_inertial(save=True)
    dp.plot_thrust(show=False, save=True)
    dp.plot_induced_velocity(show=False, save=True)
    dp.plot_3d(show=False, save=True)
    dp.plot_sensors(show=False, save=True)


def run_simulation(dynamics: DynamicsModel, thrust: ThrustModel, control: ControlSystem, sensors: Sensors,
                   estimation: StateEstimation, X: TimeState, G: EstimateState, dynamics_data: DynamicsData):
    for t in np.arange(t_delta, t_duration + t_delta, t_delta):
        t = round(t, 4)

        U = control.run_control_loop(X, G, t)
        T = thrust.solve_thrust(X, U)
        Q = thrust.calculate_torque(T, U)
        X, X_dot = dynamics.rk4(X, T, Q, t_delta)
        S = sensors.simulate_sensors(t, X, X_dot)
        G = estimation.compute_state_estimate(t, S)

        dynamics_data.append_time_instance(t, X, T, U)

        print("\nTime [s]: {}\nX [m]: {}\tY [m]: {}\tZ [m]: {}\nYaw [deg]: {}\tPitch [deg]: {}\tRoll [deg]: {}"
              "\nU [RPM]: {}\n T [N]: {}".format(t, X.x, X.y, X.z, np.degrees(X.psi), np.degrees(X.theta),
                                                 np.degrees(X.phi), U, T))


if __name__ == "__main__":
    main()
