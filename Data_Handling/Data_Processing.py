import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Data_Handling.Data_Classes import TimeState


class DynamicsData:
    plt.rcParams['axes.grid'] = True

    def __init__(self):
        self.df = pd.DataFrame(columns=["t", "x", "y", "z", "psi_[rad]", "theta_[rad]", "phi_[rad]", "psi_[deg]",
                                        "theta_[deg]", "phi_[deg]", "u", "v", "w", "p", "q", "r",
                                        "F1", "F2", "F3", "F4", "omega_1", "omega_2", "omega_3", "omega_4"])

    def append_time_instance(self, t: float, X: TimeState, T: np.array, U: np.array):
        self.df.loc[len(self.df)] = [t, X.x, X.y, X.z, X.psi, X.theta, X.phi,
                                     np.degrees(X.psi), np.degrees(X.theta), np.degrees(X.phi),
                                     X.u, X.v, X.w, X.p, X.q, X.r, T[0], T[1], T[2], T[3], U[0], U[1], U[2], U[3]]

    def plot_variable(self, variable: str, show=False):
        plt.plot(self.df["t"], self.df[variable])

        if show:
            plt.show()


class ControlData:

    def __init__(self, control_data: list):
        self.control_data_list = control_data
        self.df = pd.DataFrame(columns=["time", "setpoint_x", "setpoint_y", "setpoint_z", "setpoint_yaw",
                                        "setpoint_pitch", "setpoint_roll",
                                        "input_x", "input_y", "input_z", "input_yaw", "input_pitch", "input_roll",
                                        "error_x", "error_y", "error_z", "error_yaw", "error_pitch", "error_roll",
                                        "integral_x", "integral_z", "integral_yaw", "integral_pitch",
                                        "integral_roll",
                                        "derivative_x", "derivative_y", "derivative_z", "derivative_yaw",
                                        "derivative_pitch", "derivative_roll",
                                        "output_x", "output_y", "output_z", "output_yaw", "output_pitch",
                                        "output_roll"])

        self.df["time"] = self.control_data_list[0]["time"]

        self.df["setpoint_x"] = self.control_data_list[0]["setpoint"]
        self.df["setpoint_y"] = self.control_data_list[1]["setpoint"]
        self.df["setpoint_z"] = self.control_data_list[2]["setpoint"]
        self.df["setpoint_yaw"] = self.control_data_list[3]["setpoint"]
        self.df["setpoint_pitch"] = self.control_data_list[4]["setpoint"]
        self.df["setpoint_roll"] = self.control_data_list[5]["setpoint"]

        self.df["input_x"] = self.control_data_list[0]["state_input"]
        self.df["input_y"] = self.control_data_list[1]["state_input"]
        self.df["input_z"] = self.control_data_list[2]["state_input"]
        self.df["input_yaw"] = self.control_data_list[3]["state_input"]
        self.df["input_pitch"] = self.control_data_list[4]["state_input"]
        self.df["input_roll"] = self.control_data_list[5]["state_input"]

        self.df["error_x"] = self.control_data_list[0]["error"]
        self.df["error_y"] = self.control_data_list[1]["error"]
        self.df["error_z"] = self.control_data_list[2]["error"]
        self.df["error_yaw"] = self.control_data_list[3]["error"]
        self.df["error_pitch"] = self.control_data_list[4]["error"]
        self.df["error_roll"] = self.control_data_list[5]["error"]

        self.df["integral_x"] = self.control_data_list[0]["integral"]
        self.df["integral_y"] = self.control_data_list[1]["integral"]
        self.df["integral_z"] = self.control_data_list[2]["integral"]
        self.df["integral_yaw"] = self.control_data_list[3]["integral"]
        self.df["integral_pitch"] = self.control_data_list[4]["integral"]
        self.df["integral_roll"] = self.control_data_list[5]["integral"]

        self.df["derivative_x"] = self.control_data_list[0]["derivative"]
        self.df["derivative_y"] = self.control_data_list[1]["derivative"]
        self.df["derivative_z"] = self.control_data_list[2]["derivative"]
        self.df["derivative_yaw"] = self.control_data_list[3]["derivative"]
        self.df["derivative_pitch"] = self.control_data_list[4]["derivative"]
        self.df["derivative_roll"] = self.control_data_list[5]["derivative"]

        self.df["output_x"] = self.control_data_list[0]["output"]
        self.df["output_y"] = self.control_data_list[1]["output"]
        self.df["output_z"] = self.control_data_list[2]["output"]
        self.df["output_yaw"] = self.control_data_list[3]["output"]
        self.df["output_pitch"] = self.control_data_list[4]["output"]
        self.df["output_roll"] = self.control_data_list[5]["output"]


class ThrustData:
    column_headers = ["time", "u_p", "v_p", "w_p", "v_i", "V_prime", "T"]

    def __init__(self):
        self.motor_data = [pd.DataFrame(columns=self.column_headers),
                           pd.DataFrame(columns=self.column_headers),
                           pd.DataFrame(columns=self.column_headers),
                           pd.DataFrame(columns=self.column_headers)]

        for motor_index in range(len(self.motor_data)):
            self.append_thrust_data(motor_index, 0, np.zeros(3), 0, 0, 0)

    def append_thrust_data(self, motor_index, t, V, v_i, V_prime, T) -> None:
        self.motor_data[motor_index].loc[len(self.motor_data[motor_index])] = [t, float(V[0]), float(V[1]), float(V[2]),
                                                                               v_i, V_prime, float(T)]


class TorqueData:
    column_headers = ["time", "Q_1", "Q_2", "Q_3", "Q_4", "Q_sum"]

    def __init__(self):
        self.df = pd.DataFrame(columns=self.column_headers)
        self.df.loc[0] = [0] * len(self.column_headers)

    def append_data(self, t, Q):
        self.df.loc[len(self.df)] = [t, Q[0], Q[1], Q[2], Q[3], np.sum(Q)]


class SensorData:
    column_headers = ["time", "acc_x", "acc_y", "acc_z", "acc_gn_x", "acc_gn_y", "acc_gn_z",
                      "gyro_x", "gyro_y", "gyro_z", "gyro_gn_x", "gyro_gn_y", "gyro_gn_z"]

    def __init__(self):
        self.df = pd.DataFrame(columns=self.column_headers)
        self.df.loc[0] = [0] * len(self.column_headers)

    def append_data(self, t, acc: np.array, acc_gn: np.array, gyro: np.array, gyro_gn: np.array):
        self.df.loc[len(self.df)] = [t, acc[0], acc[1], acc[2], acc_gn[0], acc_gn[1], acc_gn[2],
                                     gyro[0], gyro[1], gyro[2], gyro_gn[0], gyro_gn[1], gyro_gn[2]]


class StateEstimationData:
    column_headers = ["time", "psi_rad", "theta_rad", "phi_rad", "psi_deg", "theta_deg", "phi_deg",
                      "acc_filt_x", "acc_filt_y", "acc_filt_z",
                      "gyro_filt_x", "gyro_filt_y", "gyro_filt_z"]

    def __init__(self):
        self.df = pd.DataFrame(columns=self.column_headers)
        self.df.loc[0] = [0] * len(self.column_headers)

    def append_data(self, t, attitude_vector: np.array, acc_filt: np.array, gyro_filt: np.array):
        psi, theta, roll = attitude_vector
        self.df.loc[len(self.df)] = [t, psi, theta, roll, np.degrees(psi), np.degrees(theta), np.degrees(roll),
                                     acc_filt[0], acc_filt[1], acc_filt[2],
                                     gyro_filt[0], gyro_filt[1], gyro_filt[2]]


class DataProcessor:

    def __init__(self, dynamics: DynamicsData, control: ControlData, thrust_data: ThrustData, torque_data: TorqueData,
                 sensor_data: SensorData, estimation: StateEstimationData, disturbance: pd.DataFrame, save_path: str):

        self.dynamics = dynamics.df
        self.control = control.df
        self.thrust_data = thrust_data.motor_data
        self.torque_data = torque_data.df
        self.sensor_data = sensor_data.df
        self.estimate = estimation.df
        self.disturbance = disturbance

        self.save_path = os.path.join(save_path, str(time.strftime("%d-%m-%y %H-%M-%S")))
        os.mkdir(self.save_path)
        self.dynamics.to_csv(os.path.join(self.save_path, "Dynamics_Data.csv"))
        self.control.to_csv(os.path.join(self.save_path, "Control_Data.csv"))
        self.torque_data.to_csv(os.path.join(self.save_path, "Torque_Data.csv"))
        self.disturbance.to_csv(os.path.join(self.save_path, "Disturbance_Data.csv"))

        motor_data_save_path = os.path.join(self.save_path, "Motor Thrust Data")
        os.mkdir(motor_data_save_path)
        for motor_index in range(len(self.thrust_data)):
            self.thrust_data[motor_index].to_csv(os.path.join(motor_data_save_path,
                                                              "Motor {}.csv".format(motor_index + 1)))

        self.estimate.to_csv(os.path.join(self.save_path, "State_Estimate_Data.csv"))

        self.save_path = os.path.join(self.save_path, "Plots")
        os.mkdir(self.save_path)

    def plot_position(self, show=True, save=False):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[0].plot(self.dynamics["t"], self.dynamics["x"])
        plt.setp(axes[0], ylabel="x [m]")

        axes[1].plot(self.dynamics["t"], self.dynamics["y"])
        plt.setp(axes[1], ylabel="y [m]")

        axes[2].plot(self.dynamics["t"], self.dynamics["z"])
        plt.setp(axes[2], ylabel="z [m]")

        plt.xlabel("Time [s]")

        if save:
            self.save_plot("Position")
        if show:
            plt.show()

    def plot_attitude(self, show=True, save=False):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[0].plot(self.dynamics["t"], self.dynamics["psi_[deg]"])
        plt.setp(axes[0], ylabel="yaw [deg]")

        axes[1].plot(self.dynamics["t"], self.dynamics["theta_[deg]"])
        plt.setp(axes[1], ylabel="pitch [deg]")

        axes[2].plot(self.dynamics["t"], self.dynamics["phi_[deg]"])
        plt.setp(axes[2], ylabel="roll [deg]")

        plt.xlabel("Time [s]")

        if save:
            self.save_plot("Attitude")
        if show:
            plt.show()
        plt.close()

    def plot_inertial(self, show=True, save=False):
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        axes[0, 0].plot(self.dynamics["t"], round(self.dynamics["x"], 6), label="Output")
        axes[0, 0].plot(self.dynamics.t, self.control.setpoint_x, label="Input")
        axes[0, 0].legend()
        plt.setp(axes[0, 0], title="Displacement_X")
        plt.setp(axes[0, 0], ylabel="x [m]")
        plt.setp(axes[0, 0], xlabel="Time [s]")

        axes[1, 0].plot(self.dynamics["t"], round(self.dynamics["y"], 6), label="Output")
        axes[1, 0].plot(self.dynamics.t, self.control.setpoint_y, label="Input")
        axes[1, 0].legend()
        plt.setp(axes[1, 0], title="Displacement_Y")
        plt.setp(axes[1, 0], ylabel="y [m]")
        plt.setp(axes[1, 0], xlabel="Time [s]")

        axes[2, 0].plot(self.dynamics["t"], round(self.dynamics["z"], 6), label="Output")
        axes[2, 0].plot(self.dynamics.t, self.control.setpoint_z, label="Input")
        axes[2, 0].legend()
        plt.setp(axes[2, 0], title="Displacement_Z")
        plt.setp(axes[2, 0], ylabel="h [m]")
        plt.setp(axes[2, 0], xlabel="Time [s]")

        axes[0, 1].plot(self.dynamics["t"], round(self.dynamics["psi_[deg]"], 6), label="Output")
        axes[0, 1].plot(self.dynamics.t, np.degrees(self.control.setpoint_yaw), label="Input")
        # axes[0, 1].plot(self.dynamics.t, self.estimate.psi_deg, label="State Estimate", alpha=0.7)
        axes[0, 1].legend()
        plt.setp(axes[0, 1], title="Yaw")
        plt.setp(axes[0, 1], ylabel="yaw [deg]")
        plt.setp(axes[0, 1], xlabel="Time [s]")

        axes[1, 1].plot(self.dynamics["t"], round(self.dynamics["theta_[deg]"], 6), label="Output")
        axes[1, 1].plot(self.dynamics.t, self.control.setpoint_pitch, label="Input")
        axes[1, 1].plot(self.dynamics.t, self.estimate.theta_deg, label="State Estimate", alpha=0.7)
        axes[1, 1].legend()
        plt.setp(axes[1, 1], title="Pitch")
        plt.setp(axes[1, 1], ylabel="pitch [deg]")
        plt.setp(axes[1, 1], xlabel="Time [s]")

        axes[2, 1].plot(self.dynamics["t"], round(self.dynamics["phi_[deg]"], 6), label="Output")
        axes[2, 1].plot(self.dynamics.t, self.control.setpoint_roll, label="Input")
        axes[2, 1].plot(self.dynamics.t, self.estimate.phi_deg, label="State Estimate", alpha=0.7)
        axes[2, 1].legend()
        plt.setp(axes[2, 1], title="Roll")
        plt.setp(axes[2, 1], ylabel="roll [deg]")
        plt.setp(axes[2, 1], xlabel="Time [s]")

        if save:
            self.save_plot("Inertial Position and Attitude")
        if show:
            plt.show()
        plt.close()

    def plot_3d(self, show=True, save=True):
        ax = plt.axes(projection='3d')
        ax.plot3D(self.dynamics.x, self.dynamics.y, self.dynamics.z)
        plt.title("Inertial Frame Flight Path")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        ax.set_zlabel("Z [m]")

        if save:
            self.save_plot("3D Flight Path")
        if show:
            plt.show()
        plt.close()

    def plot_thrust(self, show=True, save=True):
        fig, axes = plt.subplots(2, 1, figsize=(10, 7))

        axes[0].plot(self.dynamics.t, self.dynamics.F1, label="F1", linewidth=1, alpha=0.7)
        axes[0].plot(self.dynamics.t, self.dynamics.F2, label="F2", linewidth=1, alpha=0.75)
        axes[0].plot(self.dynamics.t, self.dynamics.F3, label="F3", linewidth=1, alpha=0.8)
        axes[0].plot(self.dynamics.t, self.dynamics.F4, label="F4", linewidth=1, alpha=0.85)
        plt.setp(axes[0], ylabel="Propeller Thrust [N]")
        axes[0].legend()

        axes[1].plot(self.dynamics.t, self.dynamics.omega_1, label=r"$\omega$_1", linewidth=1, alpha=0.85)
        axes[1].plot(self.dynamics.t, self.dynamics.omega_2, label=r"$\omega$_2", linewidth=1, alpha=0.85)
        axes[1].plot(self.dynamics.t, self.dynamics.omega_3, label=r"$\omega$_3", linewidth=1, alpha=0.85)
        axes[1].plot(self.dynamics.t, self.dynamics.omega_4, label=r"$\omega$_4", linewidth=1, alpha=0.85)
        plt.setp(axes[1], ylabel="Propeller Speed [RPM]")
        plt.setp(axes[1], xlabel="Time [s]")
        axes[1].legend()

        # _, ax1 = plt.subplots(figsize=(12, 8))
        # plt.title("Motor Thrust and RPM")
        # plt.xlabel("Time [s]")
        # ax1.plot(self.dynamics.t, self.dynamics.F1, label="F1")
        # ax1.plot(self.dynamics.t, self.dynamics.F2, label="F2")
        # ax1.plot(self.dynamics.t, self.dynamics.F3, label="F3")
        # ax1.plot(self.dynamics.t, self.dynamics.F4, label="F4")
        # ax1.set_ylabel("Thrust [N]")
        # ax1.legend()
        #
        # ax2 = ax1.twinx()
        # ax2.plot(self.dynamics.t, self.dynamics.omega_1, label="omega_1", linestyle="--")
        # ax2.plot(self.dynamics.t, self.dynamics.omega_2, label="omega_2", linestyle="--")
        # ax2.plot(self.dynamics.t, self.dynamics.omega_3, label="omega_3", linestyle="--")
        # ax2.plot(self.dynamics.t, self.dynamics.omega_4, label="omega_4", linestyle="--")
        # ax2.set_ylabel("Motor Speed [RPM]")
        # ax2.legend()

        if save:
            self.save_plot("Motor Thrusts")
        if show:
            plt.show()
        plt.close()

    def plot_induced_velocity(self, show=True, save=False):
        for motor_index in range(len(self.thrust_data)):
            plt.plot(self.thrust_data[motor_index].time, self.thrust_data[motor_index].v_i,
                     label="Motor {}".format(motor_index + 1))

        plt.xlabel("Time [s]")
        plt.ylabel("v_i [m/s]")
        plt.legend()
        plt.title("Propeller Induced Velocity")

        if save:
            self.save_plot("Propeller Induced Velocity")
        if show:
            plt.show()
        plt.close()

    def plot_sensors(self, show=True, save=False) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        axes[0].plot(self.sensor_data.time, self.sensor_data.acc_gn_z, label="Gaussian Noise", alpha=0.6)
        axes[0].plot(self.sensor_data.time, self.estimate.acc_filt_z, label="Filtered", alpha=0.8)
        axes[0].plot(self.sensor_data.time, self.sensor_data.acc_z, label="True")
        axes[0].legend()
        plt.setp(axes[0], title="Accelerometer Data")
        plt.setp(axes[0], ylabel="a_z [m/s^2]")
        plt.setp(axes[0], xlabel="Time [s]")

        axes[1].plot(self.sensor_data.time, self.sensor_data.gyro_gn_x, label="Gaussian Noise", alpha=0.6)
        axes[1].plot(self.sensor_data.time, self.estimate.gyro_filt_x, label="Filtered", alpha=0.8)
        axes[1].plot(self.sensor_data.time, self.sensor_data.gyro_x, label="True")
        axes[1].legend()
        plt.setp(axes[1], title="Gyroscope Data")
        plt.setp(axes[1], ylabel="Omega [rad/s]")
        plt.setp(axes[1], xlabel="Time [s]")

        if save:
            self.save_plot("Sensor Data")
        if show:
            plt.show()
        plt.close()

    def plot_disturbance(self, show=True, save=True):

        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        plt.subplots_adjust(wspace=0.2, hspace=0.4)

        axes[0].plot(self.dynamics.t, self.dynamics.x, label="Gaussian Noise")

        plt.rcParams['axes.grid'] = False
        ax1 = axes[0].twinx()
        ax1.plot(self.disturbance.t, self.disturbance.F_x, color="o")

        if save:
            self.save_plot("Disturbance Response [X]")
        if show:
            plt.show()
        plt.close()

    def save_plot(self, file_name: str) -> None:
        plt.savefig(os.path.join(self.save_path, "{}.png".format(file_name)), bbox_inches="tight")
