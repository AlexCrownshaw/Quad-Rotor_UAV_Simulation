import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Simulation.Time_State import TimeState


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
                                        "input_y", "input_z", "input_yaw", "input_pitch", "input_roll",
                                        "error_y", "error_z", "error_yaw", "error_pitch", "error_roll",
                                        "error_sum_y", "error_sum_z", "error_sum_yaw", "error_sum_pitch",
                                        "error_sum_roll",
                                        "d_error_y", "d_error_z", "d_error_yaw", "d_error_pitch", "d_error_roll",
                                        "output_x", "output_y", "output_z", "output_yaw", "output_pitch",
                                        "output_roll"])

        self.df["time"] = self.control_data_list[0]["time"]

        self.df["setpoint_x"] = self.control_data_list[0]["setpoint"]
        self.df["setpoint_y"] = self.control_data_list[1]["setpoint"]
        self.df["setpoint_z"] = self.control_data_list[2]["setpoint"]
        self.df["setpoint_yaw"] = self.control_data_list[3]["setpoint"]

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

        self.df["error_sum_x"] = self.control_data_list[0]["error_sum"]
        self.df["error_sum_y"] = self.control_data_list[1]["error_sum"]
        self.df["error_sum_z"] = self.control_data_list[2]["error_sum"]
        self.df["error_sum_yaw"] = self.control_data_list[3]["error_sum"]
        self.df["error_sum_pitch"] = self.control_data_list[4]["error_sum"]
        self.df["error_sum_roll"] = self.control_data_list[5]["error_sum"]

        self.df["d_error_x"] = self.control_data_list[0]["d_error"]
        self.df["d_error_y"] = self.control_data_list[1]["d_error"]
        self.df["d_error_z"] = self.control_data_list[2]["d_error"]
        self.df["d_error_yaw"] = self.control_data_list[3]["d_error"]
        self.df["d_error_pitch"] = self.control_data_list[4]["d_error"]
        self.df["d_error_roll"] = self.control_data_list[5]["d_error"]

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

    def append_thrust_data(self, motor_index, t, V, v_i, V_prime, T) -> None:
        self.motor_data[motor_index].loc[len(self.motor_data[motor_index])] = [t, float(V[0]), float(V[1]), float(V[2]),
                                                                               v_i, V_prime, float(T)]


class DataProcessor:

    def __init__(self, dynamics: DynamicsData, control: ControlData, thrust_data: list, save_path: str):
        self.dynamics = dynamics.df
        self.control = control.df
        self.thrust_data = thrust_data.motor_data

        self.save_path = os.path.join(save_path, str(time.strftime("%d-%m-%y %H-%M-%S")))
        os.mkdir(self.save_path)
        self.dynamics.to_csv(os.path.join(self.save_path, "Dynamics_Data.csv"))
        self.control.to_csv(os.path.join(self.save_path, "Control_Data.csv"))

        motor_data_save_path = os.path.join(self.save_path, "Motor Thrust Data")
        os.mkdir(motor_data_save_path)
        for motor_index in range(len(self.thrust_data)):
            self.thrust_data[motor_index].to_csv(os.path.join(motor_data_save_path, "Motor {}.csv".format(motor_index)))

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

    def plot_inertial(self, show=True, save=False):
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))

        axes[0, 0].plot(self.dynamics["t"], self.dynamics["x"])
        plt.setp(axes[0, 0], title="Position_x")
        plt.setp(axes[0, 0], ylabel="x [m]")

        axes[1, 0].plot(self.dynamics["t"], self.dynamics["y"])
        plt.setp(axes[1, 0], title="Position_y")
        plt.setp(axes[1, 0], ylabel="y [m]")

        axes[2, 0].plot(self.dynamics["t"], self.dynamics["z"])
        plt.setp(axes[2, 0], title="Position_z")
        plt.setp(axes[2, 0], ylabel="h [m]")

        axes[2, 1].plot(self.dynamics["t"], self.dynamics["psi_[deg]"])
        plt.setp(axes[0, 1], title="Yaw")
        plt.setp(axes[0, 1], ylabel="yaw [deg]")

        axes[0, 1].plot(self.dynamics["t"], self.dynamics["theta_[deg]"])
        plt.setp(axes[1, 1], title="Pitch")
        plt.setp(axes[1, 1], ylabel="pitch [deg]")

        axes[1, 1].plot(self.dynamics["t"], self.dynamics["phi_[deg]"])
        plt.setp(axes[2, 1], title="Roll")
        plt.setp(axes[2, 1], ylabel="roll [deg]")

        plt.xlabel("Time [s]")

        if save:
            self.save_plot("Inertial Position and Attitude")
        if show:
            plt.show()

    def plot_3d(self, show=True, save=True):
        ax = plt.axes(projection='3d')
        ax.plot3D(self.dynamics.x, self.dynamics.y, self.dynamics.z)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")

        if save:
            self.save_plot("3D Flight Path")
        if show:
            plt.show()

    def plot_thrust(self, show=True, save=True):
        _, ax1 = plt.subplots(figsize=(12, 8))
        plt.title("Motor Thrust and RPM")
        plt.xlabel("Time [s]")
        ax1.plot(self.dynamics.t, self.dynamics.F1, label="F1")
        ax1.plot(self.dynamics.t, self.dynamics.F2, label="F2")
        ax1.plot(self.dynamics.t, self.dynamics.F3, label="F3")
        ax1.plot(self.dynamics.t, self.dynamics.F4, label="F4")
        ax1.set_ylabel("Thrust [N]")
        ax1.legend()

        ax2 = ax1.twinx()
        ax2.plot(self.dynamics.t, self.dynamics.omega_1, label="omega_1", linestyle="--")
        ax2.plot(self.dynamics.t, self.dynamics.omega_2, label="omega_2", linestyle="--")
        ax2.plot(self.dynamics.t, self.dynamics.omega_3, label="omega_3", linestyle="--")
        ax2.plot(self.dynamics.t, self.dynamics.omega_4, label="omega_4", linestyle="--")
        ax2.set_ylabel("Motor Speed [RPM]")
        ax2.legend()

        if save:
            self.save_plot("Motor Thrusts")
        if show:
            plt.show()

    def save_plot(self, file_name: str) -> None:
        plt.savefig(os.path.join(self.save_path, "{}.png".format(file_name)), bbox_inches="tight")
