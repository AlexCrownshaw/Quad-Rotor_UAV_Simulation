import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Simulation.Time_State import TimeState


class DynamicsData:
    plt.rcParams['axes.grid'] = True

    def __init__(self):
        self.df = pd.DataFrame(columns=["t", "x", "y", "z", "psi_[rad]", "theta_[rad]", "phi_[rad]", "psi_[deg]",
                                        "theta_[deg]", "phi_[deg]", "u", "v", "w", "p", "q", "r",
                                        "F1", "F2", "F3", "F4"])

    def append_time_instance(self, t: float, X: TimeState, U: np.array):
        self.df.loc[len(self.df)] = [t, X.x, X.y, X.z, X.psi, X.theta, X.phi,
                                     np.degrees(X.psi), np.degrees(X.theta), np.degrees(X.phi),
                                     X.u, X.v, X.w, X.p, X.q, X.r, U[0], U[1], U[2], U[3]]

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


class Plotter:

    def __init__(self, dynamics: DynamicsData, control: ControlData):
        self.dynamics = dynamics.df
        self.control = control.df

    def plot_position(self, show=True):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[0].plot(self.dynamics["t"], self.dynamics["x"])
        plt.setp(axes[0], ylabel="x [m]")

        axes[1].plot(self.dynamics["t"], self.dynamics["y"])
        plt.setp(axes[1], ylabel="y [m]")

        axes[2].plot(self.dynamics["t"], self.dynamics["z"])
        plt.setp(axes[2], ylabel="z [m]")

        plt.xlabel("Time [s]")
        if show:
            plt.show()

    def plot_attitude(self, show=True):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[0].plot(self.dynamics["t"], self.dynamics["psi_[deg]"])
        plt.setp(axes[0], ylabel="yaw [degrees]")

        axes[1].plot(self.dynamics["t"], self.dynamics["theta_[deg]"])
        plt.setp(axes[1], ylabel="pitch [degrees]")

        axes[2].plot(self.dynamics["t"], self.dynamics["phi_[deg]"])
        plt.setp(axes[2], ylabel="roll [degrees]")

        plt.xlabel("Time [s]")

        if show:
            plt.show()

    def plot_inertial(self, show=True):
        fig, axes = plt.subplots(3, 2, figsize=(15, 8))

        axes[0, 0].plot(self.dynamics["t"], self.dynamics["x"])
        plt.setp(axes[0, 0], ylabel="x [m]")

        axes[1, 0].plot(self.dynamics["t"], self.dynamics["y"])
        plt.setp(axes[1, 0], ylabel="y [m]")

        axes[2, 0].plot(self.dynamics["t"], self.dynamics["z"])
        plt.setp(axes[2, 0], ylabel="z [m]")

        axes[0, 1].plot(self.dynamics["t"], self.dynamics["theta_[deg]"])
        plt.setp(axes[0, 1], ylabel="pitch [degrees]")

        axes[1, 1].plot(self.dynamics["t"], self.dynamics["phi_[deg]"])
        plt.setp(axes[1, 1], ylabel="roll [degrees]")

        axes[2, 1].plot(self.dynamics["t"], self.dynamics["psi_[deg]"])
        plt.setp(axes[2, 1], ylabel="yaw [degrees]")

        plt.xlabel("Time [s]")

        if show:
            plt.show()

    def plot_3d(self):
        ax = plt.axes(projection='3d')
        ax.plot3D(self.dynamics.x, self.dynamics.y, self.dynamics.z)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.show()
