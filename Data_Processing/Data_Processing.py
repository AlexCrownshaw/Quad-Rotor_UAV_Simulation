import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Simulation.Time_State import TimeState


class DataProcessing:

    plt.rcParams['axes.grid'] = True

    def __init__(self):
        self. sim_df = pd.DataFrame(columns=["t", "x", "y", "z", "psi_[rad]", "theta_[rad]", "phi_[rad]", "psi_[deg]",
                                             "theta_[deg]", "phi_[deg]", "u", "v", "w", "p", "q", "r",
                                             "F1", "F2", "F3", "F4"])

    def append_time_instance(self, t: float, X: TimeState, U: np.array):
        self.sim_df.loc[len(self.sim_df)] = [t, X.x, X.y, X.z, X.psi, X.theta, X.phi,
                                             np.degrees(X.psi), np.degrees(X.theta), np.degrees(X.phi),
                                             X.u, X.v, X.w, X.p, X.q, X.r, U[0], U[1], U[2], U[3]]

    def plot_variable(self, variable: str, show=False):

        plt.plot(self.sim_df["t"], self.sim_df[variable])

        if show:
            plt.show()

    def plot_position(self, show=True):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[0].plot(self.sim_df["t"], self.sim_df["x"])
        plt.setp(axes[0], ylabel="x [m]")

        axes[1].plot(self.sim_df["t"], self.sim_df["y"])
        plt.setp(axes[1], ylabel="y [m]")

        axes[2].plot(self.sim_df["t"], self.sim_df["z"])
        plt.setp(axes[2], ylabel="z [m]")

        plt.xlabel("Time [s]")
        if show:
            plt.show()

    def plot_attitude(self, show=True):
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))

        axes[0].plot(self.sim_df["t"], self.sim_df["psi_[deg]"])
        plt.setp(axes[0], ylabel="yaw [degrees]")

        axes[1].plot(self.sim_df["t"], self.sim_df["theta_[deg]"])
        plt.setp(axes[1], ylabel="pitch [degrees]")

        axes[2].plot(self.sim_df["t"], self.sim_df["phi_[deg]"])
        plt.setp(axes[2], ylabel="roll [degrees]")

        plt.xlabel("Time [s]")

        if show:
            plt.show()

    def plot_inertial(self, show=True):
        fig, axes = plt.subplots(3, 2, figsize=(15, 8))

        axes[0, 0].plot(self.sim_df["t"], self.sim_df["x"])
        plt.setp(axes[0, 0], ylabel="x [m]")

        axes[1, 0].plot(self.sim_df["t"], self.sim_df["y"])
        plt.setp(axes[1, 0], ylabel="y [m]")

        axes[2, 0].plot(self.sim_df["t"], self.sim_df["z"])
        plt.setp(axes[2, 0], ylabel="z [m]")

        axes[0, 1].plot(self.sim_df["t"], self.sim_df["theta_[deg]"])
        plt.setp(axes[0, 1], ylabel="pitch [degrees]")

        axes[1, 1].plot(self.sim_df["t"], self.sim_df["phi_[deg]"])
        plt.setp(axes[1, 1], ylabel="roll [degrees]")

        axes[2, 1].plot(self.sim_df["t"], self.sim_df["psi_[deg]"])
        plt.setp(axes[2, 1], ylabel="yaw [degrees]")

        plt.xlabel("Time [s]")

        if show:
            plt.show()

    def plot_3d(self):
        ax = plt.axes(projection='3d')
        ax.plot3D(self.sim_df.x, self.sim_df.y, self.sim_df.z)
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.show()
