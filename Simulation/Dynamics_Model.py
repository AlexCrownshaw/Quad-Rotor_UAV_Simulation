import json

import numpy as np

from typing import Tuple
from Simulation.Time_State import TimeState, StateDerivative


class DynamicsModel:

    g = 9.81  # m/s^2

    def __init__(self, t_delta, structural_json_path, flight_path_json_path):
        self.t_delta = t_delta

        # Load simulation config properties
        self.properties, self.dimensions = self.load_structural_json(structural_json_path)
        self.boundary_conditions, self.maneuvers = self.load_flight_path_json(flight_path_json_path)

        # Define simulation constants
        self.moment_inertia_matrix = np.array([self.properties["I_x"], 0, 0, 0, self.properties["I_y"], 0, 0, 0,
                                               self.properties["I_z"]]).reshape(3, 3)
        self.gravity_vector_inertial = np.array([0, 0, self.properties["mass"] * 9.81]).reshape(3, 1)

        print(self)

    @staticmethod
    def load_structural_json(json_path: str) -> Tuple[dict, dict]:
        with open(json_path) as json_file:
            json_object = json.load(json_file)
            return json_object["properties"], json_object["dimensions"]

    @staticmethod
    def load_flight_path_json(json_path: str) -> Tuple[dict, dict]:
        with open(json_path) as json_file:
            json_object = json.load(json_file)
            return json_object["Boundary_Conditions"], json_object["Maneuvers"]

    def compute_moments(self, motor_thrusts: np.array) -> Tuple[float, float, float]:
        L = float(np.sum(motor_thrusts * np.array([self.dimensions["d_y"], -self.dimensions["d_y"],
                                                   -self.dimensions["d_y"], self.dimensions["d_y"]])))
        M = float(np.sum(motor_thrusts * np.array([self.dimensions["d_x"], -self.dimensions["d_x"],
                                                   self.dimensions["d_x"], -self.dimensions["d_x"]])))
        N = float(np.sum(np.array([0, 0, 0])))

        return L, M, N

    @staticmethod
    def thrust_vector_body(motor_thrusts) -> np.array:
        return np.array([0, 0, -np.sum(motor_thrusts)]).reshape(3, 1)

    def gravity_vector_body(self, theta, phi) -> np.array:
        return np.array([-self.g * np.sin(theta), self.g * np.sin(phi) * np.cos(theta),
                         self.g * np.cos(phi) * np.cos(theta)]).reshape(3, 1)

    def compute_state_derivative(self, X: TimeState, U) -> StateDerivative:

        # Pre-calculate trigonometric terms
        cos_phi = np.cos(X.phi)
        sin_phi = np.sin(X.phi)
        cos_theta = np.cos(X.theta)
        sin_theta = np.sin(X.theta)
        cos_psi = np.cos(X.psi)
        sin_psi = np.sin(X.psi)

        # Forces sum up in the body z-axis
        F_z = -np.sum(U.thrusts)

        # Compute Moments
        L, M, N = self.compute_moments(U.thrusts)

        # Define state derivative object
        sd = StateDerivative()

        # Compute linear body acceleration
        sd.u_dot = -self.g * sin_theta + X.r * X.v - X.q * X.w
        sd.v_dot = self.g * sin_phi * cos_theta - X.r * X.u + X.p * X.w
        sd.w_dot = 1 / self.properties["mass"] * F_z + self.g * cos_phi * cos_theta + X.q * X.u - X.p * X.v

        # Compute rotational body acceleration
        sd.p_dot = 1 / self.properties["I_xx"] * (L + (self.properties["I_yy"] - self.properties["I_zz"]) * X.q * X.r)
        sd.q_dot = 1 / self.properties["I_yy"] * (M + (self.properties["I_zz"] - self.properties["I_xx"]) * X.p * X.r)
        sd.r_dot = 1 / self.properties["I_zz"] * (N + (self.properties["I_xx"] - self.properties["I_yy"]) * X.p * X.q)

        # Compute linear inertial acceleration
        sd.x_dot = cos_theta * cos_psi * X.u + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * X.v + \
                   (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * X.w
        sd.y_dot = cos_theta * sin_psi * X.u + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * X.v + \
                   (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * X.w
        sd.z_dot = -1 * (-sin_theta * X.u + sin_phi * cos_theta * X.v + cos_phi * cos_theta * X.w)

        # Compute rotational inertial acceleration
        sd.psi_dot = (X.q * sin_phi + X.r * cos_phi) / cos_theta
        sd.theta_dot = X.q * cos_phi - X.r * sin_phi
        sd.phi_dot = X.p + (X.q * sin_phi + X.r * cos_phi) * sin_theta / cos_theta

        return sd

    def rk4(self) -> TimeState:
        pass

    @staticmethod
    def get_rotation_matrix(yaw, pitch, roll, transpose=False) -> np.array:
        r = np.array([np.cos(pitch) * np.cos(yaw),
                      np.cos(pitch) * np.sin(yaw),
                      -np.sin(pitch),
                      -np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw),
                      np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw),
                      np.sin(roll) * np.cos(pitch),
                      np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw),
                      -np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw),
                      np.cos(roll) * np.cos(pitch)]).reshape(3, 3)

        if transpose:
            r = r.transpose()

        return r
