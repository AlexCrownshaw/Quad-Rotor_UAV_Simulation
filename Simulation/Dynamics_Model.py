import numpy as np

from typing import Tuple
from Data_Handling.Data_Classes import TimeState, StateDerivative


class DynamicsModel:
    g = 9.81  # m/s^2

    def __init__(self, properties, dimensions):
        self.properties = properties
        self.dimensions = dimensions

        # Define simulation constants
        # self.moment_inertia_matrix = np.array([self.properties["I_xx"], 0, 0, 0, self.properties["I_yy"], 0, 0, 0,
        #                                        self.properties["I_zz"]]).reshape(3, 3)
        self.gravity_vector_inertial = np.array([0, 0, self.properties["mass"] * 9.81]).reshape(3, 1)

        print(self)

    def compute_moments(self, T: np.array, Q: np.array) -> Tuple[float, float, float]:
        L = float(np.sum(T * np.array([self.dimensions["d_y"], -self.dimensions["d_y"],
                                       -self.dimensions["d_y"], self.dimensions["d_y"]])))
        M = float(np.sum(T * np.array([self.dimensions["d_x"], -self.dimensions["d_x"],
                                       self.dimensions["d_x"], -self.dimensions["d_x"]])))
        N = float(np.sum(Q))

        return L, M, N

    @staticmethod
    def thrust_vector_body(motor_thrusts) -> np.array:
        return np.array([0, 0, -np.sum(motor_thrusts)]).reshape(3, 1)

    def gravity_vector_body(self, theta, phi) -> np.array:
        return np.array([-self.g * np.sin(theta), self.g * np.sin(phi) * np.cos(theta),
                         self.g * np.cos(phi) * np.cos(theta)]).reshape(3, 1)

    def compute_motor_thrusts(self, motor_speeds: np.array) -> np.array:
        pass

    def compute_state_derivative(self, X: np.array, T: np.array, Q: np.array, D: np.array) -> np.array:
        # Apply suitable variable names
        u, v, w = X[0], X[1], X[2]
        p, q, r = X[3], X[4], X[5]
        psi, theta, phi = X[9], X[10], X[11]

        # Pre-calculate trigonometric terms
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        cos_psi, sin_psi = np.cos(psi), np.sin(psi)

        # Forces sum up in the body z-axis
        F_z = -np.sum(T)

        # Compute Moments
        L, M, N = self.compute_moments(T, Q)

        # Define state derivative object
        X_dot = np.zeros(12)

        # Compute body frame acceleration
        X_dot[0] = -self.g * sin_theta + r * v - q * w  # u_dot
        X_dot[1] = self.g * sin_phi * cos_theta - r * u + p * w  # v_dot
        X_dot[2] = 1 / self.properties["mass"] * F_z + self.g * cos_phi * cos_theta + q * u - p * v  # w_dot

        # Compute rotational body frame acceleration
        X_dot[3] = 1 / self.properties["I_xx"] * (
                L + (self.properties["I_yy"] - self.properties["I_zz"]) * q * r)  # p_dot
        X_dot[4] = 1 / self.properties["I_yy"] * (
                M + (self.properties["I_zz"] - self.properties["I_xx"]) * p * r)  # q_dot
        X_dot[5] = 1 / self.properties["I_zz"] * (
                N + (self.properties["I_xx"] - self.properties["I_yy"]) * p * q)  # r_dot

        # Compute linear inertial acceleration
        X_dot[6] = cos_theta * cos_psi * u + (-cos_phi * sin_psi + sin_phi * sin_theta * cos_psi) * v + \
                   (sin_phi * sin_psi + cos_phi * sin_theta * cos_psi) * w  # x_dot
        X_dot[7] = cos_theta * sin_psi * u + (cos_phi * cos_psi + sin_phi * sin_theta * sin_psi) * v + \
                   (-sin_phi * cos_psi + cos_phi * sin_theta * sin_psi) * w  # y_dot
        X_dot[8] = -1 * (-sin_theta * u + sin_phi * cos_theta * v + cos_phi * cos_theta * w)  # z_dot

        # Add inertial frame acceleration due to gravity.
        X_dot[6] += D[0] / self.properties["mass"]
        X_dot[7] += D[1] / self.properties["mass"]
        X_dot[8] += D[2] / self.properties["mass"]

        # Compute rotational inertial acceleration
        X_dot[9] = (q * sin_phi + r * cos_phi) / cos_theta  # psi_dot
        X_dot[10] = q * cos_phi - r * sin_phi  # theta_dot
        X_dot[11] = p + (q * sin_phi + r * cos_phi) * sin_theta / cos_theta  # phi_dot

        return X_dot

    def rk4(self, X: TimeState, T: np.array, Q: np.array, D: np.array, dt: float) -> Tuple[TimeState, StateDerivative]:
        k1 = self.compute_state_derivative(X.state_vector, T, Q, D)
        k2 = self.compute_state_derivative(X.state_vector + k1 * dt / 2, T, Q, D)
        k3 = self.compute_state_derivative(X.state_vector + k2 * dt / 2, T, Q, D)
        k4 = self.compute_state_derivative(X.state_vector + k3 * dt, T, Q, D)

        X_dot_calc = 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X_calc = X.state_vector + X_dot_calc * dt

        return TimeState(X_calc), StateDerivative(X_dot_calc)
