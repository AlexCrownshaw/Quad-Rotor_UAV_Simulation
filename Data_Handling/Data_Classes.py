import numpy as np


class TimeState:

    def __init__(self, state_vector: np.array):
        self.state_vector = state_vector
        self.body_rate_vector = state_vector[3:6]

        self.u, self.v, self.w = state_vector[0], state_vector[1], state_vector[2]
        self.p, self.q, self.r = state_vector[3], state_vector[4], state_vector[5]
        self.x, self.y, self.z = state_vector[6], state_vector[7], state_vector[8]

        # Unwrap Euler angles
        psi = float(state_vector[9])
        self.psi = self.unwrap_euler_angle(psi)
        self.theta = self.unwrap_euler_angle(float(state_vector[10]))
        self.phi = self.unwrap_euler_angle(float(state_vector[11]))

    @staticmethod
    def unwrap_euler_angle(euler_angle: float) -> float:
        if euler_angle > 0:
            return euler_angle % (2 * np.pi)
        elif euler_angle < 0:
            return euler_angle % (-2 * np.pi)
        else:
            return euler_angle


class StateDerivative:

    def __init__(self, derivative_vector: np.array):
        self.derivative_vector = derivative_vector
        self.acc_body_vector = self.derivative_vector[0:3]

        self.u_dot = self.derivative_vector[0]
        self.v_dot = self.derivative_vector[1]
        self.w_dot = self.derivative_vector[2]
        self.p_dot = self.derivative_vector[3]
        self.q_dot = self.derivative_vector[4]
        self.r_dot = self.derivative_vector[5]
        self.x_dot = self.derivative_vector[6]
        self.y_dot = self.derivative_vector[7]
        self.z_dot = self.derivative_vector[8]
        self.psi_dot = self.derivative_vector[9]
        self.theta_dot = self.derivative_vector[10]
        self.phi_dot = self.derivative_vector[11]


class SensorState:

    def __init__(self, acc, gyro):
        self.acc = acc
        self.gyro = gyro


class EstimateState:

    def __init__(self, attitude_vector: np.array):
        self.psi = attitude_vector[0]
        self.theta = attitude_vector[1]
        self.phi = attitude_vector[2]
