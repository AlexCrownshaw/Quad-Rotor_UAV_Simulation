import numpy as np


class TimeState:
    
    def __init__(self, state_vector: np.array):
        self.state_vector = state_vector
        
        self.u, self.v, self.w = state_vector[0], state_vector[1], state_vector[2]
        self.p, self.q, self.r = state_vector[3], state_vector[4], state_vector[5]
        self.x, self.y, self.z = state_vector[6], state_vector[7], state_vector[8]

        # Unwrap Euler angles
        self.psi, self.theta = state_vector[9] % 2 * np.pi, state_vector[10] % 2 * np.pi
        self.phi = state_vector[11] % 2 * np.pi


class StateDerivative:

    def __init__(self, derivative_vector: np.array):
        self.derivative_vector = derivative_vector

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
