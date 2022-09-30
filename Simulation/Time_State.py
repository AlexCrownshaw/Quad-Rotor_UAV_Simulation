import numpy as np

from Simulation.Dynamics_Model import DynamicsModel


class TimeState(DynamicsModel):

    u_dot, v_dot, w_dot = float
    p_dot, q_dot, r_dot = float
    x_dot, y_dot, z_dot = float
    psi_dot, theta_dot, phi_dot = float
    
    u, v, w = float
    p, q, r = float
    x, y, z = float
    psi, theta, phi = float

    def __init__(self, prev_time_instance: object, motor_thrusts: np.array):
        self.prev_time_instance = prev_time_instance
        self.motor_thrusts = motor_thrusts
        self.thrust_vector = np.array([0, 0, -np.sum(motor_thrusts)]).reshape(3, 1)
        self.moments = self.compute_moments(self.motor_thrusts)
