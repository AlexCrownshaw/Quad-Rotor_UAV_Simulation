from typing import Tuple

import numpy as np

from Data_Handling.Data_Classes import SensorState, EstimateState
from Data_Handling.Data_Processing import StateEstimationData


class StateEstimation:

    TYPE = "comp_filt"  # "ekf"

    def __init__(self, frequency: float, acc_lpf_cutoff_freq: int, gyro_lpf_cutoff_freq: int, alpha):
        # Create low pass filter objects
        t_sample = 1 / frequency

        self.acc_lpf_x = RCLowPassFilter(t_sample, acc_lpf_cutoff_freq)
        self.acc_lpf_y = RCLowPassFilter(t_sample, acc_lpf_cutoff_freq)
        self.acc_lpf_z = RCLowPassFilter(t_sample, acc_lpf_cutoff_freq)

        self.gyro_lpf_p = RCLowPassFilter(t_sample, gyro_lpf_cutoff_freq)
        self.gyro_lpf_q = RCLowPassFilter(t_sample, gyro_lpf_cutoff_freq)
        self.gyro_lpf_r = RCLowPassFilter(t_sample, gyro_lpf_cutoff_freq)

        if self.TYPE == "comp_filt":
            self.comp_filt = ComplimentaryFilter(alpha, frequency)
        elif self.TYPE == "ekf":
            self.ekf = EKF(frequency)

        self.ekf = EKF(frequency)

        self.data = StateEstimationData()

    def compute_state_estimate(self, t: float, S: SensorState) -> EstimateState:
        # Filter sensor outputs
        acc_filt = np.transpose(np.array([self.acc_lpf_x.compute_lfp(S.acc[0]),
                                          self.acc_lpf_y.compute_lfp(S.acc[1]),
                                          self.acc_lpf_z.compute_lfp(S.acc[2])]))
        gyro_filt = np.transpose(np.array([self.gyro_lpf_p.compute_lfp(S.gyro[0]),
                                           self.gyro_lpf_q.compute_lfp(S.gyro[1]),
                                           self.gyro_lpf_r.compute_lfp(S.gyro[2])]))

        # Compute attitude estimate
        if self.TYPE == "comp_filt":
            phi, theta = self.comp_filt.compute_comp_filt(acc_filt, gyro_filt)

        elif self.TYPE == "ekf":
            self.ekf.ekf_prediction(gyro_filt)
            theta, phi = self.ekf.ekf_update(acc_filt)

        psi = 0
        attitude_vector = np.array([psi, theta, phi])

        # Append data
        self.data.append_data(t, attitude_vector, acc_filt, gyro_filt)

        # self.ekf.ekf_prediction(gyro_filt)
        # self.ekf.ekf_update(acc_filt)

        return EstimateState(attitude_vector)

    def return_data(self) -> StateEstimationData:
        return self.data


class ComplimentaryFilter:

    def __init__(self, alpha: float, frequency: int):
        self.alpha = alpha
        self.dt = 1/frequency

        self.phi_hat, self.theta_hat = 0, 0

    def compute_comp_filt(self, acc, gyro) -> Tuple[float, float]:
        a_x, a_y, a_z = acc
        p, q, r = gyro

        # Compute Euler angles according to acceleration data
        phi_hat_acc = np.arctan(a_y/a_z)
        theta_hat_acc = np.arcsin(a_x/9.81)

        # pre-compute common trigonometric terms
        sin_phi = np.sin(self.phi_hat)
        cos_phi = np.cos(self.phi_hat)
        tan_theta = np.tan(self.theta_hat)

        # Convert body rates to Euler rates
        phi_dot = p + tan_theta * (q * sin_phi + r * cos_phi)
        theta_dot = q * cos_phi - r * sin_phi

        # Compute state estimates
        self.phi_hat = phi_hat_acc * self.alpha + (1 - self.alpha) * (self.phi_hat + self.dt * phi_dot)
        self.theta_hat = -theta_hat_acc * self.alpha + (1 - self.alpha) * (self.theta_hat + self.dt * theta_dot)

        return self.phi_hat, self.theta_hat


class EKF:

    g = 9.81

    def __init__(self, operating_frequency):
        self.t_sample = 1 / operating_frequency

        self.psi, self.theta, self.phi = 0, 0, 0
        self.P = np.zeros(4)
        self.Q = np.eye(4)
        self.R = np.eye(4)

    def ekf_prediction(self, gyro: np.array) -> None:
        # Extract gyro sensor data
        p, q, r = gyro

        # Pre-compute common trig terms and check for division by zero error
        sin_phi, cos_phi = np.sin(self.phi), np.cos(self.phi)
        cos_theta, tan_theta = np.cos(self.theta), np.tan(self.theta)

        # Compute Euler rates based on previous state Euler angles
        # https://uk.mathworks.com/help/aeroblks/customvariablemass6dofeulerangles.html
        psi_dot = q * sin_phi / cos_phi + r * cos_phi / cos_theta
        theta_dot = q * cos_phi - r * sin_phi
        phi_dot = p + tan_theta * (q * sin_phi + r * cos_phi)

        # Perform Euler integration to compute new attitude f(x, u)
        self.psi = self.euler_integration(self.psi, psi_dot)
        self.theta = self.euler_integration(self.theta, theta_dot)
        self.phi = self.euler_integration(self.phi, phi_dot)

        # Recalculate common trig terms
        sin_phi, cos_phi = np.sin(self.phi), np.cos(self.phi)
        cos_theta, tan_theta = np.cos(self.theta), np.tan(self.theta)

        # Compute Jacobian matrix df(x, u)/dx
        A = np.array([tan_theta * (q * cos_phi - r * sin_phi),
                      (tan_theta ** 2 + 1) * (q * sin_phi + r * cos_phi),
                      -q * sin_phi - r * cos_phi, 0])

        # Update covariance matrix [P_n+1 = P_n + T * (A*P_n + P_n*A^T + Q)]
        self.P = np.array([self.t_sample * (2 * self.P[0] * A[0] + self.P[1] * A[1] + self.P[2] * A[1] + self.Q[0]) +
                           self.P[0],
                           self.t_sample * (self.P[0] * A[2] + self.P[1] * A[0] + self.P[1] * A[3] + self.P[3] * A[1] +
                                            self.Q[1]) + self.P[1],
                           self.t_sample * (self.P[0] * A[2] + self.P[2] * A[0] + self.P[2] * A[3] + self.P[3] * A[1] +
                                            self.Q[2]) + self.P[2],
                           self.t_sample * (self.P[1] * A[2] + self.P[2] * A[2] + 2 * self.P[3] * A[3] + self.Q[3]) +
                           self.P[3]])

    def ekf_update(self, acc: np.array) -> np.array:
        # Extract acc sensor data
        a_x, a_y, a_z = acc

        # Compute common trig terms
        sin_phi, cos_phi = np.sin(self.phi), np.cos(self.phi)
        sin_theta, cos_theta = np.sin(self.theta), np.cos(self.theta)

        # compute output function h(x, u)
        h = np.array([self.g * sin_theta,
                      self.g * (-cos_theta * sin_phi),
                      self.g * (-cos_theta * cos_phi)])

        # Compute Jacobian matrix dh(x, u)/dx
        C = np.array([0, self.g * cos_theta, -self.g * cos_phi * cos_theta, self.g * sin_phi * sin_theta,
                      self.g * sin_phi * cos_theta, self.g * sin_theta * cos_phi])
        C = np.pad(C, (0, 2), "constant").reshape(2, 4)
        print(C)

        C_T = np.transpose(C)
        k_a = self.P.dot(C_T)
        print(k_a)
        k_b = self.R + C.dot(self.P.dot(C_T))

        # Compute Kalman gain [K = P*C' * [C*P*C'+R]^-1]
        K = k_a * np.invert(k_b)

        # K = np.array([(self.P[0]*C[0] + self.P[1]*C[1])
        #               *(-self.P[0]*C[2]**2 - self.P[1]*C[2]*C[3] - self.P[2]*C[2]*C[3] - self.P[3]*C[3]**2 - self.R[3])
        #               /(-self.P[0]*self.P[3]*C[0]**2*C[3]**2 + 2*self.P[0]*self.P[3]*C[0]*C[1]*C[2]*C[3]
        #                 - self.P[0]*self.P[3]*C[1]**2*C[2]**2 - self.P[0]*C[0]**2*self.R[3]
        #                 + self.P[0]*C[0]*C[2]*self.R[1] + self.P[0]*C[0]*C[2]*self.R[2] - self.P[0]*C[2]**2*self.R[0]
        #                 + self.P[1]*self.P[2]*C[0]**2*C[3]**2 - 2*self.P[1]*self.P[2]*C[0]*C[1]*C[2]*C[3]
        #                 + self.P[1]*self.P[2]*C[1]**2*C[2]**2 - self.P[1]*C[0]*C[1]*self.R[3]
        #                 + self.P[1]*C[0]*C[3]*self.R[2] + self.P[1]*C[1]*C[2]*self.R[1]
        #                 - self.P[1]*C[2]*C[3]*self.R[0] - self.P[2]*C[0]*C[1]*self.R[3] + self.P[2]*C[0]*C[3]*self.R[1]
        #                 + self.P[2]*C[1]*C[2]*self.R[2] - self.P[2]*C[2]*C[3]*self.R[0] - self.P[3]*C[1]**2*self.R[3]
        #                 + self.P[3]*C[1]*C[3]*self.R[1] + self.P[3]*C[1]*C[3]*self.R[2] - self.P[3]*C[3]**2*self.R[0]
        #                 - self.R[0]*self.R[3] + self.R[1]*self.R[2]) + (self.P[0]*C[2] + self.P[1]*C[3])*
        #               (self.P[0]*C[0]*C[2] + self.P[1]*C[1]*C[2] + self.P[2]*C[0]*C[3] + self.P[3]*C[1]*C[3]
        #                + self.R[2])/(-self.P[0]*self.P[3]*C[0]**2*C[3]**2 + 2*self.P[0]*self.P[3]*C[0]*C[1]*C[2]*C[3]
        #                              - self.P[0]*self.P[3]*C[1]**2*C[2]**2 - self.P[0]*C[0]**2*self.R[3]
        #                              + self.P[0]*C[0]*C[2]*self.R[1] + self.P[0]*C[0]*C[2]*self.R[2]
        #                              - self.P[0]*C[2]**2*self.R[0] + self.P[1]*self.P[2]*C[0]**2*C[3]**2
        #                              - 2*self.P[1]*self.P[2]*C[0]*C[1]*C[2]*C[3] + self.P[1]*self.P[2]*C[1]**2*C[2]**2
        #                              - self.P[1]*C[0]*C[1]*self.R[3] + self.P[1]*C[0]*C[3]*self.R[2]
        #                              + self.P[1]*C[1]*C[2]*self.R[1] - self.P[1]*C[2]*C[3]*self.R[0]
        #                              - self.P[2]*C[0]*C[1]*self.R[3] + self.P[2]*C[0]*C[3]*self.R[1]
        #                              + self.P[2]*C[1]*C[2]*self.R[2] - self.P[2]*C[2]*C[3]*self.R[0]
        #                              - self.P[3]*C[1]**2*self.R[3] + self.P[3]*C[1]*C[3]*self.R[1]
        #                              + self.P[3]*C[1]*C[3]*self.R[2] - self.P[3]*C[3]**2*self.R[0]
        #                              - self.R[0]*self.R[3] + self.R[1]*self.R[2]),
        #               (self.P[0]*C[0] + self.P[1]*C[1])*(self.P[0]*C[0]*C[2] + self.P[1]*C[0]*C[3]
        #                                                  + self.P[2]*C[1]*C[2] + self.P[3]*C[1]*C[3] + self.R[1])
        #               /(-self.P[0]*self.P[3]*C[0]**2*C[3]**2 + 2*self.P[0]*self.P[3]*C[0]*C[1]*C[2]*C[3]
        #                 - self.P[0]*self.P[3]*C[1]**2*C[2]**2 - self.P[0]*C[0]**2*self.R[3]
        #                 + self.P[0]*C[0]*C[2]*self.R[1] + self.P[0]*C[0]*C[2]*self.R[2] - self.P[0]*C[2]**2*self.R[0]
        #                 + self.P[1]*self.P[2]*C[0]**2*C[3]**2 - 2*self.P[1]*self.P[2]*C[0]*C[1]*C[2]*C[3]
        #                 + self.P[1]*self.P[2]*C[1]**2*C[2]**2 - self.P[1]*C[0]*C[1]*self.R[3]
        #                 + self.P[1]*C[0]*C[3]*self.R[2] + self.P[1]*C[1]*C[2]*self.R[1] - self.P[1]*C[2]*C[3]*self.R[0]
        #                 - self.P[2]*C[0]*C[1]*self.R[3] + self.P[2]*C[0]*C[3]*self.R[1] + self.P[2]*C[1]*C[2]*self.R[2]
        #                 - self.P[2]*C[2]*C[3]*self.R[0] - self.P[3]*C[1]**2*self.R[3] + self.P[3]*C[1]*C[3]*self.R[1]
        #                 + self.P[3]*C[1]*C[3]*self.R[2] - self.P[3]*C[3]**2*self.R[0] - self.R[0]*self.R[3]
        #                 + self.R[1]*self.R[2]) + (self.P[0]*C[2] + self.P[1]*C[3])
        #               *(-self.P[0]*C[0]**2 - self.P[1]*C[0]*C[1] - self.P[2]*C[0]*C[1] - self.P[3]*C[1]**2 - self.R[0])
        #               /(-self.P[0]*self.P[3]*C[0]**2*C[3]**2 + 2*self.P[0]*self.P[3]*C[0]*C[1]*C[2]*C[3]
        #                 - self.P[0]*self.P[3]*C[1]**2*C[2]**2 - self.P[0]*C[0]**2*self.R[3]
        #                 + self.P[0]*C[0]*C[2]*self.R[1] + self.P[0]*C[0]*C[2]*self.R[2] - self.P[0]*C[2]**2*self.R[0]
        #                 + self.P[1]*self.P[2]*C[0]**2*C[3]**2 - 2*self.P[1]*self.P[2]*C[0]*C[1]*C[2]*C[3]
        #                 + self.P[1]*self.P[2]*C[1]**2*C[2]**2 - self.P[1]*C[0]*C[1]*self.R[3]
        #                 + self.P[1]*C[0]*C[3]*self.R[2] + self.P[1]*C[1]*C[2]*self.R[1] - self.P[1]*C[2]*C[3]*self.R[0]
        #                 - self.P[2]*C[0]*C[1]*self.R[3] + self.P[2]*C[0]*C[3]*self.R[1] + self.P[2]*C[1]*C[2]*self.R[2]
        #                 - self.P[2]*C[2]*C[3]*self.R[0] - self.P[3]*C[1]**2*self.R[3] + self.P[3]*C[1]*C[3]*self.R[1]
        #                 + self.P[3]*C[1]*C[3]*self.R[2] - self.P[3]*C[3]**2*self.R[0] - self.R[0]*self.R[3]
        #                 + self.R[1]*self.R[2]),
        #              (self.P[2]*C[0] + self.P[3]*C[1])
        #               *(-self.P[0]*C[2]**2 - self.P[1]*C[2]*C[3] - self.P[2]*C[2]*C[3] - self.P[3]*C[3]**2 - self.R[3])
        #               /(-self.P[0]*self.P[3]*C[0]**2*C[3]**2 + 2*self.P[0]*self.P[3]*C[0]*C[1]*C[2]*C[3]
        #                 - self.P[0]*self.P[3]*C[1]**2*C[2]**2 - self.P[0]*C[0]**2*self.R[3]
        #                 + self.P[0]*C[0]*C[2]*self.R[1] + self.P[0]*C[0]*C[2]*self.R[2] - self.P[0]*C[2]**2*self.R[0]
        #                 + self.P[1]*self.P[2]*C[0]**2*C[3]**2 - 2*self.P[1]*self.P[2]*C[0]*C[1]*C[2]*C[3]
        #                 + self.P[1]*self.P[2]*C[1]**2*C[2]**2 - self.P[1]*C[0]*C[1]*self.R[3]
        #                 + self.P[1]*C[0]*C[3]*self.R[2] + self.P[1]*C[1]*C[2]*self.R[1] - self.P[1]*C[2]*C[3]*self.R[0]
        #                 - self.P[2]*C[0]*C[1]*self.R[3] + self.P[2]*C[0]*C[3]*self.R[1] + self.P[2]*C[1]*C[2]*self.R[2]
        #                 - self.P[2]*C[2]*C[3]*self.R[0] - self.P[3]*C[1]**2*self.R[3] + self.P[3]*C[1]*C[3]*self.R[1]
        #                 + self.P[3]*C[1]*C[3]*self.R[2] - self.P[3]*C[3]**2*self.R[0] - self.R[0]*self.R[3]
        #                 + self.R[1]*self.R[2]) + (self.P[2]*C[2] + self.P[3]*C[3])
        #               *(self.P[0]*C[0]*C[2] + self.P[1]*C[1]*C[2] + self.P[2]*C[0]*C[3] + self.P[3]*C[1]*C[3]
        #                 + self.R[2])/(-self.P[0]*self.P[3]*C[0]**2*C[3]**2 + 2*self.P[0]*self.P[3]*C[0]*C[1]*C[2]*C[3]
        #                               - self.P[0]*self.P[3]*C[1]**2*C[2]**2 - self.P[0]*C[0]**2*self.R[3]
        #                               + self.P[0]*C[0]*C[2]*self.R[1] + self.P[0]*C[0]*C[2]*self.R[2]
        #                               - self.P[0]*C[2]**2*self.R[0] + self.P[1]*self.P[2]*C[0]**2*C[3]**2
        #                               - 2*self.P[1]*self.P[2]*C[0]*C[1]*C[2]*C[3] + self.P[1]*self.P[2]*C[1]**2*C[2]**2
        #                               - self.P[1]*C[0]*C[1]*self.R[3] + self.P[1]*C[0]*C[3]*self.R[2]
        #                               + self.P[1]*C[1]*C[2]*self.R[1] - self.P[1]*C[2]*C[3]*self.R[0]
        #                               - self.P[2]*C[0]*C[1]*self.R[3] + self.P[2]*C[0]*C[3]*self.R[1]
        #                               + self.P[2]*C[1]*C[2]*self.R[2] - self.P[2]*C[2]*C[3]*self.R[0]
        #                               - self.P[3]*C[1]**2*self.R[3] + self.P[3]*C[1]*C[3]*self.R[1]
        #                               + self.P[3]*C[1]*C[3]*self.R[2] - self.P[3]*C[3]**2*self.R[0]
        #                               - self.R[0]*self.R[3] + self.R[1]*self.R[2]),
        #               (self.P[2]*C[0] + self.P[3]*C[1])*(self.P[0]*C[0]*C[2] + self.P[1]*C[0]*C[3] + self.P[2]*C[1]*C[2]
        #                                                  + self.P[3]*C[1]*C[3]
        #                 + self.R[1])/(-self.P[0]*self.P[3]*C[0]**2*C[3]**2 + 2*self.P[0]*self.P[3]*C[0]*C[1]*C[2]*C[3]
        #                               - self.P[0]*self.P[3]*C[1]**2*C[2]**2 - self.P[0]*C[0]**2*self.R[3]
        #                               + self.P[0]*C[0]*C[2]*self.R[1] + self.P[0]*C[0]*C[2]*self.R[2]
        #                               - self.P[0]*C[2]**2*self.R[0] + self.P[1]*self.P[2]*C[0]**2*C[3]**2
        #                               - 2*self.P[1]*self.P[2]*C[0]*C[1]*C[2]*C[3] + self.P[1]*self.P[2]*C[1]**2*C[2]**2
        #                               - self.P[1]*C[0]*C[1]*self.R[3] + self.P[1]*C[0]*C[3]*self.R[2]
        #                               + self.P[1]*C[1]*C[2]*self.R[1] - self.P[1]*C[2]*C[3]*self.R[0]
        #                               - self.P[2]*C[0]*C[1]*self.R[3] + self.P[2]*C[0]*C[3]*self.R[1]
        #                               + self.P[2]*C[1]*C[2]*self.R[2] - self.P[2]*C[2]*C[3]*self.R[0]
        #                               - self.P[3]*C[1]**2*self.R[3] + self.P[3]*C[1]*C[3]*self.R[1]
        #                               + self.P[3]*C[1]*C[3]*self.R[2] - self.P[3]*C[3]**2*self.R[0]
        #                               - self.R[0]*self.R[3] + self.R[1]*self.R[2])
        #               + (self.P[2]*C[2] + self.P[3]*C[3])*(-self.P[0]*C[0]**2 - self.P[1]*C[0]*C[1]
        #                                                    - self.P[2]*C[0]*C[1] - self.P[3]*C[1]**2 - self.R[0])
        #               /(-self.P[0]*self.P[3]*C[0]**2*C[3]**2 + 2*self.P[0]*self.P[3]*C[0]*C[1]*C[2]*C[3]
        #                 - self.P[0]*self.P[3]*C[1]**2*C[2]**2 - self.P[0]*C[0]**2*self.R[3]
        #                 + self.P[0]*C[0]*C[2]*self.R[1] + self.P[0]*C[0]*C[2]*self.R[2] - self.P[0]*C[2]**2*self.R[0]
        #                 + self.P[1]*self.P[2]*C[0]**2*C[3]**2 - 2*self.P[1]*self.P[2]*C[0]*C[1]*C[2]*C[3]
        #                 + self.P[1]*self.P[2]*C[1]**2*C[2]**2 - self.P[1]*C[0]*C[1]*self.R[3]
        #                 + self.P[1]*C[0]*C[3]*self.R[2] + self.P[1]*C[1]*C[2]*self.R[1] - self.P[1]*C[2]*C[3]*self.R[0]
        #                 - self.P[2]*C[0]*C[1]*self.R[3] + self.P[2]*C[0]*C[3]*self.R[1] + self.P[2]*C[1]*C[2]*self.R[2]
        #                 - self.P[2]*C[2]*C[3]*self.R[0] - self.P[3]*C[1]**2*self.R[3] + self.P[3]*C[1]*C[3]*self.R[1]
        #                 + self.P[3]*C[1]*C[3]*self.R[2] - self.P[3]*C[3]**2*self.R[0] - self.R[0]*self.R[3]
        #                 + self.R[1]*self.R[2])])

        # Update covariance matrix

        # Update state estimate
        theta = 0
        phi = 0

        return theta, phi

    def common_trig_terms(self) -> Tuple[float, float, float, float, float]:
        # Compute common trigonometric terms
        sin_phi, cos_phi = np.sin(self.phi), np.cos(self.phi)
        sin_theta, cos_theta, tan_theta = np.sin(self.theta), np.cos(self.theta), np.tan(self.theta)

        # Check for division by zero error
        if cos_theta == 0:
            cos_theta = 1e-6

        return sin_phi, cos_phi, sin_theta, cos_theta, tan_theta

    def euler_integration(self, prev_state, derivative) -> float:
        return prev_state + derivative * self.t_sample


class RCLowPassFilter:

    def __init__(self, f_cutoff: float, t_sample: float):
        # Compute lpf constants
        RC = 1 / (2 * np.pi * f_cutoff)
        self.c1 = t_sample / (t_sample + RC)
        self.c2 = RC / (t_sample + RC)

        self.prev_output: float = 0

    def compute_lfp(self, lpf_input) -> float:
        output = self.c1 * lpf_input + self.c2 * self.prev_output
        self.prev_output = output

        return output
