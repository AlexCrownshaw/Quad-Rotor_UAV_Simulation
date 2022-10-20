import numpy as np

from Data_Handling.Data_Classes import SensorState, EstimateState
from Data_Handling.Data_Processing import StateEstimationData


class StateEstimation:

    def __init__(self, frequency: float, acc_lpf_cutoff_freq: int, gyro_lpf_cutoff_freq: int):
        # Create low pass filter objects
        t_sample = 1 / frequency

        self.acc_lpf_x = RCLowPassFilter(t_sample, acc_lpf_cutoff_freq)
        self.acc_lpf_y = RCLowPassFilter(t_sample, acc_lpf_cutoff_freq)
        self.acc_lpf_z = RCLowPassFilter(t_sample, acc_lpf_cutoff_freq)

        self.gyro_lpf_p = RCLowPassFilter(t_sample, gyro_lpf_cutoff_freq)
        self.gyro_lpf_q = RCLowPassFilter(t_sample, gyro_lpf_cutoff_freq)
        self.gyro_lpf_r = RCLowPassFilter(t_sample, gyro_lpf_cutoff_freq)

        self.ahrs = AHRS()

        self.data = StateEstimationData()

    def compute_state_estimate(self, t: float, S: SensorState) -> EstimateState:
        # Filter sensor outputs
        acc_filt = np.transpose(np.array([self.acc_lpf_x.compute_lfp(S.acc[0]),
                                          self.acc_lpf_y.compute_lfp(S.acc[1]),
                                          self.acc_lpf_z.compute_lfp(S.acc[2])]))
        gyro_filt = np.transpose(np.array([self.gyro_lpf_p.compute_lfp(S.gyro[0]),
                                           self.gyro_lpf_q.compute_lfp(S.gyro[1]),
                                           self.gyro_lpf_r.compute_lfp(S.gyro[2])]))

        # Compute AHRS
        attitude_vector = self.ahrs.compute_ahrs(acc_filt, gyro_filt)

        # Append data
        self.data.append_data(t, attitude_vector, acc_filt, gyro_filt)

        return EstimateState(attitude_vector)

    def return_data(self) -> StateEstimationData:
        return self.data


class AHRS:

    def __init__(self):
        self.yaw_prev, self.pitch_prev, self.roll_prev = 0, 0, 0

    def compute_ahrs(self, acc: np.array, gyro: np.array) -> np.array:
        gyro_inertial = self.get_rotation_matrix(self.yaw_prev, self.pitch_prev, self.roll_prev,
                                                 transpose=True).dot(gyro)

        yaw, pitch, roll = 0, 0, 0
        ahrs_state = np.array([yaw, pitch, roll])

        return ahrs_state

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
