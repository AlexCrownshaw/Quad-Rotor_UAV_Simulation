import numpy as np

from Simulation.Time_State import TimeState, StateDerivative
from Data_Processing.Data_Processing import SensorData


class Sensors:

    def __init__(self, acc_config: dict, gyro_config: dict, mag_config: dict):

        self.acc = IMUSensor(acc_config)
        self.gyro = IMUSensor(gyro_config)
        self.mag = IMUSensor(mag_config)

        self.data = SensorData()

    def simulate_sensors(self, t, X: TimeState, X_dot: StateDerivative) -> object:
        acc_vector = self.acc.gaussian_noise(t, X_dot.acc_body_vector)
        gyro_vector = self.gyro.gaussian_noise(t, X.body_rate_vector)

        # Assume the inertial frame [XYZ] is aligned to the ECEF frame [NED]
        # mag_vector = self.mag.gaussian_noise(t, X)

        self.data.append_data(t, X_dot.acc_body_vector, acc_vector, X.body_rate_vector, gyro_vector)

        return SensorState(acc_vector, gyro_vector)

    def return_sensor_data(self) -> SensorData:
        return self.data


class IMUSensor:

    def __init__(self, config: dict):
        self.bandwidth = config["bandwidth"]  # Hz
        self.spec_density = config["spectral_density"] * 9.81  # m/s^2/Hz

        # Calculate sample time
        self.t_sample = 1/self.bandwidth  # s

        # Calculate std deviation for additive white gaussian noise
        self.std = np.sqrt(self.spec_density * self.bandwidth)

        self.t_prev = 0
        self.vector_gn = np.array(3*[0])

    def gaussian_noise(self, t: float, vector: np.array) -> np.array:
        # Check if elapsed time is greater than instrument sample time
        if t - self.t_prev > self.t_sample:
            self.vector_gn = vector + np.random.normal(0, self.std, 3)

        return self.vector_gn


class GPS:

    def __init__(self):
        pass


class SensorState:

    def __init__(self, acc, gyro):
        self.acc = acc
        self.gyro = gyro
