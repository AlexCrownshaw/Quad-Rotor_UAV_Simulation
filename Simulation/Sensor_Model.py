import numpy as np

from Data_Handling.Data_Classes import TimeState, StateDerivative, SensorState
from Data_Handling.Data_Processing import SensorData


class Sensors:

    g = 9.81

    def __init__(self, acc_config: dict, gyro_config: dict, mag_config: dict):

        self.acc = IMUSensor(acc_config)
        self.gyro = IMUSensor(gyro_config)
        # self.mag = IMUSensor(mag_config)

        # Define previous state variables
        self.acc_vector_prev = np.zeros(3)
        self.gyro_vector_prev = np.zeros(3)

        # Time values
        self.t_prev = 0
        self.t_acc = 1 / acc_config["bandwidth"]
        self.t_gyro = 1 / gyro_config["bandwidth"]

        self.data = SensorData()

    def simulate_sensors(self, t, X: TimeState, X_dot: StateDerivative) -> object:
        # Check if sensor is ready for update
        if (t - self.t_prev) <= self.t_acc:
            # Add acceleration due to gravity vector
            acc_body_vector = X_dot.acc_body_vector + self.gravity_vector_body(X.theta, X.phi)
            # Simulate sensor noise
            acc_vector = self.acc.gaussian_noise(acc_body_vector)
        else:
            acc_vector = self.acc_vector_prev

        # Check if sensor is ready for update
        if (t - self.t_prev) >= self.t_gyro:
            # Simulate sensor noise
            gyro_vector = self.gyro.gaussian_noise(X.body_rate_vector)
        else:
            gyro_vector = self.gyro_vector_prev

        # Assume the inertial frame [XYZ] is aligned to the ECEF frame [NED]
        # mag_vector = self.mag.gaussian_noise(t, X)

        self.data.append_data(t, X_dot.acc_body_vector, acc_vector, X.body_rate_vector, gyro_vector)

        self.t_prev = t
        self.acc_vector_prev = acc_vector
        self.gyro_vector_prev = gyro_vector

        return SensorState(acc_vector, gyro_vector)

    def return_sensor_data(self) -> SensorData:
        return self.data

    def gravity_vector_body(self, theta, phi) -> np.array:
        return np.array([-self.g * np.sin(theta), self.g * np.sin(phi) * np.cos(theta),
                         self.g * np.cos(phi) * np.cos(theta)])


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

    def gaussian_noise(self, vector: np.array) -> np.array:
        self.vector_gn = vector + np.random.normal(0, self.std, 3)

        return self.vector_gn


class GPS:

    def __init__(self):
        pass
