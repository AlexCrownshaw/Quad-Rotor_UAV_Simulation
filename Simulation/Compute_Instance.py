import numpy as np

from Simulation.QRUAV_Sim import QRUAVSim


class TimeInstance(QRUAVSim):

    def __init__(self, prev_time_instance: object, motor_thrusts: np.array):
        self.prev_time_instance = prev_time_instance
        self.motor_thrusts = motor_thrusts
        self.thrust_vector = np.array([0, 0, -np.sum(motor_thrusts)]).reshape(3, 1)
        self.moments = self.calculate_moments(self.motor_thrusts)
