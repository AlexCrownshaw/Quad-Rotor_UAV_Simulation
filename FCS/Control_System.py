import numpy as np

from FCS.PID import PID


class ControlSystem:

    motor_limit = 1000

    def __init__(self, gain_x, gain_y, gain_z, gain_yaw, gain_pitch, gain_roll) -> None:

        # Translation PID objects
        self.pid_x = PID(gain_x[0], gain_x[1], gain_x[2], self.motor_limit)
        self.pid_y = PID(gain_y[0], gain_y[1], gain_y[2], self.motor_limit)
        self.pid_z = PID(gain_z[0], gain_z[1], gain_z[2], self.motor_limit)

        # Rotational PID objects
        self.pid_roll = PID(gain_yaw[0], gain_yaw[1], gain_yaw[2], self.motor_limit)
        self.pid_pitch = PID(gain_pitch[0], gain_pitch[1], gain_pitch[2], self.motor_limit)
        self.pid_yaw = PID(gain_roll[0], gain_roll[1], gain_roll[2], self.motor_limit)

    @staticmethod
    def run_control_loop() -> np.array:
        return np.array()
