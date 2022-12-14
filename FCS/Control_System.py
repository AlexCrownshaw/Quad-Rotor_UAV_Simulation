import numpy as np

from FCS.PID import PID
from Data_Handling.Data_Classes import TimeState, EstimateState


class ControlSystem:

    ROTATIONAL_PID_TUNE = False
    USE_DYNAMICS = True

    max_rpm = 6000
    tau = 0  # PID derivative low pass filter factor
    current_inputs = np.zeros(4)

    def __init__(self, dt, maneuvers, gain_x, gain_y, gain_z, gain_yaw, gain_pitch, gain_roll) -> None:

        self.maneuvers = maneuvers
        self.upcoming_maneuvers = maneuvers

        # Translation PID objects
        self.pid_x = PID(dt, gain_x[0], gain_x[1], gain_x[2], self.tau, self.max_rpm, lim_min=0)
        self.pid_y = PID(dt, gain_y[0], gain_y[1], gain_y[2], self.tau, self.max_rpm, lim_min=0)
        self.pid_z = PID(dt, gain_z[0], gain_z[1], gain_z[2], self.tau, self.max_rpm, lim_min=0)

        # Rotational PID objects
        self.pid_yaw = PID(dt, gain_yaw[0], gain_yaw[1], gain_yaw[2], self.tau, self.max_rpm, lim_min=0)
        self.pid_pitch = PID(dt, gain_pitch[0], gain_pitch[1], gain_pitch[2], self.tau, self.max_rpm, lim_min=0)
        self.pid_roll = PID(dt, gain_roll[0], gain_roll[1], gain_roll[2], self.tau, self.max_rpm, lim_min=0)

    @staticmethod
    def motor_mixing(output_vector: np.array) -> np.array:
        return np.array([output_vector[0] + output_vector[1] - output_vector[2] + output_vector[3],
                         output_vector[0] + output_vector[1] + output_vector[2] - output_vector[3],
                         output_vector[0] - output_vector[1] - output_vector[2] - output_vector[3],
                         output_vector[0] - output_vector[1] + output_vector[2] + output_vector[3]])

    def run_control_loop(self, X: TimeState, G: EstimateState, t: float) -> np.array:
        # Collect current maneuver from flight path
        control_inputs = self.control_inputs(t)

        # Check for use ideal dynamics output as PID input
        if self.USE_DYNAMICS:
            theta = X.theta
            phi = X.phi
        else:
            theta = G.theta
            phi = G.phi

        # Check for rotational PID tune flag and run alternate control loop
        if self.ROTATIONAL_PID_TUNE:
            output_z = self.pid_z.compute_pid(X.z, control_inputs[0])

            output_yaw = self.pid_yaw.compute_pid(X.psi, control_inputs[1])
            output_pitch = self.pid_pitch.compute_pid(theta, control_inputs[2])
            output_roll = self.pid_roll.compute_pid(phi, control_inputs[3])

        else:
            output_x = self.pid_x.compute_pid(X.x, control_inputs[0])
            output_pitch = self.pid_pitch.compute_pid(theta, output_x)

            output_y = self.pid_y.compute_pid(X.y, control_inputs[1])
            output_roll = self.pid_roll.compute_pid(phi, output_y)

            output_z = self.pid_z.compute_pid(X.z, control_inputs[2])

            output_yaw = self.pid_yaw.compute_pid(X.psi, control_inputs[3])

        U = self.motor_mixing(np.array([output_z, output_yaw, output_pitch, output_roll]))

        # Fix motor speed to defined RPM bounds
        for U_index in range(len(U)):
            if U[U_index] > self.max_rpm:
                U[U_index] = self.max_rpm
            elif U[U_index] < 0:
                U[U_index] = 0

        return U

    def control_inputs(self, t) -> np.array:
        if len(self.upcoming_maneuvers) == 0:
            return self.current_inputs
        for maneuver in self.upcoming_maneuvers:
            if float(maneuver["time"]) <= t:
                self.upcoming_maneuvers = self.upcoming_maneuvers[1:]
                if self.ROTATIONAL_PID_TUNE:
                    self.current_inputs = np.array([maneuver["z"], np.radians(maneuver["yaw"]),
                                                    np.radians(maneuver["pitch"]), np.radians(maneuver["roll"])])
                else:
                    self.current_inputs = np.array([maneuver["x"], maneuver["y"], maneuver["z"],
                                                    np.radians(maneuver["yaw"])])
            elif t < self.maneuvers[0]["time"]:
                return np.array([0, 0, 0, 0])

            return self.current_inputs

    def return_pid_data(self) -> list:
        pid_data_list = []
        for pid in [self.pid_x, self.pid_y, self.pid_z, self.pid_yaw, self.pid_pitch, self.pid_roll]:
            pid_data_list.append(pid.return_data())

        return pid_data_list
