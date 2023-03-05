import numpy as np

from FCS.PID import PID
from Data_Handling.Data_Classes import TimeState, EstimateState


class ControlSystem:

    ROTATIONAL_PID_TUNE = False
    USE_DYNAMICS = True
    ROTATE_OUTPUT_ARRAY = False
    ROTATE_SET_POINTS = False

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
        U = np.array([output_vector[0] + output_vector[1] - output_vector[2] + output_vector[3],
                      output_vector[0] + output_vector[1] + output_vector[2] - output_vector[3],
                      output_vector[0] - output_vector[1] - output_vector[2] - output_vector[3],
                      output_vector[0] - output_vector[1] + output_vector[2] + output_vector[3]])

        return U

    def run_control_loop(self, X: TimeState, G: EstimateState, t: float) -> np.array:

        # Collect current maneuver from flight path
        control_inputs = self.control_inputs(t)

        if self.ROTATE_SET_POINTS:
            control_input_x = control_inputs[0]
            control_input_y = control_inputs[1]
            control_inputs[0] = control_input_x * np.cos(X.psi) - control_input_y * np.sin(X.psi)
            control_inputs[1] = control_input_x * np.sin(X.psi) + control_input_y * np.cos(X.psi)

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

        output_array = np.array([output_z, output_yaw, output_pitch, output_roll])

        if self.ROTATE_OUTPUT_ARRAY:
            output_array[1:3] = np.array(output_array[1:3]).dot(np.array([np.cos(X.psi), -np.sin(X.psi), np.sin(X.psi),
                                                                          np.cos(X.psi)]).reshape(2, 2))

            output_x = output_array[1]
            output_y = output_array[2]
            output_array[1] = output_x * np.cos(X.psi) - output_y * np.sin(X.psi)
            output_array[2] = output_x * np.sin(X.psi) + output_y * np.cos(X.psi)

        U = self.motor_mixing(output_array)

        # Fix motor speed to defined RPM bounds
        for U_index in range(len(U)):
            if U[U_index] > self.max_rpm:
                U[U_index] = self.max_rpm
            elif U[U_index] < 0:
                U[U_index] = 0

        return U

    def control_inputs(self, t):
        try:
            if self.upcoming_maneuvers[1]["time"] <= t:
                self.upcoming_maneuvers = self.upcoming_maneuvers[1:]
        except IndexError:
            pass

        if self.ROTATIONAL_PID_TUNE:
            current_inputs = np.array([self.upcoming_maneuvers[0]["z"], np.radians(self.upcoming_maneuvers[0]["yaw"]),
                                       np.radians(self.upcoming_maneuvers[0]["pitch"]),
                                       np.radians(self.upcoming_maneuvers[0]["roll"])])
        else:
            current_inputs = np.array([self.upcoming_maneuvers[0]["x"], self.upcoming_maneuvers[0]["y"],
                                       self.upcoming_maneuvers[0]["z"],
                                       np.radians(self.upcoming_maneuvers[0]["yaw"])])

        if t < self.maneuvers[0]["time"]:
            current_inputs = np.array([0, 0, 0, 0])

        return current_inputs

    def return_pid_data(self) -> list:
        pid_data_list = []
        for pid in [self.pid_x, self.pid_y, self.pid_z, self.pid_yaw, self.pid_pitch, self.pid_roll]:
            pid_data_list.append(pid.return_data())

        return pid_data_list
