from typing import Tuple

import pandas as pd


class PID:

    def __init__(self, dt, kp, ki, kd, output_limit) -> None:
        self.dt = dt

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.output_limit = output_limit

        self.pid_data = pd.DataFrame(columns=["time", "setpoint", "state_input", "output", "error", "error_sum",
                                              "d_error"])

        self.t: float = 0 - self.dt

        self.error_last: float = 0
        self.error_sum: float = 0

    def compute_pid(self, state_input: float, setpoint: float) -> float:

        # Keep track of simulation time
        self.t = self.t + self.dt

        error = setpoint - state_input
        self.error_sum = self.error_sum + error

        if self.error_sum > self.output_limit:   # error_sum anti-windup
            self.error_sum = self.output_limit

        d_error = (error - self.error_last) / self.dt
        self.error_last = error

        output = self.kp * error + self.ki * self.error_sum + self.kd * d_error

        if output > self.output_limit:  # Output anti-windup
            output = self.output_limit

        self.pid_data.loc[len(self.pid_data)] = [self.t, setpoint, state_input, output, error, self.error_sum,
                                                 d_error]

        return output

    def return_data(self) -> pd.DataFrame:
        return self.pid_data

    def clear_pid_data(self) -> None:
        self.pid_data = self.pid_data[0:0]
