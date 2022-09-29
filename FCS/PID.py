from typing import Tuple

import pandas as pd


class PID:
    pid_data = pd.DataFrame(columns=["setpoint", "input", "output", "error", "error_sum", "d_input"])

    def __init__(self, kp, ki, kd, output_limit) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

    def compute_pid(self, setpoint, _input: float, input_last, error_sum) -> Tuple[float, float]:

        error = setpoint - _input
        error_sum: float = error_sum + error

        if error_sum > self.output_limit:   # error_sum anti-windup
            error_sum = self.output_limit

        d_input = _input - input_last

        output = self.kp * error + self.ki * error_sum - self.kd * d_input

        if output > self.output_limit:  # Output anti-windup
            output = self.output_limit

        self.pid_data.loc[len(self.pid_data)] = [setpoint, _input, output, error, error_sum, d_input]

        return output, error_sum

    def compute_pid_real_time(self):
        pass

    def clear_pid_data(self) -> None:
        self.pid_data = self.pid_data[0:0]
