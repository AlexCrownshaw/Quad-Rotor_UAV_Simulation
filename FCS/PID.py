from typing import Tuple

import pandas as pd


class PID:

    pid_data = pd.DataFrame(columns=["control_input", "state_input", "output", "error", "error_sum", "d_input"])
    
    input_last: float = 0
    error_sum: float = 0

    def __init__(self, kp, ki, kd, output_limit) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

    def compute_pid(self, control_input: float, state_input: float) -> float:

        error = control_input - state_input
        self.error_sum = self.error_sum + error

        if self.error_sum > self.output_limit:   # error_sum anti-windup
            self.error_sum = self.output_limit

        d_input = state_input - self.input_last
        self.input_last = state_input

        output = self.kp * error + self.ki * self.error_sum - self.kd * d_input

        if output > self.output_limit:  # Output anti-windup
            output = self.output_limit

        self.pid_data.loc[len(self.pid_data)] = [control_input, state_input, output, error, self.error_sum, d_input]

        return output

    def compute_pid_real_time(self):
        pass

    def clear_pid_data(self) -> None:
        self.pid_data = self.pid_data[0:0]
