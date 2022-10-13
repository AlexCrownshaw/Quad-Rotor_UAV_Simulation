import pandas as pd


class PID:

    def __init__(self, dt, kp, ki, kd, tau, lim_max, lim_min) -> None:
        self.dt = dt

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.tau = tau

        self.lim_max = lim_max
        self.lim_min = lim_min

        self.t: float = 0 - self.dt

        self.integral: float = 0
        self.derivative: float = 0
        self.error_prev: float = 0
        self.state_input_prev: float = 0

        self.pid_data = pd.DataFrame(columns=["time", "setpoint", "state_input", "output", "error", "integral",
                                              "derivative"])

    def compute_pid(self, state_input: float, setpoint: float) -> float:

        # Keep track of simulation time
        self.t = self.t + self.dt

        # Proportional
        error = setpoint - state_input
        proportional = self.kp * error

        # Integral
        self.integral = self.integral + 0.5 * self.ki * self.dt * (error + self.error_prev)

        # Anti-wind-up via dynamic integral clamping
        # Define integral limits
        if self.lim_max > proportional:
            lim_max_integral: float = self.lim_max - proportional
        else:
            lim_max_integral: float = 0

        if self.lim_min < proportional:
            lim_min_integral: float = self.lim_min - proportional
        else:
            lim_min_integral: float = 0

        # Clamp integral
        if self.integral > lim_max_integral:
            self.integral = lim_max_integral
        elif self.integral < lim_min_integral:
            self.integral = lim_min_integral

        # Derivative
        # Take derivative of state_input instead of error term
        self.derivative = (2 * self.kd * (state_input - self.state_input_prev)
                           + (2 * self.tau - self.dt) * self.derivative) / (2 * self.tau + self.dt)

        # Compute output
        output = proportional + self.integral + self.derivative

        # Store values for next iteration
        self.state_input_prev = state_input
        self.error_prev = error

        # Log data to df
        self.pid_data.loc[len(self.pid_data)] = [self.t, setpoint, state_input, output, error, self.integral,
                                                 self.derivative]

        return output

    def return_data(self) -> pd.DataFrame:
        return self.pid_data

    def clear_pid_data(self) -> None:
        self.pid_data = self.pid_data[0:0]
