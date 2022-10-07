import numpy as np

from Simulation.Time_State import TimeState
from scipy.optimize import fsolve
from Data_Processing.Data_Processing import ThrustData


class ThrustModel:
    rho = 1.225  # Kg/m^3

    def __init__(self, dt,  motor_properties: dict, propeller_properties: dict, dimensions: dict):
        self.dt = dt
        self.t: float = 0

        self.motor_properties = motor_properties
        self.prop = propeller_properties
        self.dimensions = dimensions

        self.prop["A"] = self.prop["R"] ** 2 * np.pi  # Propeller disc area

        # Pitch angles
        pitch = self.prop["pitch"]
        d = self.prop["d"]
        self.theta_0 = 2 * np.arctan2(pitch, (2 * np.pi * 3 / 4 * d / 2))
        self.theta_1 = -4 / 3 * np.arctan2(pitch, 2 * np.pi * 3 / 4 * d / 2)

        # Create thrust data collection object
        self.data = ThrustData()

    def thrust_equation(self, v_i, *thrust_eqn_args) -> float:
        omega, V = thrust_eqn_args

        # Calculate propeller flow velocity using induced velocity v_i
        V_prime = np.sqrt(V[0] ** 2 + V[1] ** 2 + (V[2] - v_i) ** 2)

        # Calculate average thrust over a single revolution
        Thrust = 1 / 4 * self.rho * self.prop["a"] * self.prop["n_blades"] * self.prop["c"] * self.prop["R"] * \
                 ((V[2] - v_i) * omega * self.prop["R"] + 2 / 3 * (omega * self.prop["R"]) ** 2 *
                  (self.theta_0 + 3 / 4 * self.theta_1) + (V[0] ** 2 + V[1] ** 2) *
                  (self.theta_0 + 1 / 2 * self.theta_1))

        # Calculate residual for equation: Thrust = mass flow rate * delta Velocity
        return self.prop["eta"] * 2 * v_i * self.rho * self.prop["A"] * V_prime - Thrust

    def solve_thrust(self, X: TimeState, U: np.array) -> np.array:
        # Transform velocity relative to propeller position
        V = np.array([X.u - X.r * self.dimensions["d_y"],
                      X.v + X.r * self.dimensions["d_x"],
                      X.w - X.q * self.dimensions["d_x"] + X.p * self.dimensions["d_y"]])

        T = np.zeros(4)

        # loop through motors, omega = angular rate of each motor
        for motor_index in range(len(U)):
            # Calculate new time
            self.t = self.t + self.dt

            # Numerically solve for propeller induced velocity
            v_i0 = 100
            thrust_eqn_args = 2 * np.pi / 60 * U[motor_index], V
            v_i = fsolve(self.thrust_equation, v_i0, args=thrust_eqn_args)

            # Re-calculate v_prime with solved propeller induced velocity
            V_prime = np.sqrt(V[0] ** 2 + V[1] ** 2 + (V[2] - v_i) ** 2)
            T[motor_index] = self.prop["eta"] * 2 * v_i * self.rho * self.prop["A"] * V_prime

            # Append motor thrust data
            self.data.append_thrust_data(motor_index, self.t, V, v_i, V_prime, T[motor_index])

        return T

    def return_thrust_data(self) -> ThrustData:
        return self.data
