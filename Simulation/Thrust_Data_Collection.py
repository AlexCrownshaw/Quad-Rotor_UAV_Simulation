import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

from Simulation.Thrust_Torque_Model import ThrustModel
from Main import load_vehicle_properties
from Data_Handling.Data_Classes import TimeState

VEHICLE_PROPERTIES_JSON_PATH = r"D:\Documents-HDD\University\3rd Year\Final Year " \
                               r"Project\Quad-Rotor_UAV_Simulation\Config_JSON\Vehicle_Properties.json "

W_MIN = -2
W_MAX = 10
W_STEP = 0.5

X_MIN = -2
X_MAX = 10
X_STEP = 0.5


def simulate_thrust(X: TimeState, thrust: ThrustModel, plot_label):
    T_list = []
    rpm_list = np.arange(1000, 6000 + 200, 200)
    for rpm in rpm_list:
        U = np.array(4 * [rpm])
        T_list.append(np.sum(thrust.solve_thrust(X, U)))

    plt.plot(rpm_list, T_list, label=plot_label)


def main():
    properties, dimensions, motor, propeller = load_vehicle_properties(VEHICLE_PROPERTIES_JSON_PATH)
    thrust = ThrustModel(0.01, motor, propeller, dimensions)

    for w in np.arange(W_MIN, W_MAX + W_STEP, W_STEP):
        state_vector = np.array([0, 0, w, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        X = TimeState(state_vector)
        simulate_thrust(X, thrust, "w = {} m/s".format(str(w)))

    plt.xlabel("RPM")
    plt.ylabel("Thrust [N]")
    plt.legend(title="Varying vertical velocity")
    plt.show()

    plt.figure()
    for x in np.arange(X_MIN, X_MAX + W_STEP, X_STEP):
        state_vector = np.array([0, 0, x, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        X = TimeState(state_vector)
        simulate_thrust(X, thrust, "x = {} m/s".format(str(x)))

    plt.xlabel("RPM")
    plt.ylabel("Thrust [N]")
    plt.legend(title="Varying vertical velocity")
    plt.show()


if __name__ == "__main__":
    main()
