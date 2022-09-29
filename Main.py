from FCS.PID import PID
from Simulation.QRUAV_Sim import QRUAVSim

import numpy as np

# Initial Conditions
INITIAL_POSITION = [0, 0, 0]

# Anti-wind up value (Not necessary for simulation)
OUTPUT_LIMIT_TRANSLATION = 1000
OUTPUT_LIMIT_ROTATION = 1000

# PID Gain Values
KP_X, KI_X, KD_X = 1, 1, 1
KP_Y, KI_Y, KD_Y = 1, 1, 1
KP_Z, KI_Z, KD_Z = 1, 1, 1

KP_ROLL, KI_ROLL, KD_ROLL = 1, 1, 1
KP_PITCH, KI_PITCH, KD_PITCH = 1, 1, 1
KP_YAW, KI_YAW, KD_YAW = 1, 1, 1

STRUCTURAL_JSON_PATH = r"Config_JSON/Structural_Properties/Structure.json"
FLIGHT_PATH_JSON_PATH = r"Config_JSON/Flight_Plan/Flight_Path.json"


def main():

    # Translation PID objects
    pid_x = PID(KP_X, KI_X, KD_X, OUTPUT_LIMIT_TRANSLATION)
    pid_y = PID(KP_Y, KI_Y, KD_Y, OUTPUT_LIMIT_TRANSLATION)
    pid_z = PID(KP_Z, KI_Z, KD_Z, OUTPUT_LIMIT_TRANSLATION)

    # Rotational PID objects
    pid_roll = PID(KP_ROLL, KI_ROLL, KD_ROLL, OUTPUT_LIMIT_ROTATION)
    pid_pitch = PID(KP_PITCH, KI_PITCH, KD_PITCH, OUTPUT_LIMIT_ROTATION)
    pid_yaw = PID(KP_YAW, KI_YAW, KD_YAW, OUTPUT_LIMIT_ROTATION)

    # Instantiate Quad-rotor Simulation Object
    sim = QRUAVSim(STRUCTURAL_JSON_PATH, FLIGHT_PATH_JSON_PATH)
    print(sim)

    moments = sim.calculate_moments(motor_thrusts=np.array([20, 10, 20, 10]))
    print(moments)


if __name__ == "__main__":
    main()
