import json
from typing import Tuple

import numpy as np


class QRUAVSim:

    def __init__(self, structural_json_path, flight_path_json_path):

        # Load simulation config properties
        self.properties, self.dimensions = self.load_structural_json(structural_json_path)
        self.boundary_conditions, self.maneuvers = self.load_flight_path_json(flight_path_json_path)

        # Define simulation constants
        self.moment_inertia_matrix = np.array([self.properties["I_x"], 0, 0, 0, self.properties["I_y"], 0, 0, 0,
                                               self.properties["I_z"]]).reshape(3, 3)
        self.gravity_vector_inertial = np.array([0, 0, self.properties["mass"] * 9.81]).reshape(3, 1)

        print(self)

    @staticmethod
    def load_structural_json(json_path: str) -> Tuple[dict, dict]:
        with open(json_path) as json_file:
            json_object = json.load(json_file)
            return json_object["properties"], json_object["dimensions"]

    @staticmethod
    def load_flight_path_json(json_path: str) -> Tuple[dict, dict]:
        with open(json_path) as json_file:
            json_object = json.load(json_file)
            return json_object["Boundary_Conditions"], json_object["Maneuvers"]

    def compute_moments(self, motor_thrusts: np.array) -> np.array:
        L = np.sum(motor_thrusts * np.array([self.dimensions["d_y"], -self.dimensions["d_y"], -self.dimensions["d_y"],
                                             self.dimensions["d_y"]]))
        M = np.sum(motor_thrusts * np.array([self.dimensions["d_x"], -self.dimensions["d_x"], self.dimensions["d_x"],
                                             -self.dimensions["d_x"]]))
        N = np.sum(np.array([0, 0, 0]))

        return np.array([L, M, N])

    def compute_state_derivative(self) -> Tuple[np.array, np.array, np.array, np.array]:
        linear_body = np.array()
        rotational_body = np.array()
        linear_inertial = np.array()
        rotational_inertial = np.array()

        return linear_body, rotational_body, linear_inertial, rotational_inertial

    @staticmethod
    def get_rotation_matrix(yaw, pitch, roll, transpose=False) -> np.array:
        r = np.array([np.cos(pitch) * np.cos(yaw),
                      np.cos(pitch) * np.sin(yaw),
                      -np.sin(pitch),
                      -np.cos(roll) * np.sin(yaw) + np.sin(roll) * np.sin(pitch) * np.cos(yaw),
                      np.cos(roll) * np.cos(yaw) + np.sin(roll) * np.sin(pitch) * np.sin(yaw),
                      np.sin(roll) * np.cos(pitch),
                      np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.sin(pitch) * np.cos(yaw),
                      -np.sin(roll) * np.cos(yaw) + np.cos(roll) * np.sin(pitch) * np.sin(yaw),
                      np.cos(roll) * np.cos(pitch)]).reshape(3, 3)

        if transpose:
            r = r.transpose()

        return r

    def run_simulation(self) -> None:
        pass
