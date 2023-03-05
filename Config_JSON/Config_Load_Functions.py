import json

from typing import Tuple


def load_vehicle_properties(json_path: str) -> Tuple[dict, dict, dict, dict]:
    with open(json_path) as json_file:
        json_object = json.load(json_file)
        return json_object["properties"], json_object["dimensions"], json_object["motor"], json_object["propeller"]


def load_flight_path(json_path: str) -> Tuple[dict, dict]:
    with open(json_path) as json_file:
        json_object = json.load(json_file)
        return json_object["Initial_Conditions"], json_object["Maneuvers"]


def load_sensor_config(json_path: str) -> Tuple[dict, dict, dict]:
    with open(json_path) as json_file:
        json_object = json.load(json_file)
        return json_object["acc"], json_object["gyro"], json_object["mag"]


def load_disturbances(json_path: str) -> list:
    with open(json_path) as json_file:
        json_object = json.load(json_file)
        return json_object["disturbances"]
