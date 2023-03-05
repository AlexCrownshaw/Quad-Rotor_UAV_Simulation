import numpy as np


class Disturbance:

    def __init__(self, disturbances):
        self.disturbances = disturbances

    def get_disturbance(self, t) -> np.array:
        try:
            if self.disturbances[1]["time"] <= t:
                self.disturbances = self.disturbances[1:]
        except IndexError:
            pass

        if self.disturbances[0]["time"] <= t < self.disturbances[0]["time"] + self.disturbances[0]["duration"]:
            D = np.array([self.disturbances[0]["F_x"], self.disturbances[0]["F_y"], self.disturbances[0]["F_z"]])
        else:
            D = np.zeros(3)

        return D
