import numpy as np


def get_z_factor(lat):
    LATS = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80])
    ZS = np.array([
        0.00000898, 0.00000912, 0.00000956, 0.00001036, 0.00001171,
        0.00001395, 0.00001792, 0.00002619, 0.00005156
    ])
    return np.interp(lat, LATS, ZS)
    