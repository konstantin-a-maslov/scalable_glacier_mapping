import numpy as np


def norm_to_vis(image, p=(2, 98)):
    p_l = np.percentile(image, p[0], axis=(0, 1))
    p_r = np.percentile(image, p[1], axis=(0, 1))
    return (image - p_l) / (p_r - p_l)
