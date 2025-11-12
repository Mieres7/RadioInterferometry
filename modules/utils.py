"""
Utilidades Generales
"""

import numpy as np

def degree_to_time(theta, is_rad=False):
    if is_rad:
        theta = np.rad2deg(theta)
    h = int(theta / 15)
    m = int(((theta / 15) - h) * 60)
    s = ((((theta / 15) - h) * 60) - m) * 60
    return h, m, s
