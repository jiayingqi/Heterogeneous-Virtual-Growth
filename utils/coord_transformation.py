"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Ke Liu (liuke@pku.edu.cn)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsor:
- David C. Crawford Faculty Scholar Award from the Department of Civil and
  Environmental Engineering and Grainger College of Engineering at the
  University of Illinois

Citations:
- Jia, Y., Liu, K., Zhang, X.S., 2024. Modulate stress distribution with
  bio-inspired irregular architected materials towards optimal tissue support.
  Nature Communications 15, 4072. https://doi.org/10.1038/s41467-024-47831-2
- Jia, Y., Liu, K., Zhang, X.S., 2024. Topology optimization of irregular
  multiscale structures with tunable responses using a virtual growth rule.
  Computer Methods in Applied Mechanics and Engineering 425, 116864.
  https://doi.org/10.1016/j.cma.2024.116864
"""

import numpy as np


def cartesian2polar(x, y):
    """Convert Cartesian coordinate to polar coordinates."""
    rho = np.sqrt(x**2 + y**2)  # Radius
    phi = np.arctan2(y, x)  # Angle
    return rho, phi

def polar2cartesian(rho, phi):
    """Convert polar coordinates to cartesian coordinates."""
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
