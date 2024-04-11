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
