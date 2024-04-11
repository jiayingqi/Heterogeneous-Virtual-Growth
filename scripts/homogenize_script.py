"""
Authors:
- Yingqi Jia (yingqij2@illinois.edu)
- Ke Liu (liuke@pku.edu.cn)
- Xiaojia Shelly Zhang (zhangxs@illinois.edu)

Sponsors:
- U.S. National Science Foundation (NSF) EAGER Award CMMI-2127134
- U.S. NSF CAREER Award CMMI-2047692
- U.S. NSF Award CMMI-2245251
- U.S. Air Force Office of Scientific Research (AFOSR YIP, FA9550-23-1-0297)
- National Natural Science Foundation of China (Grant 12372159)

Reference:
- To be added
"""

import numpy as np
from homogenization.homogenization_2d import homogenized_elasticity_matrix_2d
from homogenization.homogenization_3d import homogenized_elasticity_matrix_3d


dim = 3
match dim:
    case 2:
        mesh = np.load("designs/2d/fem_mesh.npz")
        nodes = mesh["nodes"]
        elements = mesh["elements"]
        mat_table = {
            "E": 30,
            "nu": 0.25,
            "PSflag": "PlaneStress",
            "RegMesh": False,
            "thickness": 1.0,
        }
        K_eps = homogenized_elasticity_matrix_2d(nodes, elements, mat_table)
        print(K_eps.round(2))
    case 3:
        mesh = np.load("designs/3d/fem_mesh.npz")
        nodes = mesh["nodes"]
        elements = mesh["elements"]
        local_y_directs = mesh["local_y_directs"]
        mat_table = {
            "E": 70,
            "nu": 0.25,
            "thickness": 1.0,
        }
        K_eps = homogenized_elasticity_matrix_3d(nodes, elements, local_y_directs, mat_table)
        print(K_eps.round(2))
