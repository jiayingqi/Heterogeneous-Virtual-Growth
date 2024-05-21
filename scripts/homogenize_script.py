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
