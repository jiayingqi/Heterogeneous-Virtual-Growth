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
from homogenization.generate_fem_mesh_2d import generate_fem_mesh_2d
from homogenization.generate_fem_mesh_3d import generate_fem_mesh_3d


dim = 3
match dim:
    case 2:
        d, m, n = 0.20, 0.75, 0.25
        block_size = 2 * m
        data_path = "virtual_growth_data/2d/"
        symbolic_graph = np.load("designs/2d/symbolic_graph.npy")
        generate_fem_mesh_2d(symbolic_graph, block_size, data_path,
                             mesh_path="designs/2d/", check_with_pyvista=True)
    case 3:
        m, n = 1.2, 0.0
        block_size = 2 * m
        data_path = "virtual_growth_data/3d/"
        symbolic_graph = np.load("designs/3d/symbolic_graph.npy")
        generate_fem_mesh_3d(
            symbolic_graph, block_size, data_path,
            mesh_path="designs/3d/", check_with_pyvista=True)
