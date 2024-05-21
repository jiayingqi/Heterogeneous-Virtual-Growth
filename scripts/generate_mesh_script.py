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
