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
from virtual_growth.main import main


dim = 3
match dim:
    case 2:
        mesh_size = (2, 3)
        element_size = (2, 2)
        candidates = ["cross", "t"]
        num_elems = np.prod(mesh_size)
        frequency_hints = np.random.rand(num_elems, len(candidates))
        frequency_hints = frequency_hints / np.sum(frequency_hints, axis=1).reshape(-1, 1)
        d, m, n = 0.20, 0.75, 0.25
        data_path = "virtual_growth_data/2d/"
        save_path = "designs/2d/"
        fig_name = "symbolic_graph.jpg"
        gif_name = "symbolic_graph.gif"
    case 3:
        mesh_size = (1, 1, 1)
        element_size = (3, 3, 3)
        candidates = ["cross", "t_line", "cross_line"]
        num_elems = np.prod(mesh_size)
        frequency_hints = np.random.rand(num_elems, len(candidates))
        frequency_hints = frequency_hints / np.sum(frequency_hints, axis=1).reshape(-1, 1)
        d, m, n = 0.2, 1.0, 0.0
        data_path = "virtual_growth_data/3d/"
        save_path = "designs/3d/"
        fig_name = "symbolic_graph.jpg"
        gif_name = "symbolic_graph.gif"

if __name__ == "__main__":
    main(mesh_size, element_size, candidates, frequency_hints,
         d, m, n, num_tries=10, print_frequency=True, make_figure=True,
         make_gif=True, data_path=data_path, save_path=save_path,
         fig_name=fig_name, gif_name=gif_name, save_mesh=True,
         save_mesh_path=save_path, periodic=True)
