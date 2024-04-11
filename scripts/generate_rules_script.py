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

from virtual_growth.pair_rules_2d import pair_rules_2d
from virtual_growth.pair_rules_3d import pair_rules_3d


dim = 3
match dim:
    case 2:
        block_names = ["corner", "cross", "line", "skew", "t", "v", "x"]
        pair_rules_2d(block_names, d=0.20, m=0.75, n=0.25, num_elems_d=3, num_elems_m=5,
                      path_name="virtual_growth_data/2d/")
    case 3:
        block_names = ["corner", "cross_line", "line", "plane_corner",
                       "cross", "plane_cross", "t", "t_line"]
        pair_rules_3d(block_names, d=0.2, m=1.2, n=0.0, num_elems_m=3,
                      path_name="virtual_growth_data/3d/")
