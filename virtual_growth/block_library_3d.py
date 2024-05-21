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


block_library = {  # Pixel representation
    # x ← ⵙ z
    #     ↓
    #     y
    "corner": np.array([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]),
    "cross_line": np.array([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
    ]),
    "line": np.array([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]),
    "plane_corner": np.array([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]),
    "cross": np.array([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
    ]),
    "plane_cross": np.array([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]),
    "t": np.array([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]),
    "t_line": np.array([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
    ]),
    "x": np.array([
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
    ]),
    "empty": np.array([
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    ]),
}


def block_library_elem(d=1, m=6):
    """
    Element representation of blocks for plotting only.
    Inputs:
        d: One half length of each side of the kernal cube.
        m: Length of the arm.
    """
    block = {
        # x ← ⵙ z
        #     ↓
        #     y
        "corner": {
            "nodes": np.array([
                [d, d, -d],
                [-d, d, -d],
                [-d, -d, -d],
                [d, -d, -d],
                [d, d, d],
                [-d, d, d],
                [-d, -d, d],
                [d, -d, d],
                [d, -m, -d],
                [-d, -m, -d],
                [-d, -m, d],
                [d, -m, d],
                [-m, d, -d],
                [-m, -d, -d],
                [-m, -d, d],
                [-m, d, d],
                [d, d, -m],
                [-d, d, -m],
                [-d, -d, -m],
                [d, -d, -m],
            ]).astype(float),
            "elements": np.array([
                [8, 0, 1, 2, 3, 4, 5, 6, 7],
                [8, 3, 2, 9, 8, 7, 6, 10, 11],
                [8, 1, 12, 13, 2, 5, 15, 14, 6],
                [8, 16, 17, 18, 19, 0, 1, 2, 3],
            ]),
            "cell_types": np.ones(4) * 12,
        },
        "cross_line": {
            "nodes": np.array([
                [d, d, -d],
                [-d, d, -d],
                [-d, -d, -d],
                [d, -d, -d],
                [d, d, d],
                [-d, d, d],
                [-d, -d, d],
                [d, -d, d],
                [-m, d, -d],
                [-m, -d, -d],
                [-m, -d, d],
                [-m, d, d],
                [m, d, -d],
                [m, -d, -d],
                [m, -d, d],
                [m, d, d],
                [d, d, m],
                [-d, d, m],
                [-d, -d, m],
                [d, -d, m],
                [d, d, -m],
                [-d, d, -m],
                [-d, -d, -m],
                [d, -d, -m],
                [d, m, -d],
                [-d, m, -d],
                [-d, m, d],
                [d, m, d],
            ]).astype(float),
            "elements": np.array([
                [8, 0, 1, 2, 3, 4, 5, 6, 7],
                [8, 1, 8, 9, 2, 5, 11, 10, 6],
                [8, 12, 0, 3, 13, 15, 4, 7, 14],
                [8, 4, 5, 6, 7, 16, 17, 18, 19],
                [8, 20, 21, 22, 23, 0, 1, 2, 3],
                [8, 24, 25, 1, 0, 27, 26, 5, 4],
            ]),
            "cell_types": np.ones(6) * 12,
        },
        "plane_corner": {
            "nodes": np.array([
                [d, d, -d],
                [-d, d, -d],
                [-d, -d, -d],
                [d, -d, -d],
                [d, d, d],
                [-d, d, d],
                [-d, -d, d],
                [d, -d, d],
                [d, -m, -d],
                [-d, -m, -d],
                [-d, -m, d],
                [d, -m, d],
                [-m, d, -d],
                [-m, -d, -d],
                [-m, -d, d],
                [-m, d, d],
            ]).astype(float),
            "elements": np.array([
                [8, 0, 1, 2, 3, 4, 5, 6, 7],
                [8, 3, 2, 9, 8, 7, 6, 10, 11],
                [8, 1, 12, 13, 2, 5, 15, 14, 6],
            ]),
            "cell_types": np.ones(3) * 12,
        },
        "line": {
            "nodes": np.array([
                [d, m, -d],
                [-d, m, -d],
                [-d, m, d],
                [d, m, d],
                [d, -m, -d],
                [-d, -m, -d],
                [-d, -m, d],
                [d, -m, d],
            ]).astype(float),
            "elements": np.array([
                [8, 0, 1, 2, 3, 4, 5, 6, 7],
            ]),
            "cell_types": np.ones(1) * 12,
        },
        "cross": {
            "nodes": np.array([
                [d, d, -d],
                [-d, d, -d],
                [-d, -d, -d],
                [d, -d, -d],
                [d, d, d],
                [-d, d, d],
                [-d, -d, d],
                [d, -d, d],
                [-m, d, -d],
                [-m, -d, -d],
                [-m, -d, d],
                [-m, d, d],
                [m, d, -d],
                [m, -d, -d],
                [m, -d, d],
                [m, d, d],
                [d, d, m],
                [-d, d, m],
                [-d, -d, m],
                [d, -d, m],
                [d, d, -m],
                [-d, d, -m],
                [-d, -d, -m],
                [d, -d, -m],
                [d, m, -d],
                [-d, m, -d],
                [-d, m, d],
                [d, m, d],
                [d, -m, -d],
                [-d, -m, -d],
                [-d, -m, d],
                [d, -m, d],
            ]).astype(float),
            "elements": np.array([
                [8, 0, 1, 2, 3, 4, 5, 6, 7],
                [8, 1, 8, 9, 2, 5, 11, 10, 6],
                [8, 12, 0, 3, 13, 15, 4, 7, 14],
                [8, 4, 5, 6, 7, 16, 17, 18, 19],
                [8, 20, 21, 22, 23, 0, 1, 2, 3],
                [8, 24, 25, 1, 0, 27, 26, 5, 4],
                [8, 3, 2, 29, 28, 7, 6, 30, 31],
            ]),
            "cell_types": np.ones(7) * 12,
        },
        "plane_cross": {
            "nodes": np.array([
                [d, d, -d],
                [-d, d, -d],
                [-d, -d, -d],
                [d, -d, -d],
                [d, d, d],
                [-d, d, d],
                [-d, -d, d],
                [d, -d, d],
                [-m, d, -d],
                [-m, -d, -d],
                [-m, -d, d],
                [-m, d, d],
                [m, d, -d],
                [m, -d, -d],
                [m, -d, d],
                [m, d, d],
                [d, m, -d],
                [-d, m, -d],
                [-d, m, d],
                [d, m, d],
                [d, -m, -d],
                [-d, -m, -d],
                [-d, -m, d],
                [d, -m, d],
            ]).astype(float),
            "elements": np.array([
                [8, 0, 1, 2, 3, 4, 5, 6, 7],
                [8, 1, 8, 9, 2, 5, 11, 10, 6],
                [8, 12, 0, 3, 13, 15, 4, 7, 14],
                [8, 16, 17, 1, 0, 19, 18, 5, 4],
                [8, 3, 2, 21, 20, 7, 6, 22, 23],
            ]),
            "cell_types": np.ones(5) * 12,
        },
        "t": {
            "nodes": np.array([
                [d, d, -d],
                [-d, d, -d],
                [-d, -d, -d],
                [d, -d, -d],
                [d, d, d],
                [-d, d, d],
                [-d, -d, d],
                [d, -d, d],
                [-m, d, -d],
                [-m, -d, -d],
                [-m, -d, d],
                [-m, d, d],
                [m, d, -d],
                [m, -d, -d],
                [m, -d, d],
                [m, d, d],
                [d, m, -d],
                [-d, m, -d],
                [-d, m, d],
                [d, m, d],
            ]).astype(float),
            "elements": np.array([
                [8, 0, 1, 2, 3, 4, 5, 6, 7],
                [8, 1, 8, 9, 2, 5, 11, 10, 6],
                [8, 12, 0, 3, 13, 15, 4, 7, 14],
                [8, 16, 17, 1, 0, 19, 18, 5, 4],
            ]),
            "cell_types": np.ones(4) * 12,
        },
        "t_line": {
            "nodes": np.array([
                [d, d, -d],
                [-d, d, -d],
                [-d, -d, -d],
                [d, -d, -d],
                [d, d, d],
                [-d, d, d],
                [-d, -d, d],
                [d, -d, d],
                [-m, d, -d],
                [-m, -d, -d],
                [-m, -d, d],
                [-m, d, d],
                [m, d, -d],
                [m, -d, -d],
                [m, -d, d],
                [m, d, d],
                [d, m, -d],
                [-d, m, -d],
                [-d, m, d],
                [d, m, d],
                [d, d, m],
                [-d, d, m],
                [-d, -d, m],
                [d, -d, m],
            ]).astype(float),
            "elements": np.array([
                [8, 0, 1, 2, 3, 4, 5, 6, 7],
                [8, 1, 8, 9, 2, 5, 11, 10, 6],
                [8, 12, 0, 3, 13, 15, 4, 7, 14],
                [8, 16, 17, 1, 0, 19, 18, 5, 4],
                [8, 4, 5, 6, 7, 20, 21, 22, 23],
            ]),
            "cell_types": np.ones(5) * 12,
        },
    }

    return block
