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


block_library = {  # Pixel representation
    # y
    # ↑
    # O → x
    "corner": np.array([
        [0, 1, 0],
        [0, 1, 1],
        [0, 0, 0],
    ]).astype(int),
    "cross": np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]).astype(int),
    "line": np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ]).astype(int),
    "skew": np.array([
        [0, 1, 0],
        [0, 0, 1],
        [0, 0, 0],
    ]).astype(int),
    "t": np.array([
        [0, 0, 0],
        [1, 1, 1],
        [0, 1, 0],
    ]).astype(int),
    "v": np.array([
        [0, 0, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]).astype(int),
    "x": np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]).astype(int),
    "empty": np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]).astype(int),
}


def block_library_elem(d=1, m=6, n=0):
    """
    Element representation of blocks for plotting only.
    Inputs:
        d: One half thickness of the arm.
        m: Length of the arm.
        n: Offset of the center.
    """
    block = {
        # y
        # ↑
        # O → x
        "corner": {
            "nodes": np.array([
                [-d, m, 0],
                [d, m, 0],
                [m, d, 0],
                [m, -d, 0],
                [d-n+d*n/m, d-n+d*n/m, 0],
                [-(m*n+d*(m+n))/m, -(m*n+d*(m+n))/m, 0],
            ]).astype(float),
            "elements": np.array([
                [4, 0, 5, 4, 1],
                [4, 4, 5, 3, 2],
            ]),
            "cell_types": np.ones(2) * 9,
        },
        "cross": {
            "nodes": np.array([
                [-d, d, 0],
                [-d, -d, 0],
                [d, -d, 0],
                [d, d, 0],
                [-m, d, 0],
                [-m, -d, 0],
                [-d, -m, 0],
                [d, -m, 0],
                [m, -d, 0],
                [m, d, 0],
                [d, m, 0],
                [-d, m, 0],
            ]).astype(float),
            "elements": np.array([
                [4, 0, 1, 2, 3],
                [4, 0, 4, 5, 1],
                [4, 1, 6, 7, 2],
                [4, 3, 2, 8, 9],
                [4, 0, 3, 10, 11],
            ]),
            "cell_types": np.ones(5) * 9,
        },
        "line": {
            "nodes": np.array([
                [m, d, 0],
                [-m, d, 0],
                [-m, -d, 0],
                [m, -d, 0],
            ]).astype(float),
            "elements": np.array([
                [4, 0, 1, 2, 3],
            ]),
            "cell_types": np.ones(1) * 9,
        },
        "skew": {
            "nodes": np.array([
                [m, d, 0],
                [d, m, 0],
                [-d, m, 0],
                [m, -d, 0],
            ]).astype(float),
            "elements": np.array([
                [4, 0, 1, 2, 3],
            ]),
            "cell_types": np.ones(1) * 9,
        },
        "t": {
            "nodes": np.array([
                [-d, d+n, 0],
                [-d, -d+n, 0],
                [d, -d+n, 0],
                [d, d+n, 0],
                [-m, d, 0],
                [-m, -d, 0],
                [m, d, 0],
                [m, -d, 0],
                [-d, -m, 0],
                [d, -m, 0],
            ]).astype(float),
            "elements": np.array([
                [4, 0, 1, 2, 3],
                [4, 4, 5, 1, 0],
                [4, 3, 2, 7, 6],
                [4, 1, 8, 9, 2],
            ]),
            "cell_types": np.ones(4) * 9,
        },
        "v": {
            "nodes": np.array([
                [-d, -m+2*d, 0],
                [-d, -m, 0],
                [d, -m, 0],
                [d, -m+2*d, 0],
                [-m, d, 0],
                [-m, -d, 0],
                [m, d, 0],
                [m, -d, 0],
            ]).astype(float),
            "elements": np.array([
                [4, 0, 1, 2, 3],
                [4, 4, 5, 1, 0],
                [4, 3, 2, 7, 6],
            ]),
            "cell_types": np.ones(3) * 9,
        },
        "x": {
            "nodes": np.array([
                [-m+2*d, d/2, 0],
                [-m+2*d, -d/2, 0],
                [-d/2, -m+2*d, 0],
                [d/2, -m+2*d, 0],
                [m-2*d, -d/2, 0],
                [m-2*d, d/2, 0],
                [d/2, m-2*d, 0],
                [-d/2, m-2*d, 0],
                [-m, d, 0],
                [-m, -d, 0],
                [-d, -m, 0],
                [d, -m, 0],
                [m, -d, 0],
                [m, d, 0],
                [d, m, 0],
                [-d, m, 0],
            ]).astype(float),
            "elements": np.array([
                [4, 0, 8, 9, 1],
                [4, 1, 9, 10, 2],
                [4, 2, 10, 11, 3],
                [4, 3, 11, 12, 4],
                [4, 4, 12, 13, 5],
                [4, 5, 13, 14, 6],
                [4, 6, 14, 15, 7],
                [4, 7, 15, 8, 0],
            ]),
            "cell_types": np.ones(8) * 9,
        },
    }

    return block
