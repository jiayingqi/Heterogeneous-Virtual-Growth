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
import scipy as sp

from homogenization.global_stiffness_matrix_2d import global_stiffness_matrix
from utils.array_list_operations import compare_matrices, find_indices


def topology_matrices(nodes):
    """Find the topology matrices B_0, B_a, and B_eps of the given unit cell."""
    min_x, max_x = min(nodes[0]), max(nodes[0])
    min_y, max_y = min(nodes[1]), max(nodes[1])
    lx, ly = max_x-min_x, max_y-min_y

    # Find master and slave nodes
    left_master_nodes = np.argwhere(np.isclose(nodes[0], min_x)).flatten()
    bottom_master_nodes = np.argwhere(np.isclose(nodes[1], min_y)).flatten()
    right_slave_nodes = np.argwhere(np.isclose(nodes[0], max_x)).flatten()
    top_slave_nodes = np.argwhere(np.isclose(nodes[1], max_y)).flatten()
    master_nodes = np.hstack([left_master_nodes, bottom_master_nodes])
    slave_nodes = np.hstack([right_slave_nodes, top_slave_nodes])

    assert left_master_nodes.size == right_slave_nodes.size
    assert bottom_master_nodes.size == top_slave_nodes.size

    # Find interior nodes
    all_nodes = np.arange(nodes.shape[1])
    interior_nodes = np.setdiff1d(all_nodes, np.hstack((
        master_nodes, slave_nodes)))
    assert (master_nodes.size + slave_nodes.size + interior_nodes.size == nodes.shape[1])

    # Compute B_0 matrix
    independent_nodes = np.hstack((master_nodes, interior_nodes))
    B_0 = np.zeros((all_nodes.size, independent_nodes.size), dtype=int)
    B_0[independent_nodes, np.arange(independent_nodes.size).astype(int)] = 1

    left_coords = nodes.T[left_master_nodes]
    left_coords[:, 0] += lx
    right_coords = nodes.T[right_slave_nodes]
    indices = find_indices(independent_nodes, left_master_nodes)
    args = compare_matrices(left_coords, right_coords, precision=6)
    cols = indices[args]
    B_0[right_slave_nodes, cols] = 1

    bottom_coords = nodes.T[bottom_master_nodes]
    bottom_coords[:, 1] += ly
    top_coords = nodes.T[top_slave_nodes]
    indices = find_indices(independent_nodes, bottom_master_nodes)
    args = compare_matrices(bottom_coords, top_coords, precision=6)
    cols = indices[args]
    B_0[top_slave_nodes, cols] = 1

    assert np.allclose(np.sum(B_0, axis=1), 1)
    B_0 = np.kron(B_0, np.eye(2, dtype=int))
    assert B_0.shape == (2*all_nodes.size, 2*independent_nodes.size)

    # Compute B_a matrix
    B_a = np.zeros((all_nodes.size, 2), dtype=int)
    B_a[right_slave_nodes, 0] = 1
    B_a[top_slave_nodes, 1] = 1
    B_a = np.kron(B_a, np.eye(2, dtype=int))
    assert B_a.shape == (2*all_nodes.size, 4)

    a_1x, a_1y = lx, 0.0
    a_2x, a_2y = 0.0, ly
    B_eps = np.array([
        [a_1x, 0.0, a_1y/2],
        [0.0, a_1y, a_1x/2],
        [a_2x, 0.0, a_2y/2],
        [0.0, a_2y, a_2x/2],
    ])

    V = np.linalg.det(np.array([
        [a_1x, a_1y],
        [a_2x, a_2y],
    ]))  # Volume
    return B_0, B_a, B_eps, V


def homogenized_elasticity_matrix_2d(nodes, elements, mat_table):
    """Compute the homogenized constitutive matrix."""
    K_uc = global_stiffness_matrix(nodes, elements, mat_table)
    B_0, B_a, B_eps, V = topology_matrices(nodes.T)
    eps = sp.sparse.eye(B_0.shape[1]) * 1e-8  # Eliminate the singularity
    D_0 = - np.linalg.inv(B_0.T@K_uc@B_0 + eps) @ (B_0.T@K_uc@B_a)
    D_a = B_0@D_0 + B_a
    K_delta_a = D_a.T @ K_uc @ D_a
    K_eps = B_eps.T @ K_delta_a @ B_eps / V
    return K_eps
