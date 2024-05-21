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
import scipy as sp

from homogenization.global_stiffness_matrix_3d import global_stiffness_matrix
from utils.array_list_operations import compare_matrices, find_indices


def topology_matrices(nodes):
    """Find the topology matrices B_0, B_a, and B_eps of the given unit cell."""
    min_x, max_x = min(nodes[:, 0]), max(nodes[:, 0])
    min_y, max_y = min(nodes[:, 1]), max(nodes[:, 1])
    min_z, max_z = min(nodes[:, 2]), max(nodes[:, 2])
    lx, ly, lz = max_x-min_x, max_y-min_y, max_z-min_z

    # Find master and slave nodes
    back_master_nodes = np.argwhere(np.isclose(nodes[:, 0], 0.0)).flatten()
    left_master_nodes = np.argwhere(np.isclose(nodes[:, 1], 0.0)).flatten()
    bottom_master_nodes = np.argwhere(np.isclose(nodes[:, 2], 0.0)).flatten()
    front_slave_nodes = np.argwhere(np.isclose(nodes[:, 0], lx)).flatten()
    right_slave_nodes = np.argwhere(np.isclose(nodes[:, 1], ly)).flatten()
    top_slave_nodes = np.argwhere(np.isclose(nodes[:, 2], lz)).flatten()

    master_nodes = np.hstack([back_master_nodes, left_master_nodes, bottom_master_nodes])
    slave_nodes = np.hstack([front_slave_nodes, right_slave_nodes, top_slave_nodes])

    assert back_master_nodes.size == front_slave_nodes.size
    assert left_master_nodes.size == right_slave_nodes.size
    assert bottom_master_nodes.size == top_slave_nodes.size

    # Find interior nodes
    all_nodes = np.arange(nodes.shape[0])
    interior_nodes = np.setdiff1d(all_nodes, np.hstack((
        master_nodes, slave_nodes)))
    assert (master_nodes.size + slave_nodes.size + interior_nodes.size == nodes.shape[0])

    # Compute B_0 matrix
    independent_nodes = np.hstack((master_nodes, interior_nodes))
    B_0 = np.zeros((all_nodes.size, independent_nodes.size), dtype=int)
    B_0[independent_nodes, np.arange(independent_nodes.size).astype(int)] = 1

    back_coords = nodes[back_master_nodes]
    back_coords[:, 0] += lx
    front_coords = nodes[front_slave_nodes]
    indices = find_indices(independent_nodes, back_master_nodes)
    args = compare_matrices(back_coords, front_coords, precision=6)
    cols = indices[args]
    B_0[front_slave_nodes, cols] = 1

    left_coords = nodes[left_master_nodes]
    left_coords[:, 1] += ly
    right_coords = nodes[right_slave_nodes]
    indices = find_indices(independent_nodes, left_master_nodes)
    args = compare_matrices(left_coords, right_coords, precision=6)
    cols = indices[args]
    B_0[right_slave_nodes, cols] = 1

    bottom_coords = nodes[bottom_master_nodes]
    bottom_coords[:, 2] += lz
    top_coords = nodes[top_slave_nodes]
    indices = find_indices(independent_nodes, bottom_master_nodes)
    args = compare_matrices(bottom_coords, top_coords, precision=6)
    cols = indices[args]
    B_0[top_slave_nodes, cols] = 1

    assert np.allclose(np.sum(B_0, axis=1), 1)
    B_0 = np.kron(B_0, np.eye(6, dtype=int))
    assert B_0.shape == (6*all_nodes.size, 6*independent_nodes.size)

    # Compute B_a matrix
    B_a = np.zeros((all_nodes.size, 3), dtype=int)
    B_a[front_slave_nodes, 0] = 1
    B_a[right_slave_nodes, 1] = 1
    B_a[top_slave_nodes, 2] = 1
    B_a = np.kron(B_a, np.vstack((np.eye(3), np.zeros((3, 3))))).astype(int)
    assert B_a.shape == (6*all_nodes.size, 9)

    a_1x, a_1y, a_1z = lx, 0.0, 0.0
    a_2x, a_2y, a_2z = 0.0, ly, 0.0
    a_3x, a_3y, a_3z = 0.0, 0.0, lz

    B_eps = np.array([
        [a_1x, 0.0, 0.0, a_1y/2, 0.0, a_1z/2],
        [0.0, a_1y, 0.0, a_1x/2, a_1z/2, 0.0],
        [0.0, 0.0, a_1z, 0.0, a_1y/2, a_1x/2],
        [a_2x, 0.0, 0.0, a_2y/2, 0.0, a_2z/2],
        [0.0, a_2y, 0.0, a_2x/2, a_2z/2, 0.0],
        [0.0, 0.0, a_2z, 0.0, a_2y/2, a_2x/2],
        [a_3x, 0.0, 0.0, a_3y/2, 0.0, a_3z/2],
        [0.0, a_3y, 0.0, a_3x/2, a_3z/2, 0.0],
        [0.0, 0.0, a_3z, 0.0, a_3y/2, a_3x/2],
    ])

    V = np.linalg.det(np.array([
        [a_1x, a_1y, a_1z],
        [a_2x, a_2y, a_2z],
        [a_3x, a_3y, a_3z],
    ]))  # Volume
    return B_0, B_a, B_eps, V


def homogenized_elasticity_matrix_3d(nodes, elements, local_y_directs, mat_table):
    """Compute the homogenized constitutive matrix."""
    E = mat_table["E"]
    nu = mat_table["nu"]
    t = mat_table["thickness"]

    G = E / 2 / (1+nu)
    A = t**2
    Iy = Iz = t**4 / 12
    J = Iy + Iz

    K_uc = global_stiffness_matrix(nodes, elements, local_y_directs, A, E, G, Iy, Iz, J)
    B_0, B_a, B_eps, V = topology_matrices(nodes)
    eps = sp.sparse.eye(B_0.shape[1]) * 1e-8  # Eliminate the singularity
    D_0 = - np.linalg.inv(B_0.T@K_uc@B_0 + eps) @ (B_0.T@K_uc@B_a)
    D_a = B_0@D_0 + B_a
    K_delta_a = D_a.T @ K_uc @ D_a
    return B_eps.T @ K_delta_a @ B_eps / V
