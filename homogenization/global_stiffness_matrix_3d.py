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
from scipy import sparse


def element_stiffness_matrix(nodes, elements, local_y_directs, A, E, G, Iy, Iz, J):
    """Compute the element stiffness matrix."""
    # Geometries
    num_elems = elements.shape[0]
    elem_nodes = nodes[elements.flatten()].reshape(num_elems, 6)
    le = np.sqrt((elem_nodes[:, 0] - elem_nodes[:, 3])**2
                 + (elem_nodes[:, 1] - elem_nodes[:, 4])**2
                 + (elem_nodes[:, 2] - elem_nodes[:, 5])**2)
    a = le / 2

    # Element stiffness matrix
    zero = np.zeros(num_elems)
    ke = np.array([
        [A*E/(2*a), zero, zero, zero, zero, zero, -A*E/(2*a), zero, zero, zero, zero, zero],
        [zero, 3*E*Iz/(2*a**3), zero, zero, zero, 3*E*Iz/(2*a**2), zero, -3*E*Iz/(2*a**3), zero, zero, zero, 3*E*Iz/(2*a**2)],
        [zero, zero, 3*E*Iy/(2*a**3), zero, -3*E*Iy/(2*a**2), zero, zero, zero, -3*E*Iy/(2*a**3), zero, -3*E*Iy/(2*a**2), zero],
        [zero, zero, zero, G*J/(2*a), zero, zero, zero, zero, zero, -G*J/(2*a), zero, zero],
        [zero, zero, zero, zero, 2*E*Iy/a, zero, zero, zero, 3*E*Iy/(2*a**2), zero, E*Iy/a, zero],
        [zero, zero, zero, zero, zero, 2*E*Iz/a, zero, -3*E*Iz/(2*a**2), zero, zero, zero, E*Iz/a],
        [zero, zero, zero, zero, zero, zero, A*E/(2*a), zero, zero, zero, zero, zero],
        [zero, zero, zero, zero, zero, zero, zero, 3*E*Iz/(2*a**3), zero, zero, zero, -3*E*Iz/(2*a**2)],
        [zero, zero, zero, zero, zero, zero, zero, zero, 3*E*Iy/(2*a**3), zero, 3*E*Iy/(2*a**2), zero],
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, G*J/(2*a), zero, zero],
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, 2*E*Iy/a, zero],
        [zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, 2*E*Iz/a],
    ])  # Triangular matrix
    ke = np.rot90(ke, axes=(0, 2))
    ke = np.transpose(ke, axes=(0, 2, 1))
    ke = np.flip(ke, axis=0)

    ke += np.transpose(ke, axes=(0, 2, 1))  # Recover to the full matrix
    index = np.arange(12)
    ke[:, index, index] /= 2

    # Transformation matrix
    local_x_directs = np.vstack([
        elem_nodes[:, 3] - elem_nodes[:, 0],
        elem_nodes[:, 4] - elem_nodes[:, 1],
        elem_nodes[:, 5] - elem_nodes[:, 2],
    ]).T / le.reshape(-1, 1)
    local_y_directs = local_y_directs / np.linalg.norm(
        local_y_directs, 2, axis=1).reshape(-1, 1)
    local_z_directs = np.cross(local_x_directs, local_y_directs)

    T3 = np.zeros((num_elems, 3, 3))
    T3[:, 0, 0], T3[:, 0, 1], T3[:, 0, 2] = local_x_directs.T
    T3[:, 1, 0], T3[:, 1, 1], T3[:, 1, 2] = local_y_directs.T
    T3[:, 2, 0], T3[:, 2, 1], T3[:, 2, 2] = local_z_directs.T
    T = np.zeros((num_elems, 12, 12))
    T[:, :3, :3] = T[:, 3:6, 3:6] = T[:, 6:9, 6:9] = T[:, 9:, 9:] = T3

    # Transformed element stiffness matrix
    ke = np.einsum('mij,mjk->mik', np.transpose(T, axes=(0, 2, 1)), ke)
    ke = np.einsum('mij,mjk->mik', ke, T)
    return ke


def global_stiffness_matrix(nodes, elements, local_y_directs, A, E, G, Iy, Iz, J):
    """
    Compute the global stiffness matrix.
    Reference: http://what-when-how.com/the-finite-element-method/fem-for-frames-finite-element-method-part-1/
    Inputs:
        nodes: An (num_nodes, 3) array. The three columns are x-, y-, and z-
               coordinates of each node, respectively.
        elements: An (num_elems, 2) array. The two columns are the start and
                  end nodes of each element, respectively.
        local_y_directs: An (num_elems, 3) array. The three columns are x-, y-,
                         and z-components of y-direction in the local system,
                         respectively.
        A: A float or an (N,) array. Cross-sectional areas for each element.
        E: A float or an (N,) array. Young's moduli for each element.
        G: A float or an (N,) array. Shear moduli for each element.
        Iy: A float or an (N,) array. The second moment of area with respect to
            the local y-axis.
        Iz: A float or an (N,) array. The second moment of area with respect to
            the local z-axis.
        J: A float or an (N,) array. The polar moment of inertia with respect
           to the local x-axis.
    Output:
        K: An (num_nodes*6, num_nodes*6) array. The global stiffness matrix.
    """
    ke = element_stiffness_matrix(nodes, elements, local_y_directs, A, E, G, Iy, Iz, J)
    num_nodes = nodes.shape[0]
    num_dofs = 6 * num_nodes

    temp1 = np.tile(elements[:, 0].reshape(-1, 1), (1, 6))
    temp2 = np.tile(elements[:, 1].reshape(-1, 1), (1, 6))
    temp = np.hstack((temp1, temp2))
    temp[:, 0] = 6 * temp[:, 0]
    temp[:, 1] = 6 * temp[:, 1] + 1
    temp[:, 2] = 6 * temp[:, 2] + 2
    temp[:, 3] = 6 * temp[:, 3] + 3
    temp[:, 4] = 6 * temp[:, 4] + 4
    temp[:, 5] = 6 * temp[:, 5] + 5
    temp[:, 6] = 6 * temp[:, 6]
    temp[:, 7] = 6 * temp[:, 7] + 1
    temp[:, 8] = 6 * temp[:, 8] + 2
    temp[:, 9] = 6 * temp[:, 9] + 3
    temp[:, 10] = 6 * temp[:, 10] + 4
    temp[:, 11] = 6 * temp[:, 11] + 5
    temp = np.tile(temp, (12, 1, 1))
    temp = np.rot90(temp, axes=(1, 0))

    row = np.rot90(temp, axes=(2, 1)).flatten()
    col = temp.flatten()
    data = ke.flatten()
    index = np.setdiff1d(np.arange(data.size),
                         np.argwhere(np.isclose(data, 0))[:, 0])
    return sparse.csc_matrix((data[index], (row[index], col[index])),
                             shape=(num_dofs, num_dofs))
