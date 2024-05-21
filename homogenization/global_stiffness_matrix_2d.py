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
from scipy import sparse


def gaussian_point(l):
    """Weights and coordinates of Gauss integration points."""
    Wgt = 1
    r_vec = np.array([-1, 1, 1, -1])*np.sqrt(3)/3
    s_vec = np.array([-1, -1, 1, 1])*np.sqrt(3)/3
    return Wgt, r_vec[l], s_vec[l]


def local_shape_function(r, s):
    """Shape functions and derivatives with respect to the local coordinates."""
    r_vec = np.array([-1, 1, 1, -1])
    s_vec = np.array([-1, -1, 1, 1])
    N = (1+r_vec*r)*(1+s_vec*s)/4
    dNdr = r_vec*(1+s_vec*s)/4
    dNds = (1+r_vec*r)*s_vec/4
    return N, dNdr, dNds


def global_shape_function(dNdr, dNds, vx, vy):
    """Shape functions and derivatives with respect to the global coordinates."""
    dxdr = dNdr @ vx
    dxds = dNds @ vx
    dydr = dNdr @ vy
    dyds = dNds @ vy
    j = dxdr*dyds - dxds*dydr
    dNdx = (dNdr*dyds - dNds*dydr)/j
    dNdy = -(dNdr*dxds - dNds*dxdr)/j
    return dNdx, dNdy, j


def strain_displacement_matrix(dNdx, dNdy):
    """Shape function matrix."""
    B = np.array([
        [dNdx[0], 0, dNdx[1], 0, dNdx[2], 0, dNdx[3], 0],
        [0, dNdy[0], 0, dNdy[1], 0, dNdy[2], 0, dNdy[3]],
        [dNdy[0], dNdx[0], dNdy[1], dNdx[1], dNdy[2], dNdx[2], dNdy[3], dNdx[3]]
    ])
    return B


def elasticity_matrix(mat_table):
    """Elasticity matrix."""
    E = mat_table["E"]
    nu = mat_table["nu"]
    if mat_table["PSflag"] not in ["PlaneStress", "PlaneStrain"]:
        raise ValueError("Unsupported 'PSflag'.")
    if mat_table["PSflag"] == "PlaneStrain":
        E = E/(1-nu**2)
        nu = nu/(1-nu)
    D = np.array([
        [1, nu, 0],
        [nu, 1, 0],
        [0, 0, (1-nu)/2]
    ]) * E/(1-nu**2)
    return D


def element_stiffness_matrix(vx, vy, mat_table, D):
    """Element stiffness matrix."""
    ke = np.zeros((8, 8))
    for l in np.arange(4):
        Wgt, r, s = gaussian_point(l)
        _, dNdr, dNds = local_shape_function(r, s)
        dNdx, dNdy, j = global_shape_function(dNdr, dNds, vx, vy)
        B = strain_displacement_matrix(dNdx, dNdy)
        ke += B.T@D@B * Wgt*j*mat_table["thickness"]
    return ke


def global_stiffness_matrix(nodes, elements, mat_table):
    """
    Global stiffness matrix.
    Element connectivity should be counterclockwise as follows.
    3 ← 2
        ↑
    0 → 1
    """
    D = elasticity_matrix(mat_table)
    num_elems = elements.shape[0]
    row, col = np.zeros(64*num_elems, dtype=int), np.zeros(64*num_elems, dtype=int)
    data = np.zeros(64*num_elems)

    for el in np.arange(num_elems):  # Each element
        vx = nodes[elements[el, :], 0]  # x-coordinates of the vertexes of the element
        vy = nodes[elements[el, :], 1]  # y-coordinates of the vertexes of the element
        if ~mat_table["RegMesh"] or el == 0:
            ke = element_stiffness_matrix(vx, vy, mat_table, D)  # Element stiffness matrix
        elem_dofs = np.vstack(
            [2*elements[el, :], 2*elements[el, :]+1]).reshape(8, 1, order="F")  # Element global dofs
        index = np.arange(64*el, 64*(el+1), 1)
        row[index] = np.tile(elem_dofs, (8, 1))[:, 0]
        col[index] = np.tile(elem_dofs, (1, 8)).reshape(64, 1)[:, 0]
        data[index] = ke.reshape(64, 1, order="F")[:, 0]
    return sparse.csc_matrix((data, (row, col)))  # Sparse global stiffness matrix
