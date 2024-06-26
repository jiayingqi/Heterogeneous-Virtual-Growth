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


def create_rectangle_mesh(lx, ly, nelex, neley):
    """Generate a rectangular mesh.

    Args:
        lx: Length of the domain in x-direction.
        ly: Length of the domain in y-direction.
        nelex: Number of elements in x-direction.
        neley: Number of elements in y-direction.
    """

    # Nodes
    # -------------------------------------------------------------------------
    len_elex = lx / nelex  # element length in x-direction
    len_eley = ly / neley  # element length in y-direction

    nodex = np.arange(nelex+1) * len_elex
    nodey = np.arange(neley+1) * len_eley

    nodex = np.tile(nodex, (neley+1, 1))
    nodey = np.tile(nodey, (1, nelex+1))

    nodex = nodex.reshape(nodex.size, 1, order="F")
    nodey = nodey.reshape(nodey.size, 1)
    node = np.hstack([nodex, nodey])  # Nodal coordinate list
    # The first column is x-coordinates and the second column is y-coordinates

    # Elements
    # -------------------------------------------------------------------------
    node_vector = np.arange((nelex+1)*(neley+1))
    node_mat = node_vector.reshape(neley+1, nelex+1, order="F")
    # numbering of the global nodes:
    # y  2   5   8   11
    # ^  1   4   7   10
    # |  0   3   6   9
    # O--->x

    # submatrices
    LeftUpperSubMat = node_mat[:-1, :-1].reshape(nelex*neley, 1, order="F")
    RightUpperSubMat = node_mat[:-1, 1:].reshape(nelex*neley, 1, order="F")
    RightLowerSubMat = node_mat[1:, 1:].reshape(nelex*neley, 1, order="F")
    LeftLowerSubMat = node_mat[1:, :-1].reshape(nelex*neley, 1, order="F")
    # element coordinates
    element = np.hstack([LeftUpperSubMat, RightUpperSubMat,
                         RightLowerSubMat, LeftLowerSubMat])
    # numbering of the elements
    # y
    # ^  2   4   6
    # |  1   3   5
    # O--->x

    return node, element
