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

from utils.array_list_operations import compare_matrices, remove_repeated_list


def remove_repeated_nodes(nodes, elements, precision=None):
    """
    This function is used to generate a mesh by removing repeated nodes and
    then reorder the nodes and elements.
    """
    index = elements.ravel()
    nodes = nodes[index]
    if precision is not None:
        nodes = nodes.round(precision)
    new_nodes = np.array(remove_repeated_list(nodes.tolist()))

    index = compare_matrices(new_nodes, nodes)
    new_elements = index.reshape(elements.shape)
    return new_nodes, new_elements
