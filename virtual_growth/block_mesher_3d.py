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

from utils.remove_repeated_nodes import remove_repeated_nodes


def block_mesher(name="corner", m=3, n=0, num_elems_m=3):
    """
    Generate the fem mesh with beam elements for the input 3d block.
    Inputs:
        m: Length of the arm.
        n: Offset of the center.
        num_elems_m: Number of elements in the m-direction.
    """

    def process_raw_mesh(nodes_list, elements_list):
        """
        Remove repeated nodes and convert the mesh info to the Pyvista format.
        """
        all_nodes = np.vstack(nodes_list)
        all_elements = np.zeros(np.vstack(elements_list).shape)

        elem_count = 0
        node_count = 0
        for nodes, elements in zip(nodes_list, elements_list):
            elements += node_count
            all_elements[elem_count:elem_count+elements.shape[0]] = elements
            elem_count += elements.shape[0]
            node_count += nodes.shape[0]

        nodes, elements = remove_repeated_nodes(
            all_nodes, all_elements.astype(int), precision=12)
        elements = np.hstack((np.ones((elements.shape[0], 1))*elements.shape[1],
                             elements)).astype(int)
        cell_types = np.ones(elements.shape[0], dtype=int) * 3
        return elements, cell_types, nodes

    pnt_plus_x = (m, 0.0, 0.0)
    pnt_minus_x = (-m, 0.0, 0.0)
    pnt_plus_y = (0.0, m, 0.0)
    pnt_minus_y = (0.0, -m, 0.0)
    pnt_plus_z = (0.0, 0.0, m)
    pnt_minus_z = (0.0, 0.0, -m)
    ref_elements = np.vstack((np.arange(num_elems_m), np.arange(num_elems_m)+1)).T

    if name == "corner":
        origin = (n, n, n)
        nodes1 = np.linspace(origin, pnt_minus_x, num_elems_m+1)
        nodes2 = np.linspace(origin, pnt_minus_y, num_elems_m+1)
        nodes3 = np.linspace(origin, pnt_minus_z, num_elems_m+1)
        nodes = [nodes1, nodes2, nodes3]

    elif name == "cross_line":
        origin = (0.0, -n, 0.0)
        nodes1 = np.linspace(origin, pnt_plus_x, num_elems_m+1)
        nodes2 = np.linspace(origin, pnt_minus_x, num_elems_m+1)
        nodes3 = np.linspace(origin, pnt_plus_z, num_elems_m+1)
        nodes4 = np.linspace(origin, pnt_minus_z, num_elems_m+1)
        nodes5 = np.linspace(origin, pnt_plus_y, num_elems_m+1)
        nodes = [nodes1, nodes2, nodes3, nodes4, nodes5]

    elif name == "line":
        origin = (0.0, 0.0, 0.0)
        nodes1 = np.linspace(origin, pnt_plus_y, num_elems_m+1)
        nodes2 = np.linspace(origin, pnt_minus_y, num_elems_m+1)
        nodes = [nodes1, nodes2]

    elif name == "plane_corner":
        origin = (n, n, 0.0)
        nodes1 = np.linspace(origin, pnt_minus_x, num_elems_m+1)
        nodes2 = np.linspace(origin, pnt_minus_y, num_elems_m+1)
        nodes = [nodes1, nodes2]

    elif name == "cross":
        origin = (0.0, 0.0, 0.0)
        nodes1 = np.linspace(origin, pnt_plus_x, num_elems_m+1)
        nodes2 = np.linspace(origin, pnt_minus_x, num_elems_m+1)
        nodes3 = np.linspace(origin, pnt_plus_y, num_elems_m+1)
        nodes4 = np.linspace(origin, pnt_minus_y, num_elems_m+1)
        nodes5 = np.linspace(origin, pnt_plus_z, num_elems_m+1)
        nodes6 = np.linspace(origin, pnt_minus_z, num_elems_m+1)
        nodes = [nodes1, nodes2, nodes3, nodes4, nodes5, nodes6]

    elif name == "plane_cross":
        origin = (0.0, 0.0, 0.0)
        nodes1 = np.linspace(origin, pnt_plus_x, num_elems_m+1)
        nodes2 = np.linspace(origin, pnt_minus_x, num_elems_m+1)
        nodes3 = np.linspace(origin, pnt_plus_y, num_elems_m+1)
        nodes4 = np.linspace(origin, pnt_minus_y, num_elems_m+1)
        nodes = [nodes1, nodes2, nodes3, nodes4]

    elif name == "t":
        origin = (0.0, -n, 0.0)
        nodes1 = np.linspace(origin, pnt_plus_x, num_elems_m+1)
        nodes2 = np.linspace(origin, pnt_minus_x, num_elems_m+1)
        nodes3 = np.linspace(origin, pnt_plus_y, num_elems_m+1)
        nodes = [nodes1, nodes2, nodes3]

    elif name == "t_line":
        origin = (0.0, -n, -n)
        nodes1 = np.linspace(origin, pnt_plus_x, num_elems_m+1)
        nodes2 = np.linspace(origin, pnt_minus_x, num_elems_m+1)
        nodes3 = np.linspace(origin, pnt_plus_y, num_elems_m+1)
        nodes4 = np.linspace(origin, pnt_plus_z, num_elems_m+1)
        nodes = [nodes1, nodes2, nodes3]
        nodes = [nodes1, nodes2, nodes3, nodes4]

    elements = [ref_elements.copy() for i in range(len(nodes))]
    elements, cell_types, nodes = process_raw_mesh(nodes, elements)

    return elements, cell_types, nodes
