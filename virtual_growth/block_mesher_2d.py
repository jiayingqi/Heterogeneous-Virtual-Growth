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
from utils.create_rectangle_mesh import create_rectangle_mesh


def block_mesher(name="cross", d=1, m=6, n=0, num_elems_d=4, num_elems_m=10):
    """
    Generate the fem mesh with solid elements for the input 2d block.
    Inputs:
        d: One half thickness of the arm.
        m: Length of the arm.
        n: Offset of the center.
        num_elems_d: Number of elements in the d-direction.
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
        nodes = np.hstack((nodes, np.zeros((nodes.shape[0], 1))))
        elements = np.hstack((np.ones((elements.shape[0], 1))*elements.shape[1],
                             elements)).astype(int)
        cell_types = np.ones(elements.shape[0], dtype=int) * 9
        return elements, cell_types, nodes

    def coord_transformation(r, s, vx, vy):
        """
        Compute coordinates in a rectangular field after tranformation.
        Inputs:
            r, s: Coordinates in the natural coordinate system.
            vx, vy: Coordinates of four vertexes of the transformed field.
        Outputs:
            coord_x, coord_y: Coordinates in the transformed field.
        """
        r_vec = np.array([-1, 1, 1, -1])
        s_vec = np.array([-1, -1, 1, 1])
        temp1 = r.reshape(-1, 1) * r_vec
        temp2 = s.reshape(-1, 1) * s_vec
        N = (1 + temp1) * (1 + temp2) / 4
        coord_x = np.sum(N * vx, axis=1)
        coord_y = np.sum(N * vy, axis=1)
        return coord_x, coord_y

    if name == "corner":
        nodes1, elements1 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_m)
        nodes1 -= 1
        vx = np.array([-(m*n+d*(m+n))/m, d-n+d*n/m, d, -d])
        vy = np.array([-(m*n+d*(m+n))/m, d-n+d*n/m, m, m])
        nodes1[:, 0], nodes1[:, 1] = coord_transformation(
            nodes1[:, 0], nodes1[:, 1], vx, vy)
        nodes2, elements2 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes2 -= 1
        vx = np.array([-(m*n+d*(m+n))/m, m, m, d-n+d*n/m])
        vy = np.array([-(m*n+d*(m+n))/m, -d, d, d-n+d*n/m])
        nodes2[:, 0], nodes2[:, 1] = coord_transformation(
            nodes2[:, 0], nodes2[:, 1], vx, vy)
        elements, cell_types, nodes = process_raw_mesh(
            [nodes1, nodes2],
            [elements1, elements2])
    elif name == "cross":
        nodes1, elements1 = create_rectangle_mesh(2*d, 2*d, num_elems_d, num_elems_d)  # Middle
        nodes1 -= d
        nodes2, elements2 = create_rectangle_mesh(m-d, 2*d, num_elems_m, num_elems_d)  # Left
        nodes2[:, 0] -= m
        nodes2[:, 1] -= d
        nodes3 = nodes2.copy()  # Right
        nodes3[:, 0] += m + d
        elements3 = elements2.copy()
        nodes4, elements4 = create_rectangle_mesh(2*d, m-d, num_elems_d, num_elems_m)  # Top
        nodes4[:, 0] -= d
        nodes4[:, 1] += d
        nodes5 = nodes4.copy()  # Bottom
        nodes5[:, 1] -= m + d
        elements5 = elements4.copy()
        elements, cell_types, nodes = process_raw_mesh(
            [nodes1, nodes2, nodes3, nodes4, nodes5],
            [elements1, elements2, elements3, elements4, elements5])
    elif name == "line":
        nodes1, elements1 = create_rectangle_mesh(2*d, 2*d, num_elems_d, num_elems_d)  # Middle
        nodes1 -= d
        nodes2, elements2 = create_rectangle_mesh(m-d, 2*d, num_elems_m, num_elems_d)  # Left
        nodes2[:, 0] -= m
        nodes2[:, 1] -= d
        nodes3 = nodes2.copy()  # Right
        nodes3[:, 0] += m + d
        elements3 = elements2.copy()
        elements, cell_types, nodes = process_raw_mesh(
            [nodes1, nodes2, nodes3],
            [elements1, elements2, elements3])
    elif name == "skew":
        nodes1, elements1 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes1 -= 1
        vx = np.array([-d, m, m, d])
        vy = np.array([m, -d, d, m])
        nodes1[:, 0], nodes1[:, 1] = coord_transformation(
            nodes1[:, 0], nodes1[:, 1], vx, vy)
        elements, cell_types, nodes = process_raw_mesh([nodes1], [elements1])
    elif name == "t":
        nodes1, elements1 = create_rectangle_mesh(2*d, 2*d, num_elems_d, num_elems_d)
        nodes1 -= d
        nodes1[:, 1] += n
        nodes2, elements2 = create_rectangle_mesh(2*d, m-d+n, num_elems_d, num_elems_m)
        nodes2[:, 0] -= d
        nodes2[:, 1] -= m
        nodes3, elements3 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes3 -= 1
        vx = np.array([-m, -d, -d, -m])
        vy = np.array([-d, -d+n, d+n, d])
        nodes3[:, 0], nodes3[:, 1] = coord_transformation(
            nodes3[:, 0], nodes3[:, 1], vx, vy)
        nodes4, elements4 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes4 -= 1
        vx = np.array([d, m, m, d])
        vy = np.array([-d+n, -d, d, d+n])
        nodes4[:, 0], nodes4[:, 1] = coord_transformation(
            nodes4[:, 0], nodes4[:, 1], vx, vy)
        elements, cell_types, nodes = process_raw_mesh(
            [nodes1, nodes2, nodes3, nodes4],
            [elements1, elements2, elements3, elements4])
    elif name == "v":
        nodes1, elements1 = create_rectangle_mesh(2*d, 2*d, num_elems_d, num_elems_d)
        nodes1[:, 0] -= d
        nodes1[:, -1] -= m
        nodes2, elements2 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes2 -= 1
        vx = np.array([-m, -d, -d, -m])
        vy = np.array([-d, -m, -m+2*d, d])
        nodes2[:, 0], nodes2[:, 1] = coord_transformation(
            nodes2[:, 0], nodes2[:, 1], vx, vy)
        nodes3, elements3 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes3 -= 1
        vx = np.array([d, m, m, d])
        vy = np.array([-m, -d, d, -m+2*d])
        nodes3[:, 0], nodes3[:, 1] = coord_transformation(
            nodes3[:, 0], nodes3[:, 1], vx, vy)
        elements, cell_types, nodes = process_raw_mesh(
            [nodes1, nodes2, nodes3],
            [elements1, elements2, elements3])
    elif name == "x":
        nodes1, elements1 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_d)
        nodes1 -= 1
        vx = np.array([-m, -m+2*d, -m+2*d, -m])
        vy = np.array([-d, -d/2, d/2, d])
        nodes1[:, 0], nodes1[:, 1] = coord_transformation(
            nodes1[:, 0], nodes1[:, 1], vx, vy)
        nodes2, elements2 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes2 -= 1
        vx = np.array([-m, -d, -d/2, -m+2*d])
        vy = np.array([-d, -m, -m+2*d, -d/2])
        nodes2[:, 0], nodes2[:, 1] = coord_transformation(
            nodes2[:, 0], nodes2[:, 1], vx, vy)
        nodes3, elements3 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_d)
        nodes3 -= 1
        vx = np.array([-d, d, d/2, -d/2])
        vy = np.array([-m, -m, -m+2*d, -m+2*d])
        nodes3[:, 0], nodes3[:, 1] = coord_transformation(
            nodes3[:, 0], nodes3[:, 1], vx, vy)
        nodes4, elements4 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes4 -= 1
        vx = np.array([d, m, m-2*d, d/2])
        vy = np.array([-m, -d, -d/2, -m+2*d])
        nodes4[:, 0], nodes4[:, 1] = coord_transformation(
            nodes4[:, 0], nodes4[:, 1], vx, vy)
        nodes5, elements5 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_d)
        nodes5 -= 1
        vx = np.array([m-2*d, m, m, m-2*d])
        vy = np.array([-d/2, -d, d, d/2])
        nodes5[:, 0], nodes5[:, 1] = coord_transformation(
            nodes5[:, 0], nodes5[:, 1], vx, vy)
        nodes6, elements6 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes6 -= 1
        vx = np.array([d/2, m-2*d, m, d])
        vy = np.array([m-2*d, d/2, d, m])
        nodes6[:, 0], nodes6[:, 1] = coord_transformation(
            nodes6[:, 0], nodes6[:, 1], vx, vy)
        nodes7, elements7 = create_rectangle_mesh(2, 2, num_elems_d, num_elems_d)
        nodes7 -= 1
        vx = np.array([-d/2, d/2, d, -d])
        vy = np.array([m-2*d, m-2*d, m, m])
        nodes7[:, 0], nodes7[:, 1] = coord_transformation(
            nodes7[:, 0], nodes7[:, 1], vx, vy)
        nodes8, elements8 = create_rectangle_mesh(2, 2, num_elems_m, num_elems_d)
        nodes8 -= 1
        vx = np.array([-m+2*d, -d/2, -d, -m])
        vy = np.array([d/2, m-2*d, m, d])
        nodes8[:, 0], nodes8[:, 1] = coord_transformation(
            nodes8[:, 0], nodes8[:, 1], vx, vy)
        elements, cell_types, nodes = process_raw_mesh(
            [nodes1, nodes2, nodes3, nodes4, nodes5, nodes6, nodes7, nodes8],
            [elements1, elements2, elements3, elements4, elements5, elements6, elements7, elements8])

    return elements, cell_types, nodes
