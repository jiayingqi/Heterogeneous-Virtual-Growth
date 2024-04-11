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

import os
import numpy as np

from virtual_growth.plot_mesh import plot_mesh
from utils.array_list_operations import find_indices
from utils.remove_repeated_nodes import remove_repeated_nodes


def generate_fem_mesh_3d(symbolic_mesh, block_size, data_path, rescale=False,
                         geometry=(1, 1, 1), check_with_pyvista=False,
                         save_mesh=True, mesh_path=""):
    """Generate the fem mesh of the input symbolic graph."""
    # Load the data
    all_fem_elements = np.load(data_path + "unique_block_fem_elements.npy",
                               allow_pickle=True)
    all_fem_nodes = np.load(data_path + "unique_block_fem_nodes.npy",
                            allow_pickle=True)
    names = np.load(data_path + "unique_names.npy")

    # Extract the mesh information
    num_elem_z, num_elem_y, num_elem_x = symbolic_mesh.shape
    indices = find_indices(names, symbolic_mesh.flatten())

    elements = all_fem_elements[indices]
    num_elems = np.array(list(map(len, elements)))
    elements = np.vstack(elements)[:, 1:]

    nodes = all_fem_nodes[indices]
    num_nodes = np.array(list(map(len, nodes)))
    nodes = np.vstack(nodes)

    # Modify nodes
    offset_x = np.tile(np.arange(num_elem_x), num_elem_y*num_elem_z)
    offset_y = np.tile(np.arange(num_elem_y).reshape(-1, 1), (num_elem_z, num_elem_x)).flatten()
    offset_z = np.tile(np.arange(num_elem_z).reshape(-1, 1), (1, num_elem_x*num_elem_y)).flatten()

    offset_x = np.repeat(offset_x, num_nodes) * block_size
    offset_y = np.repeat(offset_y, num_nodes) * block_size
    offset_z = np.repeat(offset_z, num_nodes) * block_size

    nodes[:, 0] -= offset_x
    nodes[:, 1] += offset_y
    nodes[:, 2] += offset_z

    # Shift nodes to the first quadrant
    nodes[:, 0] += block_size * (num_elem_x-1/2)
    nodes[:, 1] += block_size/2
    nodes[:, 2] += block_size/2
    if rescale:
        nodes[:, 0] = nodes[:, 0] / max(nodes[:, 0]) * geometry[0]
        nodes[:, 1] = nodes[:, 1] / max(nodes[:, 1]) * geometry[1]
        nodes[:, 2] = nodes[:, 2] / max(nodes[:, 2]) * geometry[2]

    # Modify elements
    offset = np.hstack((0, np.cumsum(num_nodes)[:-1]))
    offset = np.repeat(offset, num_elems).reshape(-1, 1)
    elements += offset

    nodes, elements = remove_repeated_nodes(nodes, elements, precision=6)
    elements = np.hstack((np.ones((elements.shape[0], 1), dtype=int)*2, elements))

    cell_types = np.ones(elements.shape[0], dtype=int) * 3
    num_elements, num_nodes = elements.shape[0], nodes.shape[0]
    print(f"Number of elements: {num_elements}, Number of nodes: {num_nodes}\n")

    if check_with_pyvista:
        plot_mesh(elements, cell_types, nodes, plot_box=None, box_length=block_size,
                  color="#96ADFC", line_width=10, render_lines_as_tubes=True,
                  plot_nodes=True, point_size=20, fig_path=mesh_path, fig_name="fem_mesh.jpg")

    def find_local_y_directs(nodes, elements):
        """
        Find the y-direction of frame elements in the local coordinate system.
        This subroutine is only suitable for frames where all elements are
        along x-, y-, and z- directions in the global coordinate system. The
        subroutine also requires Iy=Iz in the local coordinate system.
        """
        start_nodes = nodes[elements[:, 0]]
        end_nodes = nodes[elements[:, 1]]
        local_x_directs = end_nodes - start_nodes
        index = np.argwhere(~np.isclose(local_x_directs, 0))[:, -1]
        candidate_directs = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        local_y_directs = candidate_directs[index]
        return local_y_directs

    local_y_directs = find_local_y_directs(nodes, elements[:, 1:])

    mesh = np.array([nodes, elements, cell_types, local_y_directs], dtype=object)
    if save_mesh:
        if mesh_path != "" and not os.path.exists(mesh_path):
            os.makedirs(mesh_path)
        np.savez(mesh_path+"fem_mesh.npz", nodes=nodes, elements=elements[:, 1:],
                 local_y_directs=local_y_directs)

    return mesh
