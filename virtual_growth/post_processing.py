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

from collections import Counter
import numpy as np
import pyvista

from utils.remove_repeated_nodes import remove_repeated_nodes


def compute_final_frequency(block_count, num_elem, aug_candidates, candidates):
    """This function is used to compute frequency distribution of designs."""
    reduced_list = aug_candidates.copy()
    for n, item in enumerate(reduced_list):
        reduced_list[n] = item[:item.index(" ")]
    counter = dict(Counter(reduced_list))

    frequency = np.zeros((num_elem, len(candidates)))
    k = 0
    for n, item in enumerate(candidates):
        frequency[:, n] = np.sum(block_count[:, k:k+counter[item]], axis=1)
        k += counter[item]
    frequency /= np.sum(frequency, axis=1).reshape(-1, 1)
    return frequency


def plot_microstructure_2d(full_mesh, block_lib, block_nodes, names,
                           block_size, solid=[], void=[], color="#96ADFC",
                           save_path="", fig_name="microstructure.jpg"):
    node_count = 0
    k = 0
    element_count = np.zeros(full_mesh.size, dtype=int)
    element_list, node_list = [], []

    for y in range(full_mesh.shape[0]):
        for x in range(full_mesh.shape[1]):
            block = full_mesh[y][x]
            parent = block[:block.index(" ")]
            index = names.tolist().index(block)

            elements = block_lib[parent]["elements"].copy()
            elements[:, 1:] += node_count
            nodes = block_nodes[index].copy()

            nodes[:, 0] += block_size * x
            nodes[:, 1] -= block_size * y  # Note here it should minus
            node_count += nodes.shape[0]

            element_list.extend(elements.tolist())
            node_list.extend(nodes.tolist())

            element_count[k] = elements.shape[0]
            k += 1

    elements = np.array(element_list)
    nodes = np.array(node_list)
    nodes, elements = remove_repeated_nodes(nodes, elements[:, 1:], precision=6)
    elements = np.hstack((
        np.full((elements.shape[0], 1), elements.shape[1]), elements,
    )).astype(int)
    cell_types = np.full(elements.shape[0], 9, dtype=int)

    pyvista.OFF_SCREEN = True
    pyvista.set_plot_theme("document")
    pyvista.start_xvfb()
    figsize = 2000
    plotter = pyvista.Plotter(window_size=[figsize, figsize])
    grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)
    plotter.add_mesh(grid, color=color, lighting=True,
                     show_edges=False, show_scalar_bar=False)

    plotter.background_color = "white"
    plotter.view_xy()
    plotter.show_axes()

    plotter.screenshot(save_path+fig_name, window_size=[figsize, figsize])
    plotter.close()

    return elements, cell_types, nodes, element_count


def plot_microstructure_3d(full_mesh, block_lib, block_nodes, names,
                           block_size, solid=[], void=[], color="#96ADFC",
                           save_path="", fig_name="microstructure.jpg"):
    node_count = 0
    k = 0
    element_count = np.zeros(full_mesh.size, dtype=int)
    element_list, node_list = [], []

    for z in range(full_mesh.shape[0]):
        for y in range(full_mesh.shape[1]):
            for x in range(full_mesh.shape[2]):
                block = full_mesh[z][y][x]
                parent = block[:block.index(" ")]
                index = names.tolist().index(block)

                elements = block_lib[parent]["elements"].copy()
                elements[:, 1:] += node_count
                nodes = block_nodes[index].copy()

                nodes[:, 0] -= block_size * x  # Note here it should minus
                nodes[:, 1] += block_size * y
                nodes[:, 2] += block_size * z
                node_count += nodes.shape[0]

                element_list.extend(elements.tolist())
                node_list.extend(nodes.tolist())

                element_count[k] = elements.shape[0]
                k += 1

    elements = np.array(element_list)
    nodes = np.array(node_list)
    nodes, elements = remove_repeated_nodes(nodes, elements[:, 1:], precision=6)
    elements = np.hstack((
        np.full((elements.shape[0], 1), elements.shape[1]), elements,
    )).astype(int)
    cell_types = np.full(elements.shape[0], 12, dtype=int)

    pyvista.OFF_SCREEN = True
    pyvista.set_plot_theme("document")
    pyvista.start_xvfb()
    figsize = 2000
    plotter = pyvista.Plotter(window_size=[figsize, figsize])
    grid = pyvista.UnstructuredGrid(elements, cell_types, nodes)
    plotter.add_mesh(grid, color=color, lighting=True,
                     show_edges=False, show_scalar_bar=False)

    plotter.background_color = "white"
    plotter.show_axes()

    plotter.screenshot(save_path+fig_name, window_size=[figsize, figsize])
    plotter.close()

    return elements, cell_types, nodes, element_count


def plot_microstructure_gif(fill_sequence, elements, cell_types, nodes,
                            element_count, dim=3, color="#96ADFC",
                            save_path="", gif_name="microstructure.gif"):
    pyvista.OFF_SCREEN = True
    pyvista.set_plot_theme("document")
    pyvista.start_xvfb()
    figsize = 2000
    plotter = pyvista.Plotter(window_size=[figsize, figsize])

    fill_sequence = fill_sequence.astype(int)
    start = np.sum(element_count[:fill_sequence[0]]).astype(int)
    end = np.sum(element_count[:fill_sequence[0]+1]).astype(int)
    elem_list = np.arange(start, end).tolist()
    grid = pyvista.UnstructuredGrid(
        elements[elem_list], cell_types[elem_list], nodes)
    plotter.add_mesh(grid, color=color, lighting=True, show_edges=False,
                     show_scalar_bar=False, name="mesh_actor")
    plotter.background_color = "white"
    if dim == 2:
        plotter.view_xy()
    plotter.open_gif(save_path+gif_name, framerate=24)
    plotter.write_frame()

    for n in fill_sequence[1:]:
        start = np.sum(element_count[:n]).astype(int)
        end = np.sum(element_count[:n+1]).astype(int)
        elem_list.extend(np.arange(start, end).tolist())
        grid = pyvista.UnstructuredGrid(
            elements[elem_list], cell_types[elem_list], nodes)
        plotter.add_mesh(grid, color=color, lighting=True,
                         show_edges=False, show_scalar_bar=False,
                         name="mesh_actor")
        plotter.write_frame()
    plotter.close()
