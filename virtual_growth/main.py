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

import os
import time
import datetime
import pytz
from tqdm import tqdm

import numpy as np

from virtual_growth.adjacency_rules import (
    augment_candidates, find_admissible_blocks_2d, find_admissible_blocks_3d)
from virtual_growth.post_processing import (
    compute_final_frequency, plot_microstructure_2d, plot_microstructure_3d,
    plot_microstructure_gif)
from utils.array_list_operations import find_indices


def main(mesh_size, elem_size, candidates, frequency_hints,
         d=1, m=6, n=0, periodic=False, num_tries=1, print_frequency=True,
         make_figure=True, make_gif=True, color="#96ADFC", data_path="",
         save_path="", fig_name="microstructure.jpg",
         gif_name="microstructure.mp4", save_mesh=False, save_mesh_path="",
         save_mesh_name="symbolic_graph.npy"):
    """
    The main function for generating microstructures.
    Inputs:
        mesh_size: A tuple containing numbers of elements in the z-, y- and
                   x-directions (y- and x-directions for 2D) in the mesh.
        elem_size: A tuple containing numbers of cells in the z-, y- and
                   x-directions (y- and x-directions for 2D) in each element.
        candidates: A list containing names of candidate blocks.
        frequency_hints: A 2D array containing frequency hints of candidate
                         blocks. Each row repents frequency hints of one
                         finite element.
        d: One half length of each side of the kernal cube (3D) or one half
           thickness of the arm (2D).
        m: Length of the arm.
        n: Offset of the center (only for 2D).
        periodic: Activate the periodic constaint or not.
        num_tries: Allowable attemps to generate the microstructure.
        print_frequency: Print the frequency distribution or not.
        make_figure: Make the figure of the microstructure or not.
        make_gif: Make the gif of the history of microstructure generation or not.
        data_path: The path for storing the information of blocks and rules.
        save_path: The path for storing the microstructure.
        fig_name: The image name for storing the microstructure.
        gif_name: The gif name for storing the history of microstructure generation.
        save_mesh: Save the symbolic mesh or not.
        save_mesh_path: The path for storing the symbolic mesh.
        save_mesh_name: The name for storing the symbolic mesh.
    """

    # Pre-check
    # -------------------------------------------------------------------------
    computation_start_time = time.time()
    start_time_info = "Program start time: "+datetime.datetime.now(
        pytz.timezone("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
    print(start_time_info)

    if frequency_hints.shape[1] != len(candidates):
        raise ValueError(
            "Candidate blocks and frequency hints are incompatible.")

    if np.count_nonzero(
        np.isclose(np.sum(frequency_hints, axis=1), 1.0)
    ) != frequency_hints.shape[0]:
        raise ValueError("Sum of frequency hints is inequal to 1.")

    if len(mesh_size) != len(elem_size):
        raise ValueError("Dimensions of the mesh and element are incompatible.")

    num_elems = np.prod(mesh_size)  # Number of elements in the mesh
    num_cells_elem = np.prod(elem_size)  # Number of cells in each element
    num_cells = num_elems * num_cells_elem  # Number of all cells in the mesh

    if frequency_hints.shape[0] != num_elems:
        raise ValueError("The number of elements and frequency hints"
                         "are incompatible.")

    if (make_figure or make_gif) and save_path != "" and not os.path.exists(save_path):
        os.makedirs(save_path)

    if save_mesh and save_mesh_path != "" and not os.path.exists(save_mesh_path):
        os.makedirs(save_mesh_path)

    # Prepare for the virtual growth
    # -------------------------------------------------------------------------
    # Compute numbers of all cells in each direction
    dim = len(mesh_size)
    if dim == 2:
        num_cells_y = mesh_size[0] * elem_size[0]
        num_cells_x = mesh_size[1] * elem_size[1]
    elif dim == 3:
        num_cells_z = mesh_size[0] * elem_size[0]
        num_cells_y = mesh_size[1] * elem_size[1]
        num_cells_x = mesh_size[2] * elem_size[2]

    # Compute numbers of all elements in each direction
    all_cells = np.arange(num_cells)
    if dim == 2:
        y_cell, x_cell = np.divmod(all_cells, num_cells_x)
        y_elem, _ = np.divmod(y_cell, elem_size[0])
        x_elem, _ = np.divmod(x_cell, elem_size[1])
        all_elems = y_elem*mesh_size[1] + x_elem
    elif dim == 3:
        z_cell, temp = np.divmod(all_cells, num_cells_x*num_cells_y)
        y_cell, x_cell = np.divmod(temp, num_cells_x)
        z_elem, _ = np.divmod(z_cell, elem_size[0])
        y_elem, _ = np.divmod(y_cell, elem_size[1])
        x_elem, _ = np.divmod(x_cell, elem_size[2])
        all_elems = z_elem*mesh_size[1]*mesh_size[2] + y_elem*mesh_size[2] + x_elem

    # Import data
    names = np.load(data_path + "unique_names.npy")
    rules = np.load(data_path + "pair_rules_encoded.npy")
    rotation_table = np.load(data_path + "rotation_table_encoded.npy")
    special_rules = np.load(data_path + "special_rules_encoded.npy")

    aug_candidates, aug_frequency_hints = augment_candidates(
        candidates, frequency_hints, names)
    aug_candidates_encoded = find_indices(names, aug_candidates)
    aug_candidates_encoded_ref = np.hstack((aug_candidates_encoded, -1))

    # Add cases of "unfilled" and "wall" to rules
    if aug_candidates_encoded.size > rules.shape[1]:
        rules = np.hstack((rules, np.ones(
            (rules.shape[0], aug_candidates_encoded.size-rules.shape[1]), dtype=int) * -1
        ))
    elif aug_candidates_encoded.size < rules.shape[1]:
        aug_candidates_encoded = np.hstack((
            aug_candidates_encoded,
            np.ones(rules.shape[1]-aug_candidates_encoded.size, dtype=int) * -1,
        ))
    rules = np.vstack((rules, aug_candidates_encoded, aug_candidates_encoded))

    # Add cases of "unfilled" and "wall" to the rotational table
    rotation_table = np.vstack((
        rotation_table,
        np.ones(rotation_table.shape[1], dtype=int) * -2,  # Wall
        np.ones(rotation_table.shape[1], dtype=int) * -1,  # Unfilled
    ))

    # Start the virtual growth of materials
    # -------------------------------------------------------------------------
    admissible = False
    for n_try in range(num_tries):
        try:
            # Initialize the mesh (a symbolic graph, -1 := unfilled)
            if dim == 2:
                full_mesh = np.full((num_cells_y, num_cells_x), -1,
                                    dtype=object)
            elif dim == 3:
                full_mesh = np.full((num_cells_z, num_cells_y, num_cells_x),
                                    -1, dtype=object)

            # Add boundaries (-2 := wall)
            if dim == 2:
                aug_full_mesh = np.full((num_cells_y+2, num_cells_x+2),
                                        -2, dtype=object)
            elif dim == 3:
                aug_full_mesh = np.full((num_cells_z+2, num_cells_y+2, num_cells_x+2),
                                        -2, dtype=object)
            block_count = np.zeros((num_elems, len(aug_candidates)))
            fill_sequence = np.zeros(num_cells)

            for n_iter in tqdm(range(num_cells)):
                # Update the information of mesh and probabilities
                # -------------------------------------------------------------
                # Update the augmented mesh
                if dim == 2:
                    aug_full_mesh[1:-1, 1:-1] = full_mesh
                elif dim == 3:
                    aug_full_mesh[1:-1, 1:-1, 1:-1] = full_mesh

                # Apply the periodic constraint
                if periodic:
                    if dim == 2:
                        aug_full_mesh[1:-1, 0] = full_mesh[:, -1]
                        aug_full_mesh[1:-1, -1] = full_mesh[:, 0]
                        aug_full_mesh[0, 1:-1] = full_mesh[-1, :]
                        aug_full_mesh[-1, 1:-1] = full_mesh[0, :]
                    elif dim == 3:
                        aug_full_mesh[1:-1, 1:-1, 0] = full_mesh[:, :, -1]
                        aug_full_mesh[1:-1, 1:-1, -1] = full_mesh[:, :, 0]
                        aug_full_mesh[1:-1, -1, 1:-1] = full_mesh[:, 0, :]
                        aug_full_mesh[1:-1, 0, 1:-1] = full_mesh[:, -1, :]
                        aug_full_mesh[-1, 1:-1, 1:-1] = full_mesh[0, :, :]
                        aug_full_mesh[0, 1:-1, 1:-1] = full_mesh[-1, :, :]

                # Find neighbor blocks for all cells
                if dim == 2:
                    left_blocks = aug_full_mesh[1:-1, :-2].flatten()
                    right_blocks = aug_full_mesh[1:-1, 2:].flatten()
                    top_blocks = aug_full_mesh[:-2, 1:-1].flatten()
                    bottom_blocks = aug_full_mesh[2:, 1:-1].flatten()
                elif dim == 3:
                    left_blocks = aug_full_mesh[1:-1, 1:-1, :-2].flatten()
                    right_blocks = aug_full_mesh[1:-1, 1:-1, 2:].flatten()
                    front_blocks = aug_full_mesh[1:-1, 2:, 1:-1].flatten()
                    back_blocks = aug_full_mesh[1:-1, :-2, 1:-1].flatten()
                    top_blocks = aug_full_mesh[2:, 1:-1, 1:-1].flatten()
                    bottom_blocks = aug_full_mesh[:-2, 1:-1, 1:-1].flatten()

                # Update probabilities of candidate blocks
                # (x0+(n-x)*p)/n=p0 => p=(p0*n-x0)/(n-x)
                probs = (aug_frequency_hints*num_cells_elem - block_count) \
                    / (num_cells_elem - np.sum(block_count, axis=1).reshape(-1, 1) + 1e-6)
                probs[probs <= 0] = 1e-6  # Laplace smoothing
                probs /= np.sum(probs, axis=1).reshape(-1, 1)  # Normalization
                probs = np.hstack((probs, np.zeros((num_elems, 1))))

                # Determine the target cell
                # -------------------------------------------------------------
                # Find cells to check
                unfilled_cells = np.argwhere(full_mesh.flatten() == -1)[:, 0]
                if dim == 2:
                    remote_cells = np.argwhere(
                        ((left_blocks == -1) | ((left_blocks == -2)))
                        & ((right_blocks == -1) | (right_blocks == -2))
                        & ((top_blocks == -1) | (top_blocks == -2))
                        & ((bottom_blocks == -1) | (bottom_blocks == -2))
                    )[:, 0]
                elif dim == 3:
                    remote_cells = np.argwhere(
                        ((left_blocks == -1) | ((left_blocks == -2)))
                        & ((right_blocks == -1) | (right_blocks == -2))
                        & ((front_blocks == -1) | (front_blocks == -2))
                        & ((back_blocks == -1) | (back_blocks == -2))
                        & ((top_blocks == -1) | (top_blocks == -2))
                        & ((bottom_blocks == -1) | (bottom_blocks == -2))
                    )[:, 0]
                if n_iter == 0:
                    checked_cells = unfilled_cells.copy()
                else:
                    checked_cells = np.setdiff1d(unfilled_cells, remote_cells)
                entropies = np.zeros(checked_cells.size)
                checked_elems = all_elems[checked_cells]

                # Find admissible_blocks
                if dim == 2:
                    admissible_blocks = find_admissible_blocks_2d(
                        rules, rotation_table, aug_candidates_encoded, special_rules,
                        left_blocks[checked_cells], right_blocks[checked_cells],
                        top_blocks[checked_cells], bottom_blocks[checked_cells])
                elif dim == 3:
                    admissible_blocks = find_admissible_blocks_3d(
                        rules, rotation_table, aug_candidates_encoded, special_rules,
                        left_blocks[checked_cells], right_blocks[checked_cells],
                        front_blocks[checked_cells], back_blocks[checked_cells],
                        top_blocks[checked_cells], bottom_blocks[checked_cells])

                index = find_indices(aug_candidates_encoded_ref, admissible_blocks.flatten())
                index = index.reshape(admissible_blocks.shape)

                admissible_probs = np.take_along_axis(probs[checked_elems], index, 1)
                admissible_probs /= np.sum(admissible_probs, 1).reshape(-1, 1)  # Normalization

                # Compute the entropies
                temp = admissible_probs.copy()
                temp[np.isclose(temp, 0)] = 1
                entropies = - np.einsum("ij,ij->i", admissible_probs,
                                        np.log10(temp))

                # Determine the target cell to be filled with a block
                temp = np.argwhere(entropies == min(entropies))[:, 0]
                target_cell = np.random.choice(temp)
                target_cell = checked_cells[target_cell]
                fill_sequence[n_iter] = target_cell
                target_element = all_elems[target_cell]

                # Determine the target block
                # -------------------------------------------------------------
                # Find admissible_blocks
                index = np.argwhere(checked_cells == target_cell)[0, 0]
                target_admissible_blocks = admissible_blocks[index]
                target_admissible_probs = admissible_probs[index]

                # Determine the block
                temp = np.flip(np.cumsum(target_admissible_probs))
                index = target_admissible_probs.size - np.argmax(temp < np.random.rand(1))
                index = 0 if index == target_admissible_probs.size else index
                target_block = target_admissible_blocks[index]

                # Fill in the target block to the target cell
                if dim == 2:
                    full_mesh[y_cell[target_cell], x_cell[target_cell]] = target_block
                elif dim == 3:
                    full_mesh[z_cell[target_cell], y_cell[target_cell], x_cell[target_cell]] = target_block
                index = np.argwhere(aug_candidates_encoded_ref == target_block)[0, 0]
                block_count[target_element, index] += 1

            admissible = True
            break
        except ValueError:
            print(f"Try {n_try+1}: No admissible blocks.")

    computation_end_time = time.time()
    computation_time = computation_end_time - computation_start_time
    print(f"Computational time: {computation_time:.3g} second(s) "
          f"/{computation_time/3600:.3g} hour(s)")

    if not admissible:
        return 0

    # Post processing
    # -------------------------------------------------------------------------
    process_start_time = time.time()

    # Compute frequency distribution
    final_frequency = compute_final_frequency(
        block_count, num_elems, aug_candidates, candidates)

    if print_frequency:
        print("Input frequency hints of candidate blocks:")
        with np.printoptions(precision=4):
            print(frequency_hints)
        print("Final frequency distribution of candidate blocks:")
        with np.printoptions(precision=4):
            print(final_frequency)

    frequency_error = np.linalg.norm(
        final_frequency-frequency_hints, 2) / num_elems
    print(f"2-norm error of frequency distribution: {frequency_error:.3g}")

    # Decode the mesh
    index = full_mesh.flatten().astype(int)
    full_mesh = names[index].reshape(full_mesh.shape)
    if save_mesh:
        np.save(save_mesh_path + save_mesh_name, full_mesh)

    # Plot generated microstructures
    if make_figure:
        block_nodes = np.load(data_path + "unique_block_nodes.npy",
                              allow_pickle=True)
        if dim == 2:
            from virtual_growth.block_library_2d import block_library_elem
            block_lib = block_library_elem(d, m, n)
            block_size = 2 * m
            elements, cell_types, nodes, element_count = plot_microstructure_2d(
                full_mesh, block_lib, block_nodes, names, block_size,
                color=color, save_path=save_path, fig_name=fig_name)
        elif dim == 3:
            from virtual_growth.block_library_3d import block_library_elem
            block_lib = block_library_elem(d, m)
            block_size = 2 * m
            elements, cell_types, nodes, element_count = plot_microstructure_3d(
                full_mesh, block_lib, block_nodes, names, block_size,
                color=color, save_path=save_path, fig_name=fig_name)

    if make_gif:
        plot_microstructure_gif(
            fill_sequence, elements, cell_types, nodes, element_count,
            dim=dim, color=color, save_path=save_path, gif_name=gif_name)

    process_end_time = time.time()
    process_time = process_end_time - process_start_time
    print(f"Post-processsing time: {process_time:.3g} second(s) "
          f"/{process_time/3600:.3g} hour(s)\n", flush=True)
