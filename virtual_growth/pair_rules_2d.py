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
import pickle
import numpy as np

from virtual_growth.block_library_2d import block_library, block_library_elem
from virtual_growth.block_mesher_2d import block_mesher
from virtual_growth.encode import encode
from utils.coord_transformation import cartesian2polar, polar2cartesian


def rotate_block(inp_block, inp_block_nodes, inp_block_fem_nodes):
    """
    Rotate the block to find the variations. There are 4 possibilities.
    y
    ↑
    O → x
    """
    inp_block = np.array(inp_block)
    out_blocks = np.zeros((4, *inp_block.shape))  # At most 4 variations
    out_block_nodes = np.zeros((4, *inp_block_nodes.shape))
    out_block_fem_nodes = np.zeros((4, *inp_block_fem_nodes.shape))

    def rotate_nodes(coords):
        """Rotate nodes in the x-y plane."""
        coord_x, coord_y, coord_z = coords.T
        rho, phi = cartesian2polar(coord_x, coord_y)
        phi += np.pi/2 * i
        coord_x, coord_y = polar2cartesian(rho, phi)
        return np.vstack((coord_x, coord_y, coord_z)).T

    for i in range(4):  # Rotation in the x-y plane
        out_blocks[i] = np.rot90(inp_block, i, axes=(0, 1))
        out_block_nodes[i] = rotate_nodes(inp_block_nodes)
        out_block_fem_nodes[i] = rotate_nodes(inp_block_fem_nodes)

    return out_blocks.astype(int), out_block_nodes, out_block_fem_nodes


def remove_repeated_blocks(inp_blocks):
    """Remove repeated blocks."""
    _, unique_indices = np.unique(inp_blocks, axis=0, return_index=True)
    return np.sort(unique_indices)


def admissible_pairs(left, right):
    """Check if two blocks are admissible or not."""

    def isdetached(edge1, edge2):
        """Check if two edges are detached or not."""
        return np.sum(edge1 + edge2) == 0

    def isconnected(edge1, edge2):
        """Check if two edges are connected or not."""
        return edge1 @ edge2 > 0

    # Basic requirement: fully connected or detached
    flag1 = isdetached(left[:, -1], right[:, 0]) \
        | isconnected(left[:, -1], right[:, 0])

    # Two corner-shaped blocks cannot be placed face to face (e.g., ┏ ┓)
    index1 = isdetached(left[:, 0], right[:, -1])
    index2 = isconnected(left[:, -1], right[:, 0])
    index3 = isdetached(left[0, :], right[0, :])
    index4 = isconnected(left[-1, :], right[-1, :])
    index5 = isdetached(left[-1, :], right[-1, :])
    index6 = isconnected(left[0, :], right[0, :])
    flag2 = index1 & index2 & ((index3 & index4) | (index5 & index6))

    # Two lines cannot be connected (e.g., — —)
    index1 = isdetached(left[0, :], right[0, :])
    index2 = isdetached(left[-1, :], right[-1, :])
    index3 = isconnected(left[:, 0], right[:, -1])
    index4 = isconnected(left[:, -1], right[:, 0])
    flag3 = index1 & index2 & index3 & index4
    return flag1 & ~flag2 & ~flag3


def generate_rotation_table(blocks, block_names):
    """
    Find the corresponding block after the rotation.
    This function returns a dictionary. The key is the input block and the
    value is a list with length=3 corresponding to the blocks after the
    counterclockwsie rotation for 1, 2, and 3 times in the x-y plane,
    respectively.
    """

    def find_block_name(block, blocks, block_names):
        """Find the block name of a given block."""
        temp = np.sum((blocks - block)**2, axis=(1, 2))
        index = np.argwhere(temp == 0)[0, 0]
        return block_names[index], index

    rotation_table = np.empty((len(block_names), 4), dtype=object)
    for n, block in enumerate(blocks):
        # Rotation in the x-y plane
        for i in range(4):
            rotated_block = np.rot90(block, i, axes=(0, 1))
            rotated_name, _ = find_block_name(rotated_block, blocks, block_names)
            rotation_table[n][i] = rotated_name

    # Convert the np.ndarray to a dictionary
    rotation_dict = {}
    for n in range(len(block_names)):
        rotation_dict[rotation_table[n, 0]] = rotation_table[n, 1:].tolist()

    return rotation_dict


def generate_special_rules(blocks, names):
    """Some blocks cannot be placed at special positions of mesh."""
    # Corner-shaped blocks cannot be placed at the mesh corners.
    special_rules = {  # The keys represent the mesh corners
        "11": [],
        "-11": [],
        "-1-1": [],
        "1-1": [],
    }
    for block, name in zip(blocks, names):
        if np.sum(block[:, 0] + block[-1, :]) == 0:
            special_rules["11"].append(name)
        if np.sum(block[:, -1] + block[-1, :]) == 0:
            special_rules["-11"].append(name)
        if np.sum(block[:, -1] + block[0, :]) == 0:
            special_rules["-1-1"].append(name)
        if np.sum(block[:, 0] + block[0, :]) == 0:
            special_rules["1-1"].append(name)
    return special_rules


def pair_rules_2d(block_names: list, d=1, m=6, n=0, num_elems_d=4, num_elems_m=10,
                  path_name=""):
    """Generate rules for virtual growth."""
    all_unique_blocks, all_unique_block_nodes = [], []
    all_block_names, all_extended_block_names = [], []
    all_unique_block_fem_nodes = []
    all_unique_block_fem_elements, all_unique_block_cell_types = [], []
    block_elem_dict = block_library_elem(d, m, n)

    for block_name in block_names:
        input_block = block_library[block_name]  # Pixel representation
        inp_block_nodes = block_elem_dict[block_name]["nodes"]
        inp_block_fem_elements, inp_block_fem_cell_types, inp_block_fem_nodes = \
            block_mesher(block_name, d, m, n, num_elems_d, num_elems_m)
        out_blocks, out_block_nodes, out_block_fem_nodes = rotate_block(
            input_block, inp_block_nodes, inp_block_fem_nodes)  # Block variations
        unique_indices = remove_repeated_blocks(out_blocks)  # Unique variations

        for count, i in enumerate(unique_indices):
            all_unique_blocks.append(out_blocks[i])
            all_unique_block_nodes.append(out_block_nodes[i])
            all_unique_block_fem_nodes.append(out_block_fem_nodes[i])
            all_unique_block_fem_elements.append(inp_block_fem_elements)
            all_unique_block_cell_types.append(inp_block_fem_cell_types)
            all_block_names.append(block_name)
            all_extended_block_names.append(block_name+f" {count}")

    rules = []
    for (left_block, left_name) in zip(all_unique_blocks, all_extended_block_names):
        for (right_block, right_name) in zip(all_unique_blocks, all_extended_block_names):
            flag = admissible_pairs(left_block, right_block)
            if flag:
                rules.append([left_name, right_name])

    rotation_table = generate_rotation_table(
        all_unique_blocks, all_extended_block_names)
    special_rules = generate_special_rules(
        all_unique_blocks, all_extended_block_names)

    if not os.path.exists(path_name):  # Check if the directory exists
        os.makedirs(path_name)  # Create the directory

    # Save the block information
    all_unique_blocks = np.array(all_unique_blocks)
    all_extended_block_names = np.array(all_extended_block_names)
    np.save(path_name+"unique_blocks.npy", all_unique_blocks)
    np.save(path_name+"unique_names.npy", all_extended_block_names)

    # Save the pair adjacency rules
    with open(path_name+"pair_rules.txt", "w+") as rfile:
        for item in rules:
            rfile.write(item[0] + ", " + item[1] + "\n")

    # Save block+name information
    with open(path_name+"blocks_with_names.txt", "w+") as rfile:
        for name, block in zip(all_extended_block_names, all_unique_blocks):
            rfile.write(name+":\n")
            np.savetxt(rfile, block, fmt="%i")
            rfile.write("\n")

    # Save the rotation table
    with open(path_name+"rotation_table.pkl", "wb") as f:
        pickle.dump(rotation_table, f)

    # Save the unique block nodes
    all_unique_block_nodes = np.array(all_unique_block_nodes, dtype=object)
    np.save(path_name+"unique_block_nodes.npy",
            all_unique_block_nodes)

    # Save the unique block fem nodes
    all_unique_block_fem_nodes = np.array(all_unique_block_fem_nodes, dtype=object)
    np.save(path_name+"unique_block_fem_nodes.npy",
            all_unique_block_fem_nodes)

    # Save the unique block fem elements
    all_unique_block_fem_elements = np.array(all_unique_block_fem_elements, dtype=object)
    np.save(path_name+"unique_block_fem_elements.npy",
            all_unique_block_fem_elements)

    # Save the unique block fem cell_types
    all_unique_block_cell_types = np.array(all_unique_block_cell_types, dtype=object)
    np.save(path_name+"unique_block_fem_cell_types.npy",
            all_unique_block_cell_types)

    # Save the special rules
    with open(path_name+"special_rules.pkl", "wb") as f:
        pickle.dump(special_rules, f)

    encode(all_extended_block_names, rotation_table, rules, special_rules,
           path_name=path_name)
