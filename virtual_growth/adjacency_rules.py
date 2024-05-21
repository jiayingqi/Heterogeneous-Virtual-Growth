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

from utils.array_list_operations import (
    nested_list_unique, fill_list_in_array, rowwise_intersection_multiple_arrays)


def augment_candidates(candidates, frequency_hints, names):
    """
    Consider the block variations (via rotation) and redistribute the frequency
    hints.
    """
    count_dict = dict()
    for item in candidates:
        count_dict[item] = 0

    aug_candidates = []
    for parent_name in candidates:
        for item in names:
            if item[:item.index(" ")] == parent_name:
                count_dict[parent_name] += 1
                aug_candidates.append(item)

    aug_frequency_hints = np.zeros((frequency_hints.shape[0],
                                    sum(count_dict.values())))
    m, n = 0, 0
    for item in count_dict.values():
        aug_frequency_hints[:, m:m+item] = frequency_hints[:, n].reshape(-1, 1) / item
        m += item
        n += 1

    return aug_candidates, aug_frequency_hints


def find_admissible_blocks_2d(rules, rotation_table, aug_candidates, special_rules,
                              left, right, top, bottom):
    """Find admissible blocks of the given cell in the 2D case."""
    # Find admissible blocks of neighboring elements according to rules
    left_blocks = rules[left.astype(int)]

    temp = rotation_table[top.astype(int), 0]
    top_blocks = rules[temp]
    temp = rotation_table[top_blocks.flatten(), 2]
    top_blocks = temp.reshape(top_blocks.shape)

    temp = rotation_table[right.astype(int), 1]
    right_blocks = rules[temp]
    temp = rotation_table[right_blocks.flatten(), 1]
    right_blocks = temp.reshape(right_blocks.shape)

    temp = rotation_table[bottom.astype(int), 2]
    bottom_blocks = rules[temp]
    temp = rotation_table[bottom_blocks.flatten(), 0]
    bottom_blocks = temp.reshape(bottom_blocks.shape)

    # Compute intersection of rules
    temp = rowwise_intersection_multiple_arrays([
        left_blocks, top_blocks, right_blocks, bottom_blocks,
        np.tile(aug_candidates, (left_blocks.shape[0], 1))])
    # Keep unique entries
    admissible_blocks = nested_list_unique(temp.tolist())
    # Fill in a numpy array
    admissible_blocks = fill_list_in_array(admissible_blocks, value=-1)

    def apply_special_rules(block1, block2, input_blocks, excluded_blocks):
        """
        Remove excluded_blocks from input_blocks if certain conditions can be
        met for block1, block2, and block3.
        """
        rows = np.equal(block1, -2) & np.equal(block2, -2)
        temp = input_blocks[rows]
        temp[np.isin(temp, excluded_blocks)] = -1
        input_blocks[rows] = temp.copy()

    # Some blocks cannot be placed at corners of the mesh
    apply_special_rules(top, right, admissible_blocks, special_rules[0])
    apply_special_rules(top, left, admissible_blocks, special_rules[1])
    apply_special_rules(bottom, left, admissible_blocks, special_rules[2])
    apply_special_rules(bottom, right, admissible_blocks, special_rules[3])

    if np.isin(0, np.sum(np.not_equal(admissible_blocks, -1), axis=1)):
        raise ValueError("No admissible blocks.")

    return admissible_blocks


def find_admissible_blocks_3d(rules, rotation_table, aug_candidates, special_rules,
                              left, right, front, back, top, bottom):
    """Find admissible blocks of the given cell in the 3D case."""
    # Find admissible blocks of neighboring elements according to rules
    left_blocks = rules[left.astype(int)]

    temp = rotation_table[back.astype(int), 0]
    back_blocks = rules[temp]
    temp = rotation_table[back_blocks.flatten(), 2]
    back_blocks = temp.reshape(back_blocks.shape)

    temp = rotation_table[right.astype(int), 1]
    right_blocks = rules[temp]
    temp = rotation_table[right_blocks.flatten(), 1]
    right_blocks = temp.reshape(right_blocks.shape)

    temp = rotation_table[front.astype(int), 2]
    front_blocks = rules[temp]
    temp = rotation_table[front_blocks.flatten(), 0]
    front_blocks = temp.reshape(front_blocks.shape)

    temp = rotation_table[top.astype(int), 3]
    temp = rotation_table[temp.astype(int), 0]
    top_blocks = rules[temp]
    temp = rotation_table[top_blocks.flatten(), 2]
    temp = rotation_table[temp.astype(int), 4]
    top_blocks = temp.reshape(top_blocks.shape)

    temp = rotation_table[bottom.astype(int), 4]
    temp = rotation_table[temp.astype(int), 0]
    bottom_blocks = rules[temp]
    temp = rotation_table[bottom_blocks.flatten(), 2]
    temp = rotation_table[temp.astype(int), 3]
    bottom_blocks = temp.reshape(bottom_blocks.shape)

    # Compute intersection of rules
    temp = rowwise_intersection_multiple_arrays([
        left_blocks, back_blocks, right_blocks, front_blocks,
        top_blocks, bottom_blocks,
        np.tile(aug_candidates, (left_blocks.shape[0], 1))])
    # Keep unique entries
    admissible_blocks = nested_list_unique(temp.tolist())
    # Fill in a numpy array
    admissible_blocks = fill_list_in_array(admissible_blocks, value=-1)

    def apply_special_rules(block1, block2, block3, input_blocks, excluded_blocks):
        """
        Remove excluded_blocks from input_blocks if certain conditions can be
        met for block1, block2, and block3.
        """
        rows = np.equal(block1, -2) & np.equal(block2, -2) & np.equal(block3, -2)
        temp = input_blocks[rows]
        temp[np.isin(temp, excluded_blocks)] = -1
        input_blocks[rows] = temp.copy()

    # Some blocks cannot be placed at corners of the mesh
    apply_special_rules(left, front, top, admissible_blocks, special_rules[0])
    apply_special_rules(right, front, top, admissible_blocks, special_rules[1])
    apply_special_rules(right, back, top, admissible_blocks, special_rules[2])
    apply_special_rules(left, back, top, admissible_blocks, special_rules[3])
    apply_special_rules(left, front, bottom, admissible_blocks, special_rules[4])
    apply_special_rules(right, front, bottom, admissible_blocks, special_rules[5])
    apply_special_rules(right, back, bottom, admissible_blocks, special_rules[6])
    apply_special_rules(left, back, bottom, admissible_blocks, special_rules[7])

    if np.isin(0, np.sum(np.not_equal(admissible_blocks, -1), axis=1)):
        raise ValueError("No admissible blocks.")

    return admissible_blocks
