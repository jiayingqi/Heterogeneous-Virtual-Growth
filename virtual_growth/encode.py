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

from utils.array_list_operations import find_indices, find_max_length_nested_list


def encode(names, rotation_table, inp_rules, special_rules, path_name=""):
    """Encode strings to numbers."""
    # Encode the rotation table
    num_rows = len(rotation_table.keys())
    num_cols = find_max_length_nested_list(rotation_table.values())
    encoded_rotation_table = np.zeros((num_rows, num_cols), dtype=int)
    for key, value in rotation_table.items():
        row = np.argwhere(names == key)[0, 0]
        indices = find_indices(names, value)
        encoded_rotation_table[row] = indices
    np.save(path_name+"rotation_table_encoded.npy", encoded_rotation_table)

    # Encode the adjacency rule
    rules = {}
    for (key, val) in inp_rules:
        if key not in rules.keys():
            rules[key] = [val]
        else:
            rules[key].append(val)
    num_rows = len(rules.keys())
    num_cols = find_max_length_nested_list(rules.values())
    encoded_rules = np.zeros((num_rows, num_cols), dtype=int)
    for key, value in rules.items():
        row = np.argwhere(names == key)[0, 0]
        indices = find_indices(names, value)
        indices = np.hstack((indices, np.ones(num_cols-indices.size, dtype=int)*-1))
        encoded_rules[row] = indices
    np.save(path_name+"pair_rules_encoded.npy", encoded_rules)

    # Encode special rules
    num_rows = len(special_rules.keys())
    num_cols = find_max_length_nested_list(special_rules.values())
    encoded_special_rules = np.zeros((num_rows, num_cols), dtype=int)
    for n, value in enumerate(special_rules.values()):
        encoded_special_rules[n] = find_indices(names, value)
    np.save(path_name+"special_rules_encoded.npy", encoded_special_rules)
