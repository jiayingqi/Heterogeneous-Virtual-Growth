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
