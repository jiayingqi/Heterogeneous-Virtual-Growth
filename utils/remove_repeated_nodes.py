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
