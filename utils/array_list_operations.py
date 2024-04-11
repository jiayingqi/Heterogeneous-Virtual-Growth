import itertools
import numpy as np
from scipy.spatial import cKDTree


def compare_matrices(array1, array2, precision=12):
    """
    Find the list "args" such that array1[args] == array2.
    Input:
        array1, array2: Two matrices containing same vectors but with
                        different orders.
        precision: Precision of numbers in array1 and array2. In the parallel
                   computation, small numerical errors might be added to array1
                   and array2, and thus it is necessary to eliminate those
                   errors with the operation of round(precision). This value
                   should be small enougth to eliminate the numerical error.
                   However, information of arrays migth be lost if it is
                   extremely small. For example, array1 = [6.01, 6.02] is the
                   serial data and array2 = [6.010001, 6.020001] is the parlell
                   data. array2.round(8) = [6.010001, 6.020001] can still never
                   matches array1.round(8) = [6.01, 6.02]. array2.round(1)
                   = [6.0, 6.0] and array2 = [6.0, 6.0] will lose the matching
                   information.
    Output:
        args: Relationships of arguments.
    """
    kd_tree = cKDTree(array1.round(precision))
    index = kd_tree.query(array2.round(precision))[1]
    return index


def remove_repeated_list(in_list):
    """Remove repeated lists in a given list."""
    # Convert the inner lists to tuples such that they are hashable
    temp = set(map(tuple, in_list))
    # Convert it back to a list
    out_list = list(map(list, temp))
    return out_list


def find_indices(parent_array, array):
    """
    Find the indices of elements of array in parent_array.
    Requirement: array is the proper subset of parent_array.
    Example:
        a = np.array([1, 3, 7, 5, 4])
        b = np.array([3, 4, 5])
        find_indices(a, b)
        return: [1 4 3]
    """
    sorter = np.argsort(parent_array)
    return sorter[np.searchsorted(parent_array, array, sorter=sorter)]


def find_max_length_nested_list(list):
    """
    Find the maximum length of sublists in a given list.
    Example:
        list = [["A"], ["A", "B", "C"], ["A", "B"]]
        find_max_length_nested_list(list)
        return: 3
    """
    return max(map(len, list))


def nested_list_unique(inp_list: list, return_type_list=False):
    """
    Keep unique items for each sublist in the given list.
    Inputs:
        inp_list: A nested list.
        return_type_list: Bool. Return a list of lists if true and return a
                          a list of sets if false.
    Output:
        out_list: A list of lists or a list of sets.
    Example:
        inp = [[2, 2], [3, 4, 4]]
        nested_list_unique(inp, return_type_list=True)
        return: [[2], [3, 4]]
        nested_list_unique(inp, return_type_list=False)
        return: [{2}, {3, 4}]
    """
    out_list = list(map(set, inp_list))
    if return_type_list:
        out_list = list(map(list, out_list))
    return out_list


def fill_list_in_array(inp_list: list, value=0):
    """
    Fill a list of ragged lists in an array and fill with the specified value.
    Inputs:
        inp_list: A nested list.
        value: Value to fill in blanks in the array.
    Output:
        Filled numpy.ndarray.
    Example:
        inp = [[1, 2], [3, 4, 5], [3, 1]]
        fill_list_in_array(inp, value=-1)
        return:
            [[ 1  2 -1]
             [ 3  4  5]
             [ 3  1 -1]]
    """
    return np.array(list(itertools.zip_longest(*inp_list, fillvalue=value))).T


def rowwise_intersection(arr1, arr2, false_value=-1):
    """
    Find row-wise intersection of two 2D arrays.
    Inputs:
        arr1, arr2: Two input 2D arrays.
        false_value: The value to fill in blanks if the intersection is empty.
    Output:
        out_array: Intersection array which has the same size as arr1.
    Example:
        arr1 = np.array([
            [1, 2, 5],
            [1, 2, 7],
        ])
        arr2 = np.array([
            [1, -3],
            [2, 4],
        ])
        rowwise_intersection(arr1, arr2)
        return:
            [[ 1 -1 -1]
             [-1  2 -1]]
    """
    bool_table = (arr1[:, :, None] == arr2[:, None, :]).any(-1)
    out_array = arr1.copy()
    out_array[~bool_table] = false_value
    return out_array


def rowwise_intersection_multiple_arrays(arrays: list, false_value=-1):
    """
    Find row-wise intersection of multiple 2D arrays.
    Inputs:
        arrays: A list of input 2D arrays.
        false_value: The value to fill in blanks if the intersection is empty.
    Output:
        out_array: Intersection array.
    Example:
        arrays = [
            np.array([
                [1, 2, 5],
                [1, 2, 7],
            ]),
            np.array([
                [1, -3],
                [2, 7],
            ]),
            np.array([
                [1, 2, 5, 4],
                [1, 2, 7, 6],
            ]),
        ]
        rowwise_intersection_multiple_arrays(arrays, false_value=-5)
        return:
            [[ 1 -5 -5]
             [-5  2  7]]
    """
    out_array = arrays[0].copy()
    for n in range(1, len(arrays)):
        out_array = rowwise_intersection(out_array, arrays[n],
                                         false_value=false_value)
    return out_array
