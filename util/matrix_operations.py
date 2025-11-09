"""A module to perform matrix operations.
"""
import numpy as np
from math import log


def matrix_vector_multiply(mat, vec):
    """Multiplies a matrix by a vector.
    Multiplies an m x n matrix by an n x 1 vector (represented
    as a list).
    Args:
        mat (2-D list): Matrix to multiply.
        vec (list): Vector to multiply.
    Returns:
        Product of mat and vec (an m x 1 vector) as a list
    """
    mat_array = np.asarray(mat)
    vec_array = np.asarray(vec)
    prod = np.dot(mat_array, vec_array)
    return prod.tolist()


def add(vec1, vec2):
    """Adds two vectors.
    Adds a length-n list to another length-n list.
    Args:
        vec1 (list): First vector.
        vec2 (list): Second vector.
    Returns:
        Sum of vec1 and vec2.
    """
    vec1_array = np.asarray(vec1)
    vec2_array = np.asarray(vec2)
    assert len(vec1_array) == len(vec2_array)
    return (vec1_array + vec2_array).tolist()


def scalar_multiply(vec, constant):
    """Multiplies a scalar by a vector.
    Multiplies a vector by a scalar.
    Args:
        vec (list): Vector to multiply.
        constant (float): Scalar to multiply.
    Returns:
        Product of vec and constant.
    """
    vec_array = np.asarray(vec)
    return (vec_array * constant).tolist()


def diagonal(mat, diag_index):
    """Returns ith diagonal of matrix, where i is the diag_index.
    Returns the ith diagonal (A_0i, A_1(i+1), ..., A_N(i-1)) of a matrix A,
    where i is the diag_index.
    Args:
        mat (2-D list): Matrix.
        diag_index (int): Index of diagonal to return.
    Returns:
        Diagonal of a matrix.
    """
    mat_array = np.asarray(mat)
    n = len(mat_array)
    indices = np.arange(n)
    row_indices = indices % n
    col_indices = (diag_index + indices) % n
    return mat_array[row_indices, col_indices].tolist()


def rotate(vec, rotation):
    """Rotates vector to the left by rotation.
    Returns the rotated vector (v_i, v_(i+1), ..., v_(i-1)) of a vector v, where i is the rotation.
    Args:
        vec (list): Vector.
        rotation (int): Index.
    Returns:
        Rotated vector.
    """
    vec_array = np.asarray(vec)
    return np.roll(vec_array, -rotation).tolist()


def conjugate_matrix(matrix):
    """Conjugates all entries of matrix.
    Returns the conjugated matrix.
    Args:
        matrix (2-D list): Matrix.
    Returns:
        Conjugated matrix.
    """
    matrix_array = np.asarray(matrix)
    conj_matrix = np.conjugate(matrix_array)
    return conj_matrix.tolist()


def transpose_matrix(matrix):
    """Transposes a matrix.
    Returns the transposed matrix.
    Args:
        matrix (2-D list): Matrix.
    Returns:
        Transposed matrix.
    """
    matrix_array = np.asarray(matrix)
    transpose = np.transpose(matrix_array)
    return transpose.tolist()
