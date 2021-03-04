"""
Functions and operations for tensor algebra

Credit to `Jean Kossaifi <http://jeankossaifi.com/>`_.
"""

import functools
import numpy as np


# TODO: I think it will be better if the following three operations would have the same signature
def khatri_rao(matrices, skip_matrix=None, reverse=False):
    """ Khatri-Rao product of a list of matrices.

    Parameters
    ----------
    matrices : list[np.ndarray]
        List of matrices. Each matrix should have the same number of columns
    skip_matrix : int
        Index of a matrix (from the `matrices`) to be skipped. By default none are skipped
    reverse : bool
        If True, perform khatri-rao product on the list of matrices in the reversed order

    Returns
    -------
    result : np.ndarray
        The result of the Khatri-Rao product is a matrix of shape `()`
    """
    if len(matrices) < 2:
        raise ValueError('khatri_roa product requires a list of at least 2 matrices, '
                         'but {} given.'.format(len(matrices)))
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]
    if reverse:
        matrices = matrices[::-1]
    n_rows, n_cols = matrices[0].shape
    for matrix in matrices[1:]:
        if matrix.shape[1] != n_cols:
            raise ValueError('All matrices must have the same number of columns.')
        n_rows *= matrix.shape[0]
    result = np.zeros((n_rows, n_cols))
    for i in range(n_cols):
        temp = matrices[0][:, i]  # Accumulates the khatri-rao product of the i-th columns
        for matrix in matrices[1:]:
            temp = np.einsum('i,j->ij', temp, matrix[:, i]).ravel()
        result[:, i] = temp  # the i-t column is the kronecker product of all the i-th columns of all matrices:
    return result


def hadamard(matrices, skip_matrix=None, reverse=False):
    """ Hadamard product of a list of matrices.

    Parameters
    ----------
    matrices : list[np.ndarray]
       List of matrices. All matrices should be of the same shape.
    skip_matrix : int
       Index of a matrix (from the `matrices`) to be skipped. By default none are skipped
    reverse : bool
       If True, perform hadamard product on the list of matrices in the reversed order

    Returns
    -------
    result : np.ndarray
        The result of the Hadamard product is a matrix of the same shape as every matrix in `matrices`
    """
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]
    if reverse:
        matrices = matrices[::-1]
    return functools.reduce(np.multiply, matrices)


def kronecker(matrices, skip_matrix=None, reverse=False):
    """ Kronecker product of a list of matrices.

    Parameters
    ----------
    matrices : list[np.ndarray]
        List of matrices.
    skip_matrix : int
        Index of a matrix (from the `matrices`) to be skipped. By default none are skipped
    reverse : bool
        If True, perform Kronecker product on the list of matrices in the reversed order

    Returns
    -------
    result : np.ndarray
        The result of the Kronecker product is a matrix of shape ``(prod(n_rows), prod(n_columns)``
        where ``prod(n_rows) = prod([m.shape[0] for m in matrices])``
        and ``prod(n_columns) = prod([m.shape[1] for m in matrices])``
    """
    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]
    if reverse:
        matrices = matrices[::-1]

    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def unfold(tensor, mode):
    """ Unfolds N-dimensional array into a 2D array.

    Parameters
    ----------
    tensor : np.ndarray
        N-dimensional array to be unfolded
    mode : int
        Specifies a mode along which a `tensor` will be unfolded

    Returns
    -------
    matrix : np.ndarray
        Unfolded version of a `tensor` with a shape ``(tensor.shape[mode], -1)``
    """
    matrix = np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))
    return matrix


def kolda_unfold(tensor, mode):
    """ Unfolds N-dimensional array into a 2D array.

    Parameters
    ----------
    tensor : np.ndarray
        N-dimensional array to be unfolded
    mode : int
        Specifies a mode along which a `tensor` will be unfolded

    Returns
    -------
    matrix : np.ndarray
        Unfolded version of a `tensor` with a shape ``(tensor.shape[mode], -1)``

    Notes
    -----
        Much slower for then ``unfold``.
    """
    matrix = np.transpose(tensor, _kolda_reorder(tensor.ndim, mode)).reshape((tensor.shape[mode], -1))
    return matrix


def fold(matrix, mode, shape):
    """ Fold a 2D array into a N-dimensional array.

    Parameters
    ----------
    matrix : np.ndarray
        Unfolded version of a tensor
    mode : int
        A mode along which original tensor was unfolded into a `matrix`
    shape : tuple
        Shape of the original tensor before it was unfolded

    Returns
    -------
    tensor : np.ndarray
        N-dimensional array of the original shape

    Notes
    -----
        At the moment it reverts unfolding operation (``unfold``). Will be generalised in a future
    """
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    tensor = np.moveaxis(np.reshape(matrix, full_shape), 0, mode)
    return tensor


def kolda_fold(matrix, mode, shape):
    """ Fold a 2D array into a N-dimensional array.

    Parameters
    ----------
    matrix : np.ndarray
        Unfolded version of a tensor
    mode : int
        A mode along which original tensor was unfolded into a `matrix`
    shape : tuple
        Shape of the original tensor before it was unfolded

    Returns
    -------
    tensor : np.ndarray
        N-dimensional array of the original shape

    Notes
    -----
        1) Much slower then ``fold``
        2) At the moment it reverts unfolding operation (``kolda_unfold``). Will be generalised in a future
    """

    unfolded_indices = _kolda_reorder(len(shape), mode)
    original_shape = [shape[i] for i in unfolded_indices]
    matrix = matrix.reshape(original_shape)

    folded_indices = list(range(len(shape)-1, 0, -1))
    folded_indices.insert(mode, 0)
    tensor = np.transpose(matrix, folded_indices)
    return tensor


def mode_n_product(tensor, matrix, mode):
    """ Mode-n product of a N-dimensional array with a matrix.

    Parameters
    ----------
    tensor : np.ndarray
        N-dimensional array
    matrix : np.ndarray
        2D array
    mode : int
        Specifies mode along which a `tensor` is multiplied by a `matrix`. The size of a `tensor` along this `mode`
        should be equal to the number of columns of the `matrix`. That is: ``tensor.shape[mode] = matrix.shape[1]``

    Returns
    -------
    result : np.ndarray
        The result of the mode-n product of a `tensor` with a `matrix` along specified `mode`.

    Notes
    -----
        Result of mode-n product does not depend on the folding/unfolding convention,
        as long as folding and unfolding operations belong to the same convention.
    """
    # TODO: Implement mode-n product with a vector
    if matrix.ndim != 2:
        raise ValueError("Mode-n product can only be performed with the 2D array, "
                         "whereas an array of order {} was provided".format(matrix.ndim))
    orig_shape = list(tensor.shape)
    new_shape = orig_shape
    new_shape[mode] = matrix.shape[0]
    result = fold(np.dot(matrix, unfold(tensor, mode)), mode, tuple(new_shape))
    return result


def _kolda_reorder(ndim, mode):
    """Reorders the elements
    """
    indices = list(range(ndim))
    element = indices.pop(mode)
    return ([element] + indices[::-1])



def sampled_khatri_rao(matrices, sample_size=None, skip_matrix=None):
    """ Sampled Khatri-Rao product of a list of matrices.

    Parameters
    ----------
    matrices : list[np.ndarray]
        List of matrices. Each matrix should have the same number of columns
    skip_matrix : int
        Index of a matrix (from the `matrices`) to be skipped. By default none are skipped
    reverse : bool
        If True, perform khatri-rao product on the list of matrices in the reversed order

    Returns
    -------
    result : np.ndarray
        The result of the Khatri-Rao product of the sampled matrices
    indices : tuple list
        list of indices for all modes

    References
    ----------
    -   Battaglino, C., Ballard, G., & Kolda, T. G. (2018). A Practical Randomized CP Tensor
        Decomposition. SIAM Journal on Matrix Analysis and Applications, 39(2), 876–901.
        http://doi.org/10.1137/17m1112303
    """
    np.random.seed(0)
    if len(matrices) < 2:
        raise ValueError('khatri_roa product requires a list of at least 2 matrices, '
                         'but {} given.'.format(len(matrices)))

    n_cols = matrices[0].shape[1]

    if skip_matrix is not None:
        matrices = [matrices[i] for i in range(len(matrices)) if i != skip_matrix]

    if sample_size is None:
        sample_size = int(10*n_cols*np.log(n_cols))

    result = np.ones((sample_size, n_cols))
    idxlist = [np.random.randint(0, m.shape[0], size=sample_size, dtype=int) for m in matrices]

    for i, (idx, mat) in enumerate(zip(idxlist, matrices)):
        if mat.shape[1] != n_cols:
            raise ValueError('All matrices must have the same number of columns.'+\
                    'Got {} and {}'.format(mat.shape, n_cols))

        result = hadamard([result, mat[idx, :]])

    return result, idxlist

def partitioned_khatri_rao(matrices, partitions, reverse=False):
    """ Khatri-Rao product of a list of matrices with variable partition size.

    Parameters
    ----------
    matrices : list[np.ndarray]
        List of matrices. Each matrix should have the same number of columns
    partitions : iterable
        containing tuples with each partition size. -1 indicates coloumn wise partition.
    reverse : bool
        If True, perform khatri-rao product on the list of matrices in the reversed order

    Returns
    -------
    result : np.ndarray
        The result of the Khatri-Rao product is a matrix of shape `()`
    """

    if len(matrices) != 2:
        raise ValueError('khatri_roa product with variable partitions requires a list of at exactly 2 matrices, '
                         'but {} given.'.format(len(matrices)))

    if len(matrices) != len(partitions):
        raise ValueError("Matrices and partitions need to be of the same size!")

    # replace coloumn wise indicators
    if -1 in partitions:
        _partitions = []
        for m, p in zip(matrices, partitions):
            if p == -1:
                _partitions.append((1,) * m.shape[1])
            else:
                _partitions.append(p)
        partitions = tuple(_partitions)

    n_partitions = len(partitions[0])
    if not all(len(part) == n_partitions for part in partitions):
        raise ValueError("Number ob partitions must be consistant across all matrices!")
    if not all(matrix.shape[1] == sum(partition) for matrix, partition in zip(matrices, partitions)):
        raise ValueError("The partitions don't fit the matrices.")

    if reverse:
        matrices = matrices[::-1]
        partitions = partitions[::-1]

    n_rows = matrices[0].shape[0] * matrices[1].shape[0]

    n_cols = sum(l*r for l, r in zip(*partitions))
        
    result = np.zeros((n_rows, n_cols))
    current_col, current_col_l, current_col_r = 0, 0, 0
    for i_left, i_right in zip(*partitions):
        matrix_l = matrices[0][:,current_col_l:current_col_l+i_left]
        matrix_r = matrices[1][:,current_col_r:current_col_r+i_right]
        _temp = kronecker([matrix_l, matrix_r])
        result[:,current_col:current_col+i_left*i_right] = _temp

        current_col_l += i_left
        current_col_r += i_right
        current_col += _temp.shape[1]
    return result
