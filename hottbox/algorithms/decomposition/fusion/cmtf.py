import functools
import warnings

import numpy as np

from hottbox.algorithms.decomposition.btd import BTD
from hottbox.algorithms.decomposition.cpd import BaseCPD
from hottbox.core.operations import hadamard, khatri_rao, partitioned_khatri_rao
from hottbox.core.structures import Tensor, TensorBTD
from hottbox.utils.generation.basic import residual_tensor, super_diag_tensor
from ..base import svd


# TODO: Organise this better - lazy work around used
class CMTF(BaseCPD):
    """ Coupled Matrix and Tensor factorization for two ``Tensors`` of order n and 2 with respect to a specified `rank`.

    Computed via alternating least squares (ALS)

    Parameters
    ----------
    max_iter : int
        Maximum number of iteration
    epsilon : float
        Threshold for the relative error of approximation.
    tol : float
        Threshold for convergence of factor matrices
    random_state : int
    verbose : bool
        If True, enable verbose output

    Attributes
    ----------
    cost : list
        A list of relative approximation errors at each iteration of the algorithm.

    References
    ----------
    -   Acar, Evrim, Evangelos E. Papalexakis, Gozde Gurdeniz, Morten A. Rasmussen,
        Anders J. Lawaetz, Mathias Nilsson and Rasmus Bro.
        “Structure-revealing data fusion.” BMC Bioinformatics (2013).
    -   Jeon, Byungsoo & Jeon, Inah & Sael, Lee & Kang, U. (2016).
        SCouT: Scalable coupled matrix-tensor factorization—Algorithm and discoveries.
        Int. Conf. Data Eng.. 811-822. 10.1109/ICDE.2016.7498292.
    """
    # TODO: change init use requiring a change in TensorCPD

    def __init__(self, max_iter=50, epsilon=10e-3, tol=10e-5,
                 random_state=None, verbose=False) -> None:
        super(CMTF, self).__init__(init='random',
                                   max_iter=max_iter,
                                   epsilon=epsilon,
                                   tol=tol,
                                   random_state=random_state,
                                   verbose=verbose)
        self.cost = []

    def copy(self):
        """ Copy of the CPD algorithm as a new object """
        new_object = super(CMTF, self).copy()
        new_object.cost = []
        return new_object

    @property
    def name(self):
        """ Name of the decomposition

        Returns
        -------
        decomposition_name : str
        """
        decomposition_name = super(CMTF, self).name
        return decomposition_name

    def decompose(self, tensor, mlst, rank):
        """ Performs factorisation using ALS on the two instances of ``tensor``
            with respect to the specified ``rank``

        Parameters
        ----------
        tensor : Tensor
            Multi-dimensional data to be decomposed
        mlst : List of `Tensor`
            List of two-dimensional `Tensor` to be decomposed
        rank : tuple
            Desired Kruskal rank for the given ``tensor``. Should contain only one value.
            If it is greater then any of dimensions then random initialisation is used

        Returns
        -------
        (fmat_a, fmat_b, t_recon, m_recon) : List(np.ndarray) or np.ndarray
            fmat_a, fmat_b are the list of components obtained by applying CMTF
            t_recon, m_recon : The reconstructed tensor and list of matrices

        """
        if not isinstance(tensor, Tensor):
            raise TypeError("Parameter `tensor` should be `Tensor`!")
        if not isinstance(mlst, list):
            raise TypeError("Parameter `mlst` should be a list of `Tensor`!")
        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")
        if len(rank) != 1:
            raise ValueError(
                "Parameter `rank` should be tuple with only one value!")
        if not all(isinstance(m, Tensor) for m in mlst):
            raise TypeError("Parameter `mlst` should be a list of `Tensor`!")
        if not all(m.order == 2 for m in mlst):
            raise ValueError(
                "All elements of `mlst` should be of order 2. It is a list of matrices!")

        modes = np.array([list(m.shape) for m in mlst])
        num_modes = len(modes)
        fmat_a, fmat_b = self._init_fmat(modes[:, 0], modes[:, 1], rank)
        norm = tensor.frob_norm
        for n_iter in range(self.max_iter):
            # Update tensor factors
            for i in range(num_modes):
                _v = hadamard([np.dot(a_i.T, a_i)
                               for k, a_i in enumerate(fmat_a) if k != i])
                _v += fmat_b[i].T.dot(fmat_b[i])
                kr_result = khatri_rao(fmat_a, skip_matrix=i, reverse=True)
                _prod_a = np.concatenate(
                    [tensor.unfold(i, inplace=False).data, mlst[i].data], axis=1)
                _prod_b = np.concatenate([kr_result.T, fmat_b[i].T], axis=1).T
                fmat_a[i] = _prod_a.dot(_prod_b).dot(np.linalg.pinv(_v))
            for i in range(num_modes):
                fmat_b[i] = mlst[i].data.T.dot(np.linalg.pinv(fmat_a[i]).T)

            t_recon, m_recon = self._reconstruct(fmat_a, fmat_b, num_modes)

            residual = np.linalg.norm(tensor.data-t_recon.data)
            for i in range(num_modes):
                residual += np.linalg.norm(mlst[i].data-m_recon[i].data)
            self.cost.append(abs(residual)/norm)

            if self.verbose:
                print('Iter {}: relative error of approximation = {}'.format(
                    n_iter, self.cost[-1]))

            # Check termination conditions
            if self.cost[-1] <= self.epsilon:
                if self.verbose:
                    print('Relative error of approximation has reached the acceptable level: {}'
                          .format(self.cost[-1]))
                break
            if self.converged:
                if self.verbose:
                    print('Converged in {} iteration(s)'.format(len(self.cost)))
                break

        if self.verbose and not self.converged and self.cost[-1] > self.epsilon:
            print('Maximum number of iterations ({}) has been reached. '
                  'Variation = {}'.format(self.max_iter, abs(self.cost[-2] - self.cost[-1])))

        # TODO: possibly make another structure
        return fmat_a, fmat_b, t_recon, m_recon

    @property
    def converged(self):
        """ Checks convergence of the CPD-ALS algorithm.
        Returns
        -------
        bool
        """
        # This insures that the cost has been computed at least twice without checking iterations
        try:
            is_converged = abs(self.cost[-2] - self.cost[-1]) <= self.tol
        except IndexError:
            is_converged = False
        return is_converged

    def _init_fmat(self, shape_i, shape_j, rank):
        """ Initialisation of matrices used in CMTF
        Parameters
        ----------
        shape_i : np.ndarray(int)
            Shape[0] of all matrices
        shape_j : np.ndarray(int)
            Shape[1] of all matrices
        rank : int
            The rank specified for factorisation
        Returns
        -------
        (fmat_a, fmat_b) : List(np.ndarray)
            Two lists of the factor matrices
        """
        self.cost = []  # Reset cost every time when method decompose is called
        _r = rank[0]
        if (np.array(shape_i) < _r).sum() != 0:
            warnings.warn(
                "Specified rank is greater then one of the dimensions of a tensor ({} > {}).\n"
                "Factor matrices have been initialized randomly.".format(
                    _r, shape_i), RuntimeWarning
            )
        fmat_a = [self.random_state.randn(i_n, _r) for i_n in shape_i]
        fmat_b = [self.random_state.randn(j_n, _r) for j_n in shape_j]
        return fmat_a, fmat_b

    @staticmethod
    def _reconstruct(fmat_a, fmat_b, n_mat):
        """ Reconstruct the tensor and matrix after the coupled factorisation
        Parameters
        ----------
        fmat_a : List(np.ndarray)
            Multidimensional data obtained from the factorisation
        fmat_b : List(np.ndarray)
            Multidimensional data obtained from the factorisation
        n_mat : int
            Number of matrices provided to fuse
        Returns
        -------
        (core_tensor, lrecon) : np.ndarray or List(np.ndarray)
            Reconstructed tensor and list of matrices obtained from the factorisation
        """
        core_values = np.repeat(np.array([1]), fmat_a[0].shape[1])
        _r = (fmat_a[0].shape[1], )
        core_shape = _r * len(fmat_a)
        core_tensor = super_diag_tensor(core_shape, values=core_values)
        for mode, fmat in enumerate(fmat_a):
            core_tensor.mode_n_product(fmat, mode=mode, inplace=True)
        lrecon = [Tensor(fmat_a[i].dot(fmat_b[i].T)) for i in range(n_mat)]
        return core_tensor, lrecon

    def plot(self):
        print('At the moment, `plot()` is not implemented for the {}'.format(self.name))


class CMTF2(CMTF):
    def decompose(self, tensor, matrix, rank):
        """ Performs factorisation using ALS on the two instances of ``tensor``
            with respect to the specified ``rank``

        Parameters
        ----------
        tensor : Tensor
            Multi-dimensional data to be decomposed
        mlst : List of `Tensor`
            List of two-dimensional `Tensor` to be decomposed
        rank : tuple
            Desired Kruskal rank for the given ``tensor``. Should contain only one value.
            If it is greater then any of dimensions then random initialisation is used

        Returns
        -------
        (fmat_a, fmat_b, t_recon, m_recon) : List(np.ndarray) or np.ndarray
            fmat_a, fmat_b are the list of components obtained by applying CMTF
            t_recon, m_recon : The reconstructed tensor and list of matrices

        """
        if not isinstance(tensor, Tensor):
            raise TypeError("Parameter `tensor` should be `Tensor`!")
        if not isinstance(matrix, Tensor):
            raise TypeError("Parameter `mlst` should be a list of `Tensor`!")
        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")
        if len(rank) != 1:
            raise ValueError(
                "Parameter `rank` should be tuple with only one value!")
        if not matrix.order == 2:
            raise ValueError(
                "All elements of `mlst` should be of order 2. It is a list of matrices!")

        self.cost = []  # Reset cost every time when method decompose is called

        modes = tensor.shape
        num_modes = len(modes)
        fmat_tensor, fmat_matrix = self._init_fmat(
            modes, matrix.shape[1], rank)
        norm = tensor.frob_norm
        norm += matrix.frob_norm

        for n_iter in range(self.max_iter):
            # Update tensor factors
            for mode in range(num_modes):
                if mode == 2:
                    # update coupled factor matrix
                    _v = hadamard([np.dot(a_i.T, a_i)
                                   for k, a_i in enumerate(fmat_tensor) if k != mode])
                    _v += fmat_matrix.T.dot(fmat_matrix)
                    kr_result = khatri_rao(
                        fmat_tensor, skip_matrix=mode, reverse=True)
                    _prod_a = np.concatenate(
                        [tensor.unfold(mode, inplace=False).data, matrix.data], axis=1)
                    _prod_b = np.concatenate(
                        [kr_result.T, fmat_matrix.T], axis=1).T
                    fmat_tensor[mode] = _prod_a.dot(
                        _prod_b).dot(np.linalg.pinv(_v))

                    # update matrix factor matrix
                    fmat_matrix = matrix.data.T.dot(
                        np.linalg.pinv(fmat_tensor[mode]).T)

                else:
                    # update tensor only factor matrices
                    kr_result = khatri_rao(fmat_tensor, skip_matrix=mode)
                    hadamard_result = hadamard(
                        [np.dot(mat.T, mat) for i, mat in enumerate(fmat_tensor) if i != mode])
                    # Do consecutive multiplication of np.ndarray
                    update = functools.reduce(np.dot, [tensor.unfold(mode, inplace=False).data,
                                                       kr_result,
                                                       np.linalg.pinv(hadamard_result)])
                    fmat_tensor[mode] = update

            t_recon, m_recon = self._reconstruct(fmat_tensor, fmat_matrix)

            residual = residual_tensor(tensor, t_recon).frob_norm
            residual += residual_tensor(matrix, m_recon).frob_norm

            self.cost.append(abs(residual/norm))

            if self.verbose:
                print('Iter {}: relative error of approximation = {}'.format(
                    n_iter, self.cost[-1]))

            # Check termination conditions
            if self.cost[-1] <= self.epsilon:
                if self.verbose:
                    print('Relative error of approximation has reached the acceptable level: {}'
                          .format(self.cost[-1]))
                break
            if self.converged:
                if self.verbose:
                    print('Converged in {} iteration(s)'.format(len(self.cost)))
                break

        if self.verbose and not self.converged and self.cost[-1] > self.epsilon:
            print('Maximum number of iterations ({}) has been reached. '
                  'Variation = {}'.format(self.max_iter, abs(self.cost[-2] - self.cost[-1])))

        t_recon, m_recon = self._reconstruct(fmat_tensor, fmat_matrix)

        # TODO: possibly make another structure
        return fmat_tensor, fmat_matrix, t_recon, m_recon

    def _init_fmat(self, shape_tensor, matrix_x, rank):
        """ Initialisation of matrices used in CMTF
        Parameters
        ----------
        shape_i : np.ndarray(int)
            Shape[0] of all matrices
        shape_j : np.ndarray(int)
            Shape[1] of all matrices
        rank : int
            The rank specified for factorisation
        Returns
        -------
        (fmat_a, fmat_b) : List(np.ndarray)
            Two lists of the factor matrices
        """
        _r = rank[0]
        if (np.array(shape_tensor) < _r).sum() != 0:
            warnings.warn(
                "Specified rank is greater then one of the dimensions of a tensor ({} > {}).\n"
                "Factor matrices have been initialized randomly.".format(
                    _r, shape_tensor), RuntimeWarning
            )
        fmat_a = [self.random_state.randn(i_n, _r) for i_n in shape_tensor]
        fmat_b = self.random_state.randn(matrix_x, _r)
        return fmat_a, fmat_b

    @staticmethod
    def _reconstruct(fmat_tensor, fmat_matrix):
        """ Reconstruct the tensor and matrix after the coupled factorisation
        Parameters
        ----------
        fmat_a : List(np.ndarray)
            Multidimensional data obtained from the factorisation
        fmat_b : List(np.ndarray)
            Multidimensional data obtained from the factorisation
        n_mat : int
            Number of matrices provided to fuse
        Returns
        -------
        (core_tensor, lrecon) : np.ndarray or List(np.ndarray)
            Reconstructed tensor and list of matrices obtained from the factorisation
        """
        core_values = np.repeat(np.array([1]), fmat_tensor[0].shape[1])
        _r = (fmat_tensor[0].shape[1], )
        core_shape = _r * len(fmat_tensor)
        core_tensor = super_diag_tensor(core_shape, values=core_values)
        for mode, fmat in enumerate(fmat_tensor):
            core_tensor.mode_n_product(fmat, mode=mode, inplace=True)

        matrix_recon = Tensor(np.sum([np.outer(fmat_tensor[2][:, i], fmat_matrix[:, i])
                                      for i in range(fmat_matrix.shape[1])], axis=0))
        return core_tensor, matrix_recon

    def plot(self):
        print('At the moment, `plot()` is not implemented for the {}'.format(self.name))


class CMTF_BTD(BTD):
    def decompose(self, tensor, matrix, rank, keep_meta=0, kr_reverse=False, factor_mat=None):
        """ Performs BTD on the ``tensor`` with respect to the specified ``rank``

        Parameters
        ----------
        tensor : Tensor
            Multi-dimensional data to be decomposed
        rank : tuple
            tuple of tuple containing each components rank.
            Length of the outer tuple defines the number of components
        keep_meta : int
            Keep meta information about modes of the given ``tensor``.
            0 - the output will have default values for the meta data
            1 - keep only mode names
            2 - keep mode names and indices
        kr_reverse : bool
        factor_mat : list(np.ndarray)
            Initial list of factor matrices.
            Specifying this option will ignore ``init``.

        Returns
        -------
        tensor_cpd : TensorCPD
            CP representation of the ``tensor``

        Notes
        -----
        khatri-rao product should be of matrices in reversed order. But this will duplicate original data (e.g. images)
        Probably this has something to do with data ordering in Python and how it relates to kr product
        """
        # TODO change for BTD

        if not isinstance(tensor, Tensor):
            raise TypeError(
                "Parameter `tensor` should be an object of `Tensor` class!")
        if len(tensor.shape) != 3:
            raise ValueError("BTD only supported for 3-way tensors!")

        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")

        # check for illegal components in rank
        if len(rank) < 1 or not all(len(comp) == 3 for comp in rank):
            raise ValueError("Parameter `rank` should be tuple containing"
                             "at least one tupel of size 3!")

        rank_one_mode, *_ = np.where(np.array(rank).sum(axis=0) == len(rank))
        if len(rank_one_mode) == 0:
            raise ValueError(
                "At least one mode needs to be 1 across all rank elements!")
        elif len(rank_one_mode) == 1:
            rank_one_mode = rank_one_mode[0]
        elif len(rank_one_mode) == 2:
            raise ValueError("Two modes with rank 1 given, must be 1 or 3!")
        elif len(rank_one_mode) == 3:
            rank_one_mode = rank_one_mode[2]
            if self.verbose:
                print("The given ranks for BTD are (1,1,1) across the board."
                      "This is basically a CPD.")
        else:
            raise ValueError(
                "Unexpected case while sanity checking given rank for BTD\n:{}".format(rank))

        if rank_one_mode != 2:
            raise ValueError("Your rank one mode must be the 3rd one."
                             "Please swap the tensor axes accordingly.")

        if not all(comp.count((sum(comp)-1)//2) >= 2 for comp in rank):
            raise ValueError(
                "At least one component has 2 different ranks for L_r and L_r:\n{}".format(rank))

        if not isinstance(matrix, Tensor):
            raise TypeError("Parameter `mlst` should be a list of `Tensor`!")

        if not matrix.order == 2:
            raise ValueError(
                "All elements of `mlst` should be of order 2. It is a list of matrices!")

        if factor_mat is None:
            fmat_tensor, fmat_matrix = self._init_fmat(
                tensor, matrix.shape[1], rank)
        else:
            # TODO sanity check for passing factor_mat
            raise NotImplementedError()

        self.cost = []  # Reset cost every time when method decompose is called
        tensor_btd = None
        # core_values = np.repeat(np.array([1]), rank)
        norm = tensor.frob_norm
        norm += matrix.frob_norm

        # extract the L_r ranks
        higher_rank_partitions = [r[0] for r in rank]

        for n_iter in range(self.max_iter):

            for mode in range(3):
                current_mode_skipped = fmat_tensor[:mode] + \
                    fmat_tensor[mode + 1:]

                if mode == rank_one_mode:
                    # list of views for rank L_r fmats
                    # A = [A_1, A_2,  ... , A_R]
                    # paired for easy use with khatri_rao
                    list_of_views = list()

                    curr_col = 0
                    for width in higher_rank_partitions:
                        l_view = current_mode_skipped[0][:,
                                                         curr_col:curr_col+width]
                        r_view = current_mode_skipped[1][:,
                                                         curr_col:curr_col+width]
                        list_of_views.append((l_view, r_view))
                        curr_col += width

                    # coloumn wise khatri rao with each view
                    _temp = [khatri_rao(pair, reverse=kr_reverse)
                             for pair in list_of_views]
                    # sum up rows
                    _temp = [el.sum(axis=1, keepdims=True) for el in _temp]
                    _temp = np.concatenate(
                        [np.hstack(_temp).T, fmat_matrix.T], axis=1)
                    # concat vectors to single matrix and pseudo inverse
                    _temp = np.linalg.pinv(_temp)
                    # dot product with unfold and transpose
                    _unfold_concat = np.concatenate(
                        [tensor.unfold(mode, inplace=False).data, matrix.data], axis=1)
                    _temp = np.dot(_unfold_concat, _temp)

                    _temp /= np.linalg.norm(_temp, axis=0)

                    # reassign calculation to fmat list
                    fmat_tensor[mode] = _temp

                    # update matrix factor matrix
                    fmat_matrix = matrix.data.T.dot(
                        np.linalg.pinv(fmat_tensor[mode]).T)

                else:
                    # set correct order for partitioned and coloumn wise
                    _partitions = [higher_rank_partitions, -1]
                    # khatri rao with partitions given by rank
                    _temp = partitioned_khatri_rao(
                        current_mode_skipped, _partitions, reverse=kr_reverse)
                    # pseudo inverse
                    _temp = np.linalg.pinv(_temp)
                    # multiply by current mode unfold and transpose
                    # NOTE Not sure why I had to transpose the unfolding.
                    _temp = np.transpose(
                        np.dot(_temp, tensor.unfold(mode, inplace=False).data.T))

                    if mode == 1:
                        curr_col = 0
                        for width in higher_rank_partitions:
                            view = _temp[:, curr_col:curr_col+width]
                            _temp[:, curr_col:curr_col +
                                  width], _ = np.linalg.qr(view)
                            curr_col += width

                    # reassign calculation to fmat list
                    fmat_tensor[mode] = _temp

            # Update cost
            tensor_btd = TensorBTD(fmat=fmat_tensor, rank=rank)
            recon_matrix = self._reconstruct_matrix(fmat_tensor, fmat_matrix)
            residual = residual_tensor(tensor, tensor_btd).frob_norm
            residual += residual_tensor(matrix, recon_matrix).frob_norm

            self.cost.append(abs(residual / norm))
            if self.verbose:
                print('Iter {}: relative error of approximation = {}'.format(
                    n_iter, self.cost[-1]))

            # Check termination conditions
            if self.cost[-1] <= self.epsilon:
                if self.verbose:
                    print('Relative error of approximation has reached the acceptable level: {}'.format(
                        self.cost[-1]))
                break
            if self.converged:
                if self.verbose:
                    print('Converged in {} iteration(s)'.format(len(self.cost)))
                break
        if self.verbose and not self.converged and self.cost[-1] > self.epsilon:
            print('Maximum number of iterations ({}) has been reached. '
                  'Variation = {}'.format(self.max_iter, abs(self.cost[-2] - self.cost[-1])))

        if keep_meta == 1:
            mode_names = {i: mode.name for i, mode in enumerate(tensor.modes)}
            tensor_btd.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor_btd.copy_modes(tensor)
        else:
            pass
        return tensor_btd, fmat_matrix, recon_matrix

    def _init_fmat(self, tensor, matrix_x, rank):
        """
        """
        fmat_tensor = super(CMTF_BTD, self)._init_fmat(tensor, rank)

        fmat_matrix = self.random_state.randn(matrix_x, len(rank))

        return fmat_tensor, fmat_matrix

    @staticmethod
    def _reconstruct_matrix(fmat_tensor, fmat_matrix):
        """ Reconstruct the tensor and matrix after the coupled factorisation
        Parameters
        ----------
        fmat_tensor : List(np.ndarray)
            Multidimensional data obtained from the factorisation
        fmat_matrix : np.ndarray
            Multidimensional data obtained from the factorisation

        Returns
        -------
        matrix_recon : Tensor
            Reconstructed matrix
        """

        matrix_recon = Tensor(np.sum([np.outer(fmat_tensor[2][:, i], fmat_matrix[:, i])
                                      for i in range(fmat_matrix.shape[1])], axis=0))
        return matrix_recon
