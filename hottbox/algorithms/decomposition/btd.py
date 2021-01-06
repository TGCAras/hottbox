import functools
import warnings
import numpy as np
from hottbox.utils.generation.basic import residual_tensor
from hottbox.core.structures import Tensor, TensorCPD, TensorBTD
from hottbox.core.operations import khatri_rao, hadamard, sampled_khatri_rao, partitioned_khatri_rao
from .base import Decomposition, svd


class BaseBTD(Decomposition):
    def __init__(self, init, max_iter, epsilon, tol, random_state, verbose):
        super(BaseBTD, self).__init__()
        self.init = init
        self.max_iter = max_iter
        self.epsilon = epsilon
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose

    def copy(self):
        """ Copy of the Decomposition as a new object """
        new_object = super(BaseBTD, self).copy()
        return new_object

    @property
    def name(self):
        """ Name of the decomposition

        Returns
        -------
        decomposition_name : str
        """
        decomposition_name = super(BaseBTD, self).name
        return decomposition_name

    def decompose(self, tensor, rank, keep_meta):
        raise NotImplementedError('Not implemented in base (BaseBTD) class')

    @property
    def converged(self):
        """ Checks convergence

        Returns
        -------
        is_converged : bool
        """
        try:  # This insures that the cost has been computed at least twice without checking number of iterations
            is_converged = abs(self.cost[-2] - self.cost[-1]) <= self.tol
        except IndexError:
            is_converged = False
        return is_converged

    def _init_fmat(self, tensor, rank):
        # TODO rework for BTD
        """ Initialisation of factor matrices

        Parameters
        ----------
        tensor : Tensor
            Multidimensional data to be decomposed
        rank : tuple
            Should be of shape (R,1), where R is the desired tensor rank. It should be passed as tuple for consistency.

        Returns
        -------
        fmat : list[np.ndarray]
            List of factor matrices
        """
        t_rank = rank[0]
        fmat = [np.array([])] * tensor.order
        # Check if all dimensions are greater then kryskal rank
        dim_check = (np.array(tensor.shape) >= t_rank).sum() == tensor.order
        if dim_check:
            if self.init == 'svd':
                for mode in range(tensor.order):
                    # TODO: don't really like this implementation
                    k = tensor.unfold(mode, inplace=False).data
                    fmat[mode], _, _ = svd(k, t_rank)
            elif self.init == 'random':
                fmat = [np.random.randn(mode_size, t_rank)
                        for mode_size in tensor.shape]
            else:
                raise NotImplementedError(
                    'The given initialization is not available')
        else:
            fmat = [np.random.randn(mode_size, t_rank)
                    for mode_size in tensor.shape]
            if self.verbose and self.init != 'random':
                warnings.warn(
                    "Specified rank value is greater then one of the dimensions of a tensor ({} > {}).\n"
                    "Factor matrices have been initialized randomly.".format(
                        t_rank, tensor.shape), RuntimeWarning
                )
        return fmat

    def plot(self):
        raise NotImplementedError('Not implemented in base (BaseBTD) class')


class BTD(BaseBTD):
    # TODO Doc String

    def __init__(self, init='svd', max_iter=50, epsilon=10e-3, tol=10e-5, random_state=None, verbose=False):
        super(BTD, self).__init__(init=init,
                                  max_iter=max_iter,
                                  epsilon=epsilon,
                                  tol=tol,
                                  random_state=random_state,
                                  verbose=verbose)
        self.cost = []

    def copy(self):
        """ Copy of the BTD algorithm as a new object """
        # TODO check whether additional things need to be copied
        new_object = super(BTD, self).copy()
        new_object.cost = []
        return new_object

    def decompose(self, tensor, rank, keep_meta=0, kr_reverse=False, factor_mat=None):
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

        if factor_mat is None:
            fmat = self._init_fmat(tensor, rank)
        else:
            # TODO sanity check for passing factor_mat
            raise NotImplementedError()

        self.cost = []  # Reset cost every time when method decompose is called
        tensor_btd = None
        # core_values = np.repeat(np.array([1]), rank)
        norm = tensor.frob_norm

        # extract the L_r ranks
        higher_rank_partitions = [r[0] for r in rank]

        for n_iter in range(self.max_iter):

            for mode in range(3):
                current_mode_skipped = fmat[:mode] + fmat[mode + 1:]

                if mode == 1:
                    current_mode_skipped.reverse()

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
                    _temp = [khatri_rao(pair, reverse=kr_reverse) for pair in list_of_views]
                    # sum up rows
                    _temp = [el.sum(axis=1, keepdims=True) for el in _temp]
                    # concat vectors to single matrix and pseudo inverse
                    _temp = np.linalg.pinv(np.hstack(_temp).T)
                    # dot product with unfold and transpose

                    _temp = np.dot(tensor.unfold(mode, inplace=False).data, _temp)

                    _temp /= np.linalg.norm(_temp, axis=0)

                    # reassign calculation to fmat list
                    fmat[mode] = _temp

                else:
                    # set correct order for partitioned and coloumn wise
                    if mode == 0:
                        _partitions = [higher_rank_partitions, -1]
                    else:
                        _partitions = [-1, higher_rank_partitions]
                    # khatri rao with partitions given by rank
                    _temp = partitioned_khatri_rao(
                        current_mode_skipped, _partitions, reverse=kr_reverse)
                    # pseudo inverse
                    _temp = np.linalg.pinv(_temp.T)
                    # multiply by current mode unfold and transpose
                    # NOTE Not sure why I had to transpose the unfolding.
                    _temp = np.dot(tensor.unfold(mode, inplace=False).data, _temp)

                    if mode == 1:
                        curr_col = 0
                        for width in higher_rank_partitions:
                            view = _temp[:,curr_col:curr_col+width]
                            _temp[:,curr_col:curr_col+width], _ = np.linalg.qr(view)
                            curr_col += width

                    # reassign calculation to fmat list
                    fmat[mode] = _temp

            # Update cost
            tensor_btd = TensorBTD(fmat=fmat, rank=rank)
            residual = residual_tensor(tensor, tensor_btd)
            self.cost.append(abs(residual.frob_norm / norm))
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
        return tensor_btd

    @property
    def converged(self):
        """ Checks convergence of the BTD-ALS algorithm.

        Returns
        -------
        bool
        """
        is_converged = super(BTD, self).converged
        return is_converged

    def _init_fmat(self, tensor, rank):
        x_size = tensor.shape
        y_size = np.array(rank).sum(axis=0)

        fmat = [np.random.randn(*size)
                for size in zip(x_size, y_size)]
        # TODO
        # fmat = super(BTD, self)._init_fmat(tensor=tensor,
        #                                    rank=rank)
        return fmat

    def plot(self):
        # TODO implement a nice plot
        print('At the moment, `plot()` is not implemented for the {}'.format(self.name))
