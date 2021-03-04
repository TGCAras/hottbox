import functools
import warnings

import numpy as np

from hottbox.algorithms.decomposition.btd import BTD_LL11
from hottbox.algorithms.decomposition.cpd import CPD
from hottbox.core.operations import hadamard, khatri_rao, partitioned_khatri_rao
from hottbox.core.structures import Tensor, TensorBTDLL11, TensorCPD
from hottbox.utils.generation.basic import residual_tensor, super_diag_tensor
from ..base import svd


class CBTTF(BTD_LL11):
    def decompose(self, LL11_tensor, cpd_tensor, rank, keep_meta=0, kr_reverse=False, factor_mat=None):
        """ Performs BTD on the ``LL11_tensor`` with respect to the specified ``rank``

        Parameters
        ----------
        LL11_tensor : Tensor
            Multi-dimensional data to be decomposed
        rank : tuple
            tuple of tuple containing each components rank.
            Length of the outer tuple defines the number of components
        keep_meta : int
            Keep meta information about modes of the given ``LL11_tensor``.
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

        # Sanity check BTD

        if not isinstance(LL11_tensor, Tensor):
            raise TypeError(
                "Parameter `LL11_tensor` should be an object of `Tensor` class!")
        if len(LL11_tensor.shape) != 4:
            raise ValueError("BTD_LL11 only supported for 4-way tensors!")

        if not isinstance(rank, tuple):
            raise TypeError("Parameter `rank` should be passed as a tuple!")

        # check for illegal components in rank
        if len(rank) < 1 or not all(len(comp) == 4 for comp in rank):
            raise ValueError("Parameter `rank` should be tuple containing"
                             "at least one tupel of size 4!")

        rank_one_mode, *_ = np.where(np.array(rank).sum(axis=0) == len(rank))
        if len(rank_one_mode) <= 1:
            raise ValueError(
                "At least two modes needs to be 1 across all rank elements!")
        elif len(rank_one_mode) == 2:
            rank_one_mode = rank_one_mode
        elif len(rank_one_mode) == 3:
            raise ValueError("Three modes with rank 1 given, must be 2 or 4!")
        elif len(rank_one_mode) == 4:
            rank_one_mode = rank_one_mode[2:]
            if self.verbose:
                print("The given ranks for BTD are (1,1,1,1) across the board."
                      "This is basically a CPD.")
        else:
            raise ValueError(
                "Unexpected case while sanity checking given rank for BTD\n:{}".format(rank))

        if not all(r in [2, 3] for r in rank_one_mode):
            raise ValueError("Your rank one modes must be 3rd and 4th."
                             "Please swap the tensor axes accordingly.")

        if not all(comp.count((sum(comp)-2)//2) >= 2 for comp in rank):
            raise ValueError(
                "At least one component has 2 different ranks for L_r and L_r:\n{}".format(rank))

        # CPD sanity checks

        if not isinstance(cpd_tensor, Tensor):
            raise TypeError("Parameter `cpd_tensor` should be a `Tensor`!")

        if not cpd_tensor.order == 3:
            raise ValueError(
                "cpd_tensor should have order 3")

        if factor_mat is None:
            fmat_LL11_tensor, fmat_cpd_tensor = self._init_fmat(
                LL11_tensor, cpd_tensor, rank)

            # # FIXME just a test
            # fmat_LL11_tensor[3] = np.ones(fmat_LL11_tensor[3].shape)
            # fmat_cpd_tensor[2] = np.ones(fmat_LL11_tensor[3].shape)
        else:
            # TODO sanity check for passing factor_mat
            raise NotImplementedError()

        self.cost = []  # Reset cost every time when method decompose is called
        tensor_btd_model = None
        cpd_core_values = np.repeat(np.array([1]), len(rank))
        tensor_cpd_model = None
        coupled_mode = 3
        # core_values = np.repeat(np.array([1]), rank)
        norm = LL11_tensor.frob_norm
        norm += cpd_tensor.frob_norm

        # extract the L_r ranks
        higher_rank_partitions = [r[0] for r in rank]

        for n_iter in range(self.max_iter):

            for mode in range(4):
                current_mode_skipped = fmat_LL11_tensor[:mode] + \
                    fmat_LL11_tensor[mode + 1:]

                if mode in rank_one_mode:
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

                    _temp = khatri_rao([np.hstack(_temp), current_mode_skipped[2]])

                    if mode == coupled_mode:
                        _temp = np.concatenate(
                            [_temp.T, khatri_rao(fmat_cpd_tensor[:2]).T], axis=1)
                        # concat vectors to single matrix and pseudo inverse
                        _temp = np.linalg.pinv(_temp)
                    else:
                        _temp = np.linalg.pinv(_temp.T)

                    # dot product with unfold and transpose
                    if mode == coupled_mode:
                        # concatenate for coupled case
                        _unfold_concat = np.concatenate(
                            [LL11_tensor.unfold(mode, inplace=False).data, cpd_tensor.unfold(mode-1, inplace=False).data], axis=1)
                        _temp = np.dot(_unfold_concat, _temp)
                    else:
                        _temp = np.dot(LL11_tensor.unfold(
                            mode, inplace=False).data, _temp)

                    _temp /= np.linalg.norm(_temp, axis=0)

                    # # FIXME CONSTRAIN MODE 4
                    # if mode == coupled_mode:
                    #     _temp = np.abs(_temp)
                    #     _temp = _temp / np.max(_temp, axis=0)
                    #     # print(_temp)
                    #     for col in range(_temp.shape[1]):
                    #         _temp[:, col] = np.around(_temp[:, col])

                    # reassign calculation to fmat list
                    fmat_LL11_tensor[mode] = _temp

                    if mode == coupled_mode:
                        # assign coupled faktor matrix to cpd fmats
                        fmat_cpd_tensor[mode - 1] = _temp
                        # normal cpd for remaining modes of CPD Tensor
                        for cpd_mode in range(cpd_tensor.order - 1):
                            kr_result = khatri_rao(
                                fmat_cpd_tensor, skip_matrix=cpd_mode, reverse=kr_reverse)
                            hadamard_result = hadamard(
                                [np.dot(mat.T, mat) for i, mat in enumerate(fmat_cpd_tensor) if i != cpd_mode])
                            # Do consecutive multiplication of np.ndarray
                            update = functools.reduce(np.dot, [cpd_tensor.unfold(cpd_mode, inplace=False).data,
                                                               kr_result,
                                                               np.linalg.pinv(hadamard_result)])
                            fmat_cpd_tensor[cpd_mode] = update

                else:
                    # set correct order for partitioned and coloumn wise
                    _partitions = [higher_rank_partitions, -1]
                    # khatri rao with partitions given by rank
                    _temp = partitioned_khatri_rao(
                        current_mode_skipped[:2], _partitions, reverse=kr_reverse)
                    _temp = partitioned_khatri_rao(
                        [_temp, current_mode_skipped[2]], _partitions, reverse=kr_reverse)
                    # pseudo inverse
                    _temp = np.linalg.pinv(_temp)
                    # multiply by current mode unfold and transpose
                    _temp = np.transpose(np.dot(_temp, LL11_tensor.unfold(mode, inplace=False).data.T))

                    # normalize (shifts norms into fmat_LL11_tensor[0])
                    if mode == 1:
                        curr_col = 0
                        for width in higher_rank_partitions:
                            view = _temp[:,curr_col:curr_col+width]
                            _temp[:,curr_col:curr_col+width], _ = np.linalg.qr(view)
                            curr_col += width

                    # reassign calculation to fmat list
                    fmat_LL11_tensor[mode] = _temp

            # Update cost
            tensor_btd_model = TensorBTDLL11(fmat=fmat_LL11_tensor, rank=rank)
            tensor_cpd_model = TensorCPD(fmat=fmat_cpd_tensor, core_values=cpd_core_values)
            residual = residual_tensor(LL11_tensor, tensor_btd_model).frob_norm
            residual += residual_tensor(cpd_tensor, tensor_cpd_model).frob_norm

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
            mode_names = {i: mode.name for i,
                          mode in enumerate(LL11_tensor.modes)}
            tensor_btd_model.set_mode_names(mode_names=mode_names)
        elif keep_meta == 2:
            tensor_btd_model.copy_modes(LL11_tensor)
        else:
            pass
        return tensor_btd_model, tensor_cpd_model

    def _init_fmat(self, tensor_btd, tensor_cpd, rank):
        """
        """
        fmat_tensor_btd = super(CBTTF, self)._init_fmat(tensor_btd, rank)

        cpd_decomposer = CPD()
        fmat_tensor_cpd = cpd_decomposer._init_fmat(tensor=tensor_cpd, rank=(len(rank),))

        return fmat_tensor_btd, fmat_tensor_cpd
