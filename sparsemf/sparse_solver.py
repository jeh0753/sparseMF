# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, print_function, division

import numpy as np
from six.moves import range
from scipy.sparse import issparse, coo_matrix, csr_matrix
from sparse_biscale import SBiScale


class Solver(object):
    ''' A general solver algorithm, used as a base for the Soft Impute algorithms to handle generalizable processes. '''
    def __init__(
            self,
            n_imputations=1,
            min_value=None,
            max_value=None,
            max_rank=None,
            normalizer=None,
            sparse=False):
        self.random = np.random.RandomState()
        self.random.seed(0)
        self.n_imputations = n_imputations
        self.min_value = min_value
        self.max_value = max_value
        self.max_rank = max_rank
        self.normalizer = normalizer
        self.sparse = sparse
        self.missing_mask = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if v is None or isinstance(v, (float, int)):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(field_list))

    def _check_input(self, X):
        if len(X.shape) != 2:
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))

    def _check_missing_value_mask(self):
        rows, cols, shape = self.missing_mask
        full_size = shape[0] * shape[1]
        if len(rows) == full_size:
            raise ValueError("Input matrix is not missing any values")
        if len(rows) == 0:
            raise ValueError("Input matrix must have some non-missing values")

    def _check_max_rank(self, X):
        m, n = X.shape
        full_rank = min(m, n)
        self.max_rank = min(self.max_rank, full_rank)
        return self

    def _preprocess_sparse(self, X):
        '''Note: this is the fill function for sparse matrices'''
        # Check max_rank input
        rows, cols, shape = self.missing_mask
        n, m = X.shape
        J = self.max_rank

        # Initiate U, D_sq and V
        D_sq = np.ones(J)# we call it D_sq because A=UD and B=VD and AB'=U D_sq V^T
        V = np.zeros((m,J))
        U = self.random.normal(size=(n,J))
        U = np.linalg.svd(U)[0] # takes the U component of the SVD of old U. This makes it m by m in shape
        U = U[:, :J] # this happens by default in R's version of SVD. 

        return (U, D_sq, V) # this may not work for copy step 

    def fill(
            self,
            X,
            inplace=False):
        """
        Parameters
        ----------
        X : np.array
            Data array containing NaN entries

        inplace : bool
            Modify matrix or fill a copy

        """
        if not inplace:
            X = X.copy()

        missing_mask = self.missing_mask

        X = self._preprocess_sparse(X)

        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        #TODO - separate out safety checks in _preprocess_sparse as well, and include them here instead

        self._check_input(X)
        shape = X.shape
        coo = coo_matrix(X)
        row_id = coo.row
        col_id = coo.col
        self.missing_mask = row_id, col_id, shape
        self._check_max_rank(X)
        self._check_missing_value_mask()
        return X 

    def solve(self, X, missing_mask, X_original=None):
        """
        Given an initialized matrix X and a mask of where its missing values had been, return a completion of X.
        """
        raise ValueError("%s.solve not yet implemented!" % (
            self.__class__.__name__,))

    def single_imputation(self, X):
        """ 
        Runs the algorithm through until convergence, after conducting necessary preprocessing. 
        """
        if isinstance(self.normalizer, SBiScale):
            X = csr_matrix(X)
            X = self.normalizer.fit(X)
        X_original = self.prepare_input_data(X)
        missing_mask = self.missing_mask
        X = X_original.copy()
        X_filled = self.fill(X, inplace=True) #X_filled is a decomposed set of svd components if X is sparse. It is initialized randomly

        X_result_svd = self.solve(X_filled, X_original=X_original)
        X_result = X_result_svd

        return X_result

    def multiple_imputations(self, X):
        """
        Generate multiple imputations of the same incomplete matrix, if matrix is
        not sparse.
        """
        return [self.single_imputation(X)]

    def complete(self, X):
        """
        X: scipy.sparse matrix
            The incomplete, sparse matrix to be used for fitting the model.
        Returns: 
            The SVD that best approximates the true, completed matrix X. 
        """
        imputations = self.multiple_imputations(X)
        if len(imputations) == 1:
            return imputations[0]
        else:
            return np.mean(imputations, axis=0)

