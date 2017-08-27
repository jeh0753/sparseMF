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
from scipy.sparse import issparse, coo_matrix

#from .common import generate_random_column_samples


class Solver(object):
    def __init__(
            self,
            fill_method="zero",
            n_imputations=1,
            min_value=None,
            max_value=None,
            max_rank=None,
            normalizer=None):
        self.random = np.random.RandomState()
        self.random.seed(0)
        self.fill_method = fill_method
        self.n_imputations = n_imputations
        self.min_value = min_value
        self.max_value = max_value
        self.max_rank = max_rank
        self.normalizer = normalizer
    """
        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian samples according to mean/std of column
            "sparse": process as a sparse matrix

    """
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

    def _check_missing_value_mask(self, missing):
        if not missing.any():
            raise ValueError("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            X[missing_col, col_idx] = fill_values

    def _preprocess_sparse(self, X, missing_mask):
        rows, cols, shape = missing_mask
        m, n = shape
        J = self.max_rank
        V = np.zeros((m,self.max_rank))
        U = self.random.normal(size=(n*J))
        U = U.reshape(n,J)
        U = np.linalg.svd(U)[0] # takes the U component of the SVD of old U
        D_sq = np.ones(J) # we call it D_sq because A=UD and B=VD and AB'=U D_sq V^T
        return np.array([U, D_sq, V]) # convert to np.array to allow for copying in solver step

    def fill(
            self,
            X,
            missing_mask,
            inplace=False):
        """
        Parameters
        ----------
        X : np.array
            Data array containing NaN entries

        missing_mask : np.array
            Boolean array indicating where NaN entries are

        inplace : bool
            Modify matrix or fill a copy
        """
        if not inplace:
            X = X.copy()

        fill_method = self.fill_method

        if fill_method not in ("zero", "mean", "median", "min", "random", "sparse"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)
        elif fill_method == "sparse":
            X = self._preprocess_sparse(X, missing_mask)
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        if self.fill_method == 'sparse':
            x_dense_shape = X.shape
            x_true_rows = X.row
            x_true_cols = X.col
            missing_mask = x_true_rows, x_true_cols, x_dense_shape
            return X, missing_mask 
        else:
            X = np.asarray(X)
            if X.dtype != "f" and X.dtype != "d":
                X = X.astype(float)

            self._check_input(X)
            missing_mask = np.isnan(X)
            self._check_missing_value_mask(missing_mask)
            return X, missing_mask

    def normalize_input_columns(self, X, inplace=False):
        if not inplace:
            X = X.copy()
        column_centers = np.nanmean(X, axis=0)
        column_scales = np.nanstd(X, axis=0)
        column_scales[column_scales == 0] = 1.0
        X -= column_centers
        X /= column_scales
        return X, column_centers, column_scales

    def clip(self, X):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """
        X = np.asarray(X)
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def project_result(self, X):
        """
        First undo normaliztion and then clip to the user-specified min/max
        range.
        """
        X = np.asarray(X)
        if self.normalizer is not None:
            X = self.normalizer.inverse_transform(X)
        return self.clip(X)

    def solve(self, X, missing_mask, X_original=None):
        """
        Given an initialized matrix X and a mask of where its missing values
        had been, return a completion of X.
        """
        raise ValueError("%s.solve not yet implemented!" % (
            self.__class__.__name__,))

    def single_imputation(self, X):
        X_original, missing_mask = self.prepare_input_data(X)
        X = X_original.copy()
        X_filled = self.fill(X, missing_mask, inplace=True)
        if self.fill_method != 'sparse':
            observed_mask = ~missing_mask
            X_result = self.solve(X_filled, missing_mask)
            if self.normalizer is not None:
                X = self.normalizer.fit_transform(X)
        else:
            X_result_svd = self.solve(X_filled, missing_mask, X_original=X_original)
            #X_result = X_result_svd[0]*np.diag(X_result_svd[1])*X_result_svd[2]
            X_result = X_result_svd
       # if not isinstance(X_filled, np.ndarray):
       #     raise TypeError(
       #         "Expected %s.fill() to return NumPy array but got %s" % (
       #             self.__class__.__name__,
       #             type(X_filled)))
        
       # if not isinstance(X_result, np.ndarray):
       #     raise TypeError(
       #         "Expected %s.solve() to return NumPy array but got %s" % (
       #             self.__class__.__name__,
       #             type(X_result)))

        #TODO - reinstating X_result = self.project_result(X=X_result)
        #TODO - reinstating X_result[observed_mask] = X_original[observed_mask]
        return X_result

#    def multiple_imputations(self, X):
#        """
#        generate multiple imputations of the same incomplete matrix
#        """
#        return [self.single_imputation(X) for _ in range(self.n_imputations)]
#
#    def complete(self, X):
#        """
#        expects 2d float matrix with NaN entries signifying missing values
#
#        returns completed matrix without any NaNs.
#        """
#        imputations = self.multiple_imputations(X)
#        if len(imputations) == 1:
#            return imputations[0]
#        else:
#            return np.mean(imputations, axis=0)

