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

from six.moves import range
import numpy as np
from numpy.random import RandomState
from sklearn.utils.extmath import randomized_svd
from scipy.sparse import coo_matrix, csc_matrix, issparse, csr_matrix
from sklearn.metrics import mean_squared_error

from sparse_biscale import SBiScale
from sparse_solver import Solver
from splr_matrix import SPLR
    

class SoftImpute(Solver):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.
    """
    def __init__(
            self,
            shrinkage_value=0,
            convergence_threshold=1e-05,
            max_iters=100,
            max_rank=2, 
            n_power_iterations=8,
            center=True,
            verbose=True):
        """
        Parameters
        ----------
        shrinkage_value : float
            Value by which we shrink singular values on each iteration. If
            omitted then the default value will be the maximum singular
            value of the initialized matrix (zeros for missing values) divided
            by 100.

        convergence_threshold : float
            Minimum ration difference between iterations (as a fraction of
            the Frobenius norm of the current solution) before stopping.

        max_iters : int
            Maximum number of SVD iterations

        max_rank : int, optional
            Perform a truncated SVD on each iteration with this value as its
            rank.

        n_power_iterations : int
            Number of power iterations to perform with randomized SVD

        normalizer : bool
            Uses SBiScale(), if True. This centers and scales the data

        verbose : bool
            Print debugging info
        """
        Solver.__init__(
            self,
            min_value=None,
            max_value=None,
            max_rank=max_rank,
            normalizer=SBiScale() if center == True else None,
            sparse=True)
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.n_power_iterations = n_power_iterations
        self.verbose = verbose
        self.m = None
        self.n = None
        self.X = None
        self.svd = None
        self.X_splr = None

    def _fnorm(self, SVD_old, SVD_new):
        ''' U, S, V is the order of SVD matrices. This function takes the Frobenius Norm of an SVD decomposed matrix. '''
        U_old, D_sq_old, V_old = SVD_old
        U_new, D_sq_new, V_new = SVD_new
        utu = D_sq_new.dot(U_new.T.dot(U_old))
        vtv = D_sq_old.dot(V_old.T.dot(V_new))
        uvprod = utu.dot(vtv).sum()
        sing_val_sumsq_old = (D_sq_old ** 2).sum()
        sing_val_sumsq_new = (D_sq_new ** 2).sum()
        norm = (sing_val_sumsq_old + sing_val_sumsq_new - (2 * uvprod)) / max(sing_val_sumsq_old, 1e-9) 
        return norm

    def _converged(self, SVD_old, SVD_new):
        norm = self._fnorm(SVD_old, SVD_new)
        return norm < self.convergence_threshold

    def _UD(self, U, D, n):
        ones = np.ones(n)
        return U * np.outer(ones, D)

    def _pred_sparse(self, row_id, col_id, X_svd):
        """This function predicts output for a single row id and column id pair. It returns prediction based on SVD regardless of whether the row, column pair existed in the original matrix"""
        U, s, V = X_svd
        V = V.T
        res = np.sum(U[row_id]*s*V[:,col_id])
        return res

    def _xhat_pred(self):
        """predicts x values for X original indices/columns"""
        row_ids, col_ids, _ = self.missing_mask
        x_svd = self.X_fill
        targets = zip(row_ids, col_ids)
        n_preds = len(targets)
        res = np.empty(n_preds)

        for idx, (r, c) in enumerate(targets):
            res[idx] = self._pred_sparse(r, c, x_svd)

        return res

    def _svd(self, X, max_rank=None):
        if max_rank:
            # if we have a max rank then perform the faster randomized SVD
            return randomized_svd(
                X,
                max_rank,
                n_iter=self.n_power_iterations)
        else:
            # perform a full rank SVD using ARPACK
            return np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)

    def _max_singular_value(self):
        # quick decomposition of X_filled into rank-1 SVD
        X_filled = self.X_fill
        return X_filled[1][0] #TODO - Replace with self.svd, or rename if we want it to apply for else cond.

    def _als_u_step(self):
        U, D_sq, V = self.X_fill

        B = (self.X_splr.l_mult(U.T)).T

        if self.shrinkage_value > 0:
            B = self._UD(B, D_sq / (D_sq + self.shrinkage_value), self.m)
        V, D_sq, _ = self._svd(B) # V is set to the U slot from V's SVD on purpose
        self.X_fill = U, D_sq, V
        return self 

    def _als_v_step(self):
        U, D_sq, V = self.X_fill

        A = self.X_splr.r_mult(V)

        if self.shrinkage_value > 0:
            A = self._UD(A, D_sq / (D_sq + self.shrinkage_value), self.n)

        U, D_sq, V_part = self._svd(A)
        V = V.dot(V_part.T) # just for computing the convergence criterion
        self.X_fill = U, D_sq, V
        return self

    def _als_cleanup_step(self):
        U, D_sq, V = self.X_fill
        A = self.X_splr.r_mult(V)
        U, D_sq, V_part = self._svd(A) 
        V = V.dot(V_part.T)
        D_sq = np.clip(D_sq - self.shrinkage_value, a_min=0, a_max=None) # this shrinks the singular values by lambda and clips them at zero
        self.X_fill = U, D_sq, V
        return self

    def _als_step(self):
        for i in range(self.max_iters):
            U, D_sq, V = self.X_fill
            U_old, D_sq_old, V_old = U.copy(), D_sq.copy(), V.copy()
            self._als_u_step()
            self._als_v_step()
            U, D_sq, V = self.X_fill
            converged = self._converged((U_old, D_sq_old, V_old), (U, D_sq, V))

            if converged:
                break

        if self.shrinkage_value > 0:
            self._als_cleanup_step() 

        return self

    def solve(self, X, X_original=None):
        """
        X : 3-d array
            X is a 3-d array composed of a U matrix, a D singular values array, and a V matrix.
        X_original: SPLR sparse matrix
            X_original is the training dataset passed in to the complete function, prior to being filled/processed. 
           
        Returns: 1D Array of numpy matrices 
            The SVD solution that best approximates the true completed X matrix. U x S x V.T is the order of the output.
        """
        self.X_fill = X
        self.X = X_original
        X_filled = X
        missing_mask = self.missing_mask

        max_singular_value = self._max_singular_value()
        J = self.max_rank

        if X_original is not None:
            self.n, self.m = self.X.shape
            x_res = self.X.copy()

        if self.verbose:
            print("[SoftImpute] Max Singular Value of X_init = %f" % (
                max_singular_value))

        if not self.shrinkage_value:
            # totally hackish heuristic: keep only components
            # with at least 1/50th the max singular value
            self.shrinkage_value = max_singular_value / 50.0

        shrinkage_value = self.shrinkage_value
        
        for i in range(self.max_iters):
            self.X_fill_old = self.X_fill
            U, Dsq, V = self.X_fill

            if i == 0:
                self.X_splr = SPLR(x_res)

            else:
                BD = self._UD(V, Dsq, self.m)
                x_hat = self._xhat_pred()
                x_res.data = X_original.data - x_hat 
                self.X_splr = SPLR(x_res, U, BD)

            self._als_step()
            converged = self._converged(self.X_fill_old, self.X_fill)

            if converged:
                break
            
        if self.verbose:
            print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))
        U, D_sq, V = self.X_fill
        A = self.X_splr.r_mult(V)
        U, D_sq, V_part = self._svd(A)
        V = self.X_splr.a.dot(V_part.T)
        D_sq = np.clip(D_sq - self.shrinkage_value, a_min=0, a_max=None)
        J = min((D_sq>0).sum(), J)
        return self.X_fill

    def _predict_one(self, row_id, col_id):
        """ A single row ID and column ID are input to produce a single prediction """
        irows, icols, _ = self.missing_mask
        existing = zip(irows, icols)

        if isinstance(row_id, (int, long)) and isinstance(col_id, (int, long)):
            if (row_id, col_id) in existing:
                prediction = self.X[row_id, col_id]
                if self.normalizer == None:
                    return prediction
                return self.normalizer.transform(self.X, row_id, col_id, prediction)

            elif col_id < self.m and row_id < self.n:
                prediction = self._pred_sparse(row_id, col_id, self.X_fill)
                if self.normalizer == None:
                    return prediction
                return self.normalizer.transform(self.X, row_id, col_id, prediction)
                
            else:
                if col_id >= self.m:
                    raise ValueError("Column index %s is out of range" % (col_id))
                if row_id >= self.n:
                    raise ValueError("Row index %s is out of range" % (row_id))

        else:
            raise ValueError("Input row_ids and col_ids must be integers.")

    def predict(self, row_ids, col_ids):
        """ Makes predictions for a given set of row_ids and col_ids.

        row_ids: array or integer
            Consecutive row IDs to be matched to a column ID array of equal length.
            Single integers are also accepted to produce a single prediction.
            
        col_ids: array or integer
            Consecutive column IDs to be matched to a row ID array of equal length.
            Single integers are also accepted to produce a single prediction.

        Returns: array or integer
            The expected rating for the given row_id and col_id pairs
        """
        if isinstance(row_ids, (int, long)) and isinstance(col_ids, (int, long)):
            return self._predict_one(row_ids, col_ids)

        elif len(row_ids) != len(col_ids):
            raise ValueError("Length of row list and column list must match")

        else:
            targets = zip(row_ids, col_ids)
            res = np.empty(len(targets))
            for idx, (r, c) in enumerate(targets): # TODO- this could be made into a matrix multiplication, which would add some speed
                res[idx] = self._predict_one(r, c)
            return res

    def eval(self):
        """
        Evaluate Root Mean Squared Error of Reconstructed X

        Returns: float
            The training RMSE of the model.
        """
        xhat = self._xhat_pred()
        x = self.X.data
        mse = mean_squared_error(x, xhat)
        return np.sqrt(mse)
                

if __name__ == '__main__':
    # The below code matches the example used in Trevor Hastie's vignette, and allows for a simple check that the output matches expectations.
    x =  np.array([0.86548894, -0.60041722, -0.71692924,  0.69655580,  1.23116102,  0.26644155,  0.01565179, -0.50331812, -0.34232368,  0.14486388,  0.17479031, -0.21190900,  0.55848392, -0.81026875, 0.06437356,  1.54375663, -0.82006429, -0.09754133, -0.13256942, -2.24087863])
    x_idx = np.array([0, 1, 2, 3, 4, 5, 0, 3, 4, 5, 0, 1, 3, 4, 2, 3, 4, 2, 4, 5]) 
    x_p = np.array([0, 6, 10, 14, 17, 20])
    xs = csc_matrix((x, x_idx, x_p), shape=(6,5))
    sf = SoftImpute(max_rank=3, shrinkage_value=.02, max_iters=50)
    sf.complete(xs)
    print(sf.predict(np.array([1,2]), np.array([1,1])))

