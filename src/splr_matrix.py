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


import numpy as np
from numpy.random import RandomState
from scipy.sparse import csc_matrix


class SPLR(object):
    ''' A new sparse matrix class, Sparse Plus Low Rank as described by Trevor Hastie et al. in their 2009 paper 'Spectral Regularization Algorithms for Learning Large Incomplete Matrices'.

    x: scipy.spares matrix
        The base sparse matrix.
    a: 2D array, optional
        The U portion of an SVD matrix
    b: 2D array, optional
        the VD portion of an SVD matrix
    '''
    def __init__(self, x, a=None, b=None):
        self.x = x
        self.a = a
        self.b = b

        x_dims = x.shape

        if a is None:
            self.b = None

        if b is None:
            self.a = None
        else:
            a_dims = a.shape
            b_dims = b.shape
            if a_dims[0] != x_dims[0]:
                raise ValueError("number of rows of x not equal to number of rows of a")

            if b_dims[0] != x_dims[1]:
                raise ValueError("number of columns of x not equal to number of rows of b")

            if a_dims[1] != b_dims[1]:
                raise ValueError("number of columns of a not equal to number of columns of b")

    def r_mult(self, other):
        """Left Multiplication
        This is equivalent to self.dot(other)
        """
        result = self.x.dot(other)
        result = result

        if self.a is not None:
            b_mult = self.b.T.dot(other)
            ab_mult = self.a.dot(b_mult)
            result += ab_mult

        return result

    def l_mult(self, other):
        """Left Multiplication
        This is equivalent to other.dot(self)
        """
        result = csc_matrix(other).dot(self.x) # conversion necessary for dot to be called successfully
        result = result.toarray()

        if self.a is not None:
            ab_mult = other.dot(self.a)
            ab_mult = ab_mult.dot(self.b.T)
            result += ab_mult

        return result
