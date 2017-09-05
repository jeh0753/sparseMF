import numpy as np 
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
    

class SBiScale(object):
    def __init__(self, maxit=20, thresh=1e-9, row_center=True, row_scale=False, col_center=True, col_scale=False, trace=False):
        self.maxit = maxit
        self.thresh = 1e-9
        self.row_center = row_center
        self.row_scale = row_scale
        self.col_center = col_center
        self.col_scale = col_scale
        self.trace = trace
        self.x = None
        self.m = None
        self.n = None
        self.a = None
        self.b = None
        self.tau = None 
        self.gamma = None 
        self.xhat = None 
        self.critmat = []
    
    def _prepare_suvc(self):
        a = self.a.copy()
        a = a.reshape(-1,1)
        b = self.b.copy()
        b = b.reshape(-1,1)
        a = np.hstack((a, np.ones(a.shape[0]).reshape(-1,1)))
        b = np.hstack((np.ones(b.shape[0]).reshape(-1,1), b))
        return a, b 
    
    def _pred_one(self, u, v, row, col):
        u_data = np.expand_dims(u[row,:], 0)
        return float(u_data.dot(v[col, :].T))

    def _c_suvc(self, u, v, irow, icol):
        nomega = len(irow)
        res = np.zeros(nomega)
        targets = zip(irow, icol)
        for idx, (r,c) in enumerate(targets):
            res[idx] = self._pred_one(u, v, r, c)
        return res

    def _center_scale_I(self): 
        x = self.x.data
        a, b = self._prepare_suvc()
        coo_x = coo_matrix(self.x)
        irow = coo_x.row
        icol = coo_x.col
        suvc1 = self._c_suvc(a, b, irow, icol)  
        suvc2 = self._c_suvc(self.tau.reshape(-1,1), self.gamma.reshape(-1,1), irow, icol)
        self.xhat.data = (x-suvc1) / suvc2
        return self
        
    def _col_sum_along(self, a, x):
        x = (self.x != 0)
        a = csc_matrix(a.T)
        return a.dot(x).toarray()

    def _row_sum_along(self, b, x):
        x = (self.x != 0)
        return x.dot(b)

    def _add_variables(self, x):
        self.x = x
        self.m = x.shape[0]
        self.n = x.shape[1]
        self.a = np.zeros(self.m)
        self.b = np.zeros(self.n)
        self.tau = np.ones(self.m)
        self.gamma = np.ones(self.n)
        self.xhat = self.x.copy()
        return self

    def fit(self, x):
        self._add_variables(x)
        self._center_scale_I()
        for i in xrange(self.maxit):
            # Centering
            ## Column mean
            if self.col_center:
                colsums = np.sum(self.xhat, axis=0)
                gamma_by_sum = np.multiply(colsums,(self.gamma))
                dbeta = gamma_by_sum / self._col_sum_along(1 / self.tau, self.x)
                self.b = self.b + dbeta
                self.b[np.isnan(self.b)] = 0
                self._center_scale_I()
            else:
                dbeta = 0
            
            ## Row Mean
            if self.row_center:
                rowsums = np.sum(self.xhat, axis=1).T
                tau_by_sum = np.multiply(self.tau, rowsums)
                dalpha = tau_by_sum / self._row_sum_along(1 / self.gamma, self.x)
                self.a = self.a + dalpha
                self.a[np.isnan(self.a)] = 0
                self._center_scale_I()

            else:
                dalpha = 0 
        
            #Leaving out scaling for now; does not appear to be used for my purposes
            dalpha[np.isnan(dalpha)] = 0
            dbeta[np.isnan(dbeta)] = 0
            convergence_level = np.square(dalpha).sum() + np.square(dbeta).sum()
            self.critmat.append([i + 1, convergence_level])
            if convergence_level < self.thresh:
                break

        # Complete solution
        self.xhat.row_center = np.ravel(self.a)
        self.xhat.col_center = np.ravel(self.b)
        self.xhat.row_scale = np.ravel(self.tau)
        self.xhat.col_scale = np.ravel(self.gamma)
        self.xhat.critmat = self.critmat
        
        result = self.xhat
        return result

    def transform(self, X, row_id, col_id, prediction):
        ''' Takes a single predicted value, and returns the scaled and centered version.'''
        scaled = prediction * X.row_scale[row_id] * X.col_scale[col_id]
        centered = scaled + X.row_center[row_id] + X.col_center[col_id]
        return centered

