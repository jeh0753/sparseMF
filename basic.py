import numpy as np

class SparseMF:
    def __init__(self, mat, num_latent=2, n_iterations=100):
        self.U = np.ones((mat.shape[0], num_latent))
        self.V = np.ones((num_latent, mat.shape[1]))
        self.mat = mat
        self.m_shape = self.mat.shape
        self.M = self.U.dot(self.V)
        self.U_cursor = [0,0]
        self.V_cursor = [0,0]
        self.cursor = ['U', [0,0]]
        self.num_latent = num_latent
        self.n_iterations = n_iterations

    
    def optimize_x(self, C, D, y):
        '''
        Derivative of the cost function is similar for both scenarios, and differences are encoded in the building of C and D. 
        Linear Algebra equation for partial derivative of the cost function for X or Y minimization:
        (Yi-UiVi)Ci / Ci**2
        '''
        nan_index = np.argwhere(np.isnan(y))
        y = np.delete(y, nan_index)
        C = np.delete(C, nan_index)
        D = np.delete(D, nan_index)
        return np.sum(C.dot(y-D))/np.sum(C**2)

    
    def move_cursor(self):
        u_dims = (self.m_shape[0], self.num_latent)
        v_dims = (self.num_latent, self.m_shape[1])

        if self.cursor[0] == 'U':
            self.cursor[0] = 'V'
            self.cursor[1] = self.V_cursor

            if self.U_cursor[1] < u_dims[1] - 1:
                self.U_cursor[1] += 1

            elif self.U_cursor[0] < u_dims[0] - 1:
                self.U_cursor[0] +=1
                self.U_cursor[1] = 0

            else:
                self.U_cursor = [0,0]


        elif self.cursor[0] == 'V':
            self.cursor[0] = 'U'
            self.cursor[1] = self.U_cursor

            if self.V_cursor[1] < v_dims[1] - 1:
                self.V_cursor[1] += 1

            elif self.V_cursor[0] < v_dims[0] - 1:
                self.V_cursor[0] +=1
                self.V_cursor[1] = 0

            else:
                self.V_cursor = [0,0]
        

    def update(self):
        if self.cursor[0] == 'U':
            C = self.V[self.cursor[1][1], :]
            u_row = self.U[self.cursor[1][0], :] 
            u_row[self.cursor[1][1]] = 0
            D = np.expand_dims(u_row, axis=0).dot(self.V)
            y = self.mat[self.cursor[1][0], :]
            self.U[self.cursor[1][0], self.cursor[1][1]] = self.optimize_x(C, D, y)
            self.M[self.cursor[1][0], :] = self.U[self.cursor[1][0], :].dot(self.V) 

        elif self.cursor[0] == 'V':
            C = self.U[:, self.cursor[1][0]]
            v_col = self.V[:, self.cursor[1][1]]
            v_col[self.cursor[1][0]] = 0
            D = self.U.dot(np.expand_dims(v_col, axis=1))
            y = self.mat[:, self.cursor[1][1]]
            self.V[self.cursor[1][0], self.cursor[1][1]] = self.optimize_x(C, D, y)
            self.M[:, self.cursor[1][1]] = self.U.dot(self.V[:,self.cursor[1][0]]) 

        self.move_cursor()


    def fit(self):
        for i in xrange(self.n_iterations):
            self.update()


    def score(self):
        return "RMSLE Score: ", np.sum((model.M-model.mat)**2)


if __name__ == '__main__':
    mat = np.array([[5,2,4,4,3],[3,1,2,4,1],[2,np.nan,3,1,4],[2,5,4,3,5],[4,4,5,4,np.nan]])
    model = SparseMF(mat)
    print model.U
    model.update()
    print model.U
