import numpy as np
import scipy.sparse.linalg as sla
from munkres import Munkres

class SM:
    def __init__(self, data1, data2, theta):
        self.data1 = data1
        self.data2 = data2
        assert data1.shape[1] == 2
        assert data2.shape[1] == 2
        self.theta = theta

    def get_affinity_mat(self):
        data1, data2 = self.data1, self.data2
        E1 = np.ones([data1.shape[0], data1.shape[0]])
        E2 = np.ones([data2.shape[0], data2.shape[0]])
        L1, L2 = np.zeros([np.square(data1.shape[0]), 2]), np.zeros([np.square(data2.shape[0]), 2])
        # 得到所有的边
        L1[:, 1], L1[:,0] = np.where(E1 == 1)
        L2[:, 1], L2[:,0] = np.where(E2 == 1)
        L1, L2 = L1.astype('int'), L2.astype('int')

        # 计算所有的边长
        G1 = data1[L1[:,0]] - data1[L1[:,1]]
        G2 = data2[L2[:,0]] - data2[L2[:,1]]
        G1 = np.sqrt(np.square(G1[:,0]) + np.square(G1[:,1]))
        G2 = np.sqrt(np.square(G2[:,0]) + np.square(G2[:,1]))
        G1 = G1.reshape([data1.shape[0], data1.shape[0]])
        G2 = G2.reshape([data2.shape[0], data2.shape[0]])

        G1 = np.tile(G1, [data2.shape[0], data2.shape[0]])
        G2 = np.kron(G2, np.ones([data1.shape[0], data1.shape[0]]))

        M = np.square(G1-G2)
        M = np.exp(-M/self.theta)
        for i in range(M.shape[0]):
            M[i,i] = 0

        return M

    def method(self, affinity_mat):
        # eigenValue, eigenVector = np.linalg.eig(affinity_mat, )
        D, V = sla.eigs(affinity_mat, k=1, which='LM')
        V = np.abs(V.astype('float'))
        return V

    def get_score(self, affinity_mat, X):
        return X.dot(affinity_mat).dot(X.T)


    def run(self):
        affinity_mat = self.get_affinity_mat()
        X = self.method(affinity_mat)
        X = -X
        X -= X.min()
        # print(X)
        X = X.reshape([self.data2.shape[0], self.data1.shape[0]])
        m = Munkres()

        indexes = m.compute(X.T)
        # indexes = [(j,i) for i, j in indexes]

        return indexes








if __name__ == '__main__':
    data1 = np.array([[0,1], [0,0], [2,0]])
    # data2 = np.array([[0,0], [1,0], [1,2]])
    data2 = np.array([[0,0], [1,0], [0,-2], [3,3]])
    a = SM(data1, data2, 1)
    indexes = a.run()
    print(indexes)
