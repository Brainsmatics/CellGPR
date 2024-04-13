import numpy as np
import scipy.sparse.linalg as sla
from munkres import Munkres


class MPM_2:
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
        L1[:, 1], L1[:, 0] = np.where(E1 == 1)
        L2[:, 1], L2[:, 0] = np.where(E2 == 1)
        L1, L2 = L1.astype('int'), L2.astype('int')

        # 计算所有的边长
        G1 = data1[L1[:, 0]] - data1[L1[:, 1]]
        G2 = data2[L2[:, 0]] - data2[L2[:, 1]]
        G1 = np.sqrt(np.square(G1[:, 0]) + np.square(G1[:, 1]))
        G2 = np.sqrt(np.square(G2[:, 0]) + np.square(G2[:, 1]))
        G1 = G1.reshape([data1.shape[0], data1.shape[0]])
        G2 = G2.reshape([data2.shape[0], data2.shape[0]])

        G1 = np.tile(G1, [data2.shape[0], data2.shape[0]])
        G2 = np.kron(G2, np.ones([data1.shape[0], data1.shape[0]]))

        M = np.square(G1 - G2)
        M = np.exp(-M / self.theta)
        for i in range(M.shape[0]):
            M[i, i] = 0

        return M

    def remove_confict(self, affinity_mat, n1, n2):
        vec1 = [[i for i in range(n1)] for j in range(n2)]
        vec1 = np.array(vec1).reshape(-1,1)
        vec2 = np.ones([affinity_mat.shape[0],1])
        mat1_1 = vec1.dot(vec2.T)
        mat1_2 = mat1_1.T
        vec3 = [[i for j in range(n1)] for i in range(n2)]
        vec3 = np.array(vec3).reshape(-1,1)
        mat2_1 = vec3.dot(vec2.T)
        mat2_2 = mat2_1.T
        p1 = np.multiply((mat1_1-mat1_2).astype('bool'), (mat2_1-mat2_2).astype('bool'))
        affinity_mat = np.multiply(affinity_mat, p1)
        #
        # for i in range(affinity_mat.shape[0]):
        #     for j in range(affinity_mat.shape[1]):
        #         if i == j:
        #             continue
        #         n1_1 = i%n1
        #         n1_2 = j%n1
        #         n2_1 = i//n1
        #         n2_2 = j//n1
        #         if n1_1 == n1_2 or n2_1 == n2_2:
        #             affinity_mat[i, j] = 0
        return affinity_mat

        # for i in range()

    def get_iteration(self, affinity_mat, prev_score, n1, n2):
        cur_score = np.zeros(prev_score.shape)
        # p1 = np.array([i for j in range(n2) for i in range(n1)])
        # p2 = np.array([i for i in range(n2) for j in range(n1)])
        for i in range(n1):
            for a in range(n2):
                temp = prev_score[a * n1 + i, 0] * affinity_mat[a * n1 + i, a * n1 + i]
                for j in range(n1):
                    p0 = []
                    if j == i:
                        continue
                    for b in range(n2):
                        if b == a:
                            continue
                        p0.append(prev_score[b * n1 + j, 0] * affinity_mat[a * n1 + i, b * n1 + j])
                    temp += max(p0)
                cur_score[a * n1 + i] = temp
        return cur_score

    def get_iteration1(self, affinity_mat, prev_score, n1, n2):
        # cur_score = np.zeros(prev_score.shape)
        # prev_mat = np.zeros([prev_score.shape[0], prev_score.shape[0]])
        # for i, t in enumerate(prev_score):
        #     prev_mat[i, i] = t
        prev_mat = np.multiply(prev_score, np.eye(prev_score.shape[0]))
        mat0 = np.array(affinity_mat).dot(prev_mat)
        # p = [[max(mat0[i,j::n1]) for j in range(n1)] for i in range(prev_score.shape[0])]
        p0 = np.array([])
        for j in range(n1):
            p0 = np.append(p0, np.max(mat0[:, j::n1], 1))
            # p1 = np.max(mat0[:, j::n1], 1)
        return np.sum(p0.reshape(n1,-1), 0)
        # p0 = p0.reshape(n1, -1).T
        # cur_score = np.sum(p0, 1)
        # return cur_score


    def method(self, affinity_mat):
        # eigenValue, eigenVector = np.linalg.eig(affinity_mat, )
        iter_Max = 300
        nMatch = affinity_mat.shape[0]
        prev_score = np.ones([nMatch, 1]) / nMatch
        prev_score2 = prev_score

        bCont = 1
        iter_i = 0
        thresConvergence = nMatch * np.linalg.norm(affinity_mat, ord=1) * 1e-6

        # la = prev_score.T.dot(affinity_mat.dot(prev_score))
        cur_score = np.zeros([nMatch, 1])

        while bCont and iter_i < iter_Max:
            iter_i += 1
            # cur_score = self.get_iteration(affinity_mat, prev_score, self.data1.shape[0], self.data2.shape[0])
            # print('in get_iteration, cur_score = {}'.format(cur_score))
            cur_score = self.get_iteration1(affinity_mat, prev_score, self.data1.shape[0], self.data2.shape[0])
            # print('in get_iteration1 cur_score = []'.format(cur_score1))
            sum_cur_score = np.sqrt(np.sum(np.square(cur_score)))
            cur_score = cur_score / sum_cur_score if sum_cur_score > 0 else cur_score
            diff1 = np.sum(np.square(cur_score - prev_score))
            diff2 = np.sum(np.square(cur_score - prev_score2))
            diff_min = min(diff1, diff2)
            bCont = 0 if diff_min < thresConvergence else 1
            prev_score2 = prev_score
            prev_score = cur_score
        # print(iter_i)
        return cur_score

    def get_score(self, affinity_mat, X):
        return X.T.dot(affinity_mat).dot(X)

    def run(self):
        n1, n2 = self.data1.shape[0], self.data2.shape[0]
        affinity_mat = self.get_affinity_mat()
        affinity_mat = self.remove_confict(affinity_mat, n1, n2)
        X = self.method(affinity_mat)
        X = -X
        X -= X.min()
        # print(X)
        X = X.reshape([n2, n1])
        m = Munkres()
        if n1 <= n2:
            indexes = m.compute(X.copy().T)

        else:
            indexes = m.compute(X)
            indexes = [(j, i) for i, j in indexes]
        X0 = np.zeros([n1 * n2, 1])
        for i, j in indexes:
            X0[j * n1 + i] = 1
        match_score = self.get_score(affinity_mat, X0)

        return indexes, match_score

    def test(self):
        n1, n2 = self.data1.shape[0], self.data2.shape[0]
        affinity_mat = self.get_affinity_mat()
        print(affinity_mat)
        affinity_mat = self.remove_confict(affinity_mat, n1, n2)
        print(affinity_mat)


if __name__ == '__main__':
    count0 = 0
    count1 = 0
    # for i in range(5):
    # data1 = np.load('./data1.npy')
    # data2 = data1[:50]
    # data2 = data2 + np.load('./data3.npy')*0.1
    data2 = np.random.rand(10,2)
    data1 = data2[5:]
    a = MPM_2(data1, data2, 1)
    index, match_score = a.run()
    print(index)
    # count0 += len(index)
    # for p, q in index:
    #     if p != q:
    #         count1 += 1
    # print('error rate is {}'.format(count1 / count0))
    # print(match_score)
    # data1 = np.array([[1,0],[0,1]])
    # data2 = np.array([[0,0], [1,1]])
    # a = MPM_2(data2, data1, 0.1)
    # a.test()
    # data1 = np.random.rand(50, 2)
    # np.save('./data3.npy', data1)