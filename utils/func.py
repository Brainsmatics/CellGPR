import numpy as np
from parameter import *
from munkres import Munkres
from sklearn import decomposition

def sum_distanse(data):
    res = 0
    while len(data)>1:
        t1 = data[0]
        data = data[1:]
        s0 = np.sqrt(np.sum(np.square(data-t1), 1))
        res += sum(s0)
    return res

def get_affine_mat(data1, data2):
    moving_points, fixed_points = np.array(data1), np.array(data2)
    moving_points = np.c_[moving_points, np.ones((moving_points.shape[0], 1))]
    fixed_points = np.c_[fixed_points, np.ones((fixed_points.shape[0], 1))]
    mat1, mat2 = np.zeros((3, 3)), np.zeros((3, 3))
    for i in range(fixed_points.shape[0]):
        mat1 += fixed_points[i].reshape(fixed_points[i].shape[0], 1) * moving_points[i]
        mat2 += moving_points[i].reshape(fixed_points[i].shape[0], 1) * moving_points[i]
    affine_translation = np.dot(mat1, np.linalg.inv(mat2))
    print(np.linalg.det(affine_translation))
    return affine_translation, affine_translation.dot(moving_points.T).T[:,:2]

def data_regularzation(data):
    # 通过主成分分析将数据旋转至z轴最薄，返回旋转结果以及变换矩阵
    pca = decomposition.PCA(n_components=3)
    pca.fit_transform(data)
    mat1 = pca.components_
    # mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
    mat1 = np.array([-mat1[0], mat1[1], mat1[2]]) if np.linalg.det(mat1) < 0 else mat1
    mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
    data = data.dot(mat1.T)
    r = np.zeros(3) - np.mean(data, 0)
    return data+r, mat1, r

def get_closest_points(data1, data2, distance_thread):
    assert len(data1) >= len(data2)
    s = np.array([])
    for data in data1:
        t = data2 - data
        s0 = np.sqrt(np.sum(np.square(t),1))
        # if len(np.where(s0 <0.5)[0]) > 0:
        #     print(s0)
        # print(s0)
        s = np.append(s, s0)
    s = s.reshape([len(data1), len(data2)])
    # print(s)
    m = Munkres()
    res = m.compute(s.copy().T)
    print(res)
    res0 = []
    for t, (i, j) in enumerate(res):
        # print(j,  i, s[j,i])
        if s[j,i] <= distance_thread:
            res0.append((j,i))
            # res.pop(t)
    for i, j in res0:
        print(s[j, i])
    return res0

def get_closest_points1(data1, data2, distance_thread):
    assert len(data1) >= len(data2)
    # remove outliers
    index0 = []
    for i, data in enumerate(data2):
        t = data1 - data
        s0 = np.sqrt(np.sum(np.square(t),1))
        if min(s0) > distance_thread:
            continue
        index0.append(i)
    data20 = data2[index0]

    s = np.array([])
    for data in data1:
        t = data20 - data
        s0 = np.sqrt(np.sum(np.square(t),1))
        s = np.append(s, s0)
    s = s.reshape([len(data1), len(data20)])
    # print(s)
    m = Munkres()
    res = m.compute(s.copy().T)
    # print(res)
    res0 = []
    for t, (i, j) in enumerate(res):
        if s[j,i] <= distance_thread:
            res0.append((index0[i], j))
            # print(s[j,i])
            # res.pop(t)
    return res0






def get_recover(data1, data2, m_data1, m_data2):
    assert len(m_data1) == len(m_data2)
    m_data1, mat, r = data_regularzation(m_data1)
    data1 = data1.dot(mat.T) + r
    flag = (data1[:,0] > -250) & (data1[:,0] < 250) & (data1[:,1] > -250) \
           & (data1[:,1] < 250) & (data1[:,2] > -30) & (data1[:,2] < 30)
    data1 = data1[flag]
    translation, _ = get_affine_mat(m_data1[:,:2], m_data2)
    data1 = np.c_[data1[:,:2], np.ones([len(data1),1])]
    data1 = data1.dot(translation.T)[:,:2]
    res = get_closest_points(data1, data2)
    index1 = np.where(flag == True)[0]
    res = [(i, index1[j]) for i, j in res]
    scale = np.linalg.det(translation)

    return res, scale

# def k_similarity(data1, data2, k):
#     '''
#
#     '''





if __name__ == '__main__':
    # data1 = np.array([1,2])
    # from utils.get_data import *
    # # data = get_csv_data('../testing_data/dataset1/matching1.csv', 2)
    # # data1, data2 = data[::2], data[1::2]
    # # mat, data1_trans = get_affine_mat(data1, data2)
    # # print(data2-data1_trans)
    # data1 = get_swc_data('../data/202279/results/soma5.swc')
    # data2 = get_csv_data('../data/202279/2p/L5_site3.csv', 2)
    # m_data2 = np.load('../data/202279/results/L5_site3_tp.npy')
    # m_data1 = get_swc_data('../data/202279/results/L5_site3.swc')
    # res = get_recover(data1, data2, m_data1, m_data2)
    # print(res)
    # r1 = np.load('../temp/data1.npy')
    # r2 = np.load('../temp/data2.npy')
    # res = get_closest_points(r1, r2, distance_thread=15)
    # print(res)
    data1 = np.random.rand(15,2)*100
    data2 = data1[:4][::-1]
    data2 = np.r_[[[0,0], [200,200]], data2]
    res = get_closest_points1(data1, data2, distance_thread = 500)
    print(res)

