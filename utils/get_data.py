import pandas as pd
import numpy as np
from sklearn import decomposition


def get_swc_data(swc_path):
    f = open(swc_path, 'r')
    points = []
    for line in f.readlines():
        # print(line)
        info = line.split(' ')
        x, y, z = float(info[2]), float(info[3]), float(info[4])
        points.append([x,y,z])
    return np.array(points)

def get_csv_data(csv_path, dim):
    data = []
    df = pd.read_csv(csv_path)
    if dim == 2:
        for i in range(df.shape[0]):
            data.append([df.loc[i, 'X'], df.loc[i, 'Y']])
    else:
        for i in range(df.shape[0]):
            data.append([df.loc[i, 'X'], df.loc[i, 'Y'], df.loc[i, 'Slice']])

    return np.array(data)

def write_swc(data, output_path):
    with open(output_path, 'w') as f:
        for i, data0 in enumerate(data):
            str0 = '{} 1 {} {} {} 1 -1\n'.format(i + 1, data0[0], data0[1], data0[2])
            f.write(str0)

# def data_regularzation(data):
#     # 通过主成分分析将数据旋转至z轴最薄，返回旋转结果以及变换矩阵
#     pca = decomposition.PCA(n_components=3)
#     pca.fit_transform(data)
#     mat1 = pca.components_
#     # mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
#     mat1 = np.array([-mat1[0], mat1[1], mat1[2]]) if np.linalg.det(mat1) < 0 else mat1
#     mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
#     data = mat1.dot(data.T)
#     return data.T, mat1





