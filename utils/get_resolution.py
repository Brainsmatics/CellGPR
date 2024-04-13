import numpy as np
from utils.get_data import *
import os

def get_euclidean(data1, data2):
    # data1 nx2(3) data2 2(3)
    data1, data2 = np.array(data1), np.array(data2)
    assert data1.shape[1] == data2.shape[0]
    data1 = data1 - data2
    data1 = np.sqrt(np.sum(np.square(data1), 1))
    return data1


def get_resolution(data_3d, data_2d):
    assert len(data_3d) == len(data_2d)
    data_3d, _ = data_regularzation(data_3d)
    data1 = data_3d[:,:2]
    t1, t2 = 0, 0
    while len(data1) >1:
        t1 += sum(get_euclidean(data1[:-1], data1[-1]))
        t2 += sum(get_euclidean(data_2d[:-1], data_2d[-1]))
        data1 = data1[:-1]
        data_2d = data_2d[:-1]
    return t1/t2

if __name__ == '__main__':
    data_path = '../data/202279/results'
    for filename in os.listdir(data_path):
        if filename.endswith('_tp.npy'):

            data_2d = np.load(os.path.join(data_path, filename))
            data_3d = get_swc_data(os.path.join(data_path, filename[:-7]+'.swc'))
            print(filename, get_resolution(data_3d, data_2d))

    # data_3d = get_swc_data('../data/202279/results/L23_site1.swc')
    # data_2d = np.load('../data/202279/results/L23_site1_tp.npy')
    #
    # print(get_resolution(data_3d, data_2d))