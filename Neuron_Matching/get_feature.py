# given dataset, generate feature for matching
import numpy as np
import ctypes
from sklearn.neighbors import KDTree


def get_feature(x, y, k0=10, scale=1):
    x = x.astype(np.double)
    y = y.astype(np.double)
    # x = np.random.rand(3,2)
    # y = np.array([[2,7],[3,2],[1,4]])
    x_ = x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    y_ = y.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    n1, n2 = len(x), len(y)

    # dll = ctypes.cdll.LoadLibrary(r"G:\item\neuron_matching_v3\soma_matching\methods\compute_feature\f1.so")
    dll = ctypes.cdll.LoadLibrary("./feature.so")
    size_graph1 = len(x) * (len(x)-1) * (len(x) -2) //6
    graph1 = np.zeros([size_graph1,3]).astype(np.uint32)
    graph1_ = graph1.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
    dll.get_graph(n1, graph1_)


    feature1 = np.zeros([size_graph1,3]).astype('float')
    feature1_ = feature1.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    feature2 = np.zeros([n2**3,3]).astype('float')
    feature2_ = feature2.ctypes.data_as(ctypes.POINTER(ctypes.c_double))



    dll.computeFeature(x_, n1, y_, n2, graph1_, len(graph1), feature1_, feature2_)

    kdt = KDTree(feature2, leaf_size=30, metric='euclidean')
    distance, index = kdt.query(feature1, k=k0, return_distance = True)
    # print('*'*100)

    i = index % n2
    j = (index % (n2**2)) // n2
    k = index // (n2**2)

    indH3 = np.tile(graph1*n2, len(index[0])).reshape(-1,3)
    indH3 = indH3 + np.c_[k.reshape(-1,1), j.reshape(-1,1), i.reshape(-1,1)]
    valH3 = np.exp(-distance.reshape(-1)/scale)

    return indH3, valH3

if __name__ == "__main__":
    x = np.random.rand(10,2)
    y = x
    index, score = get_feature(x, y, 500, 1e-2)
    print(index)