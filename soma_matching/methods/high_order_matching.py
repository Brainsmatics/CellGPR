import sys
sys.path.append(r'G:\item\neuron_matching_v3\soma_matching\methods')
from get_feature import get_feature
from tensor_matching import tensor_matching
import numpy as np


def high_order_matching(x, y, k0, scale):
    n1, n2 = len(x), len(y)

    indH3, valH3 = get_feature(x, y, k0, scale = 0.01)

    index, score = tensor_matching(indH3, valH3, n1, n2)

    return index, score

if __name__ == "__main__":
    data2 = np.random.rand(10,2)
    data1 = data2[5:]
    index, score = high_order_matching(data1, data2, 500, 1e-2)
    print(index)
    print(score)

    data1 = data2[5:] +np.random.rand(5, 2) * 0.05
    index, score = high_order_matching(data1, data2, 500, 1e-2)
    print(index)
    print(score)
