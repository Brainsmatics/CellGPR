import numpy as np
from get_feature import get_feature
from tensor_matching import tensor_matching

x = np.random.rand(100, 2)
y = x[::-1]*1.5

x = x[:10]


n1, n2 = len(x), len(y)

indH3, valH3 = get_feature(x, y, k0=500, scale = 0.01)

index = tensor_matching(indH3, valH3, n1, n2)

print(index)