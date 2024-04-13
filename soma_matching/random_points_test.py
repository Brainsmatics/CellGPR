import numpy as np
from SM.SM import SM
from MPM.MPM import MPM
from show_results import show_results
import multiprocessing
from multiprocessing import Pool, Process

num = 10
outliers = 40
rot = True
noise_scale = 0
scale = 0.5


data1 = (np.random.rand(num, 2)-0.5)*2
noise = np.random.randn(num, 2) * noise_scale
data2 = data1 + noise

# 进行旋转
if rot:
    angle = np.random.rand() * np.pi *2
    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],
                        [np.sin(angle), np.cos(angle)]])
    data2 = data2.dot(rot_mat)



# 加入干扰点
data0 = (np.random.rand(outliers, 2)-0.5)*2
data2 = np.r_[data2, data0]


# 打乱顺序
t = np.arange(data2.shape[0])
np.random.shuffle(t)
data2 = data2[t]
# print(t)

method = MPM(data1, data2, scale)
indexes = method.run()
# print(indexes)
# print(data1)
# print(data2)
plot0 = show_results(data1, data2, indexes)
plot0.plot_points()
# print(t)
# 计算准确率
t0 = np.array([j for i, j in indexes])
t1 = np.array([int(np.where(t == i)[0]) for i,j in indexes])
# print(t0)
# print(t)
accuracy = sum(t1-t0 == 0) /len(t0)
print('accuracy is {}'.format(accuracy))






