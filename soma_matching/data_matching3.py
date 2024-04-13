import multiprocessing
import random
from multiprocessing import Process, Queue

import numpy as np

# from methods.MPM import MPM
# from soma_matching.methods.MPM_2 import MPM_2
from soma_matching.methods.high_order_matching import high_order_matching
from utils.func import *
from utils.get_data import get_csv_data, get_swc_data
from utils.icp import best_fit_transform

'''
修正：加入变换矩阵的返回
'''

class data_matching:
    def __init__(self, data1, data2, step, thickness, theta):
        self.data1 = data1 #2d projection
        self.data2 = data2 # 3d data
        self.step = step
        self.thickness = thickness
        self.theta = theta

    def remove_outliers_2d(self, data1, data2, iter=500, thread=10):
        matching_nums = 0
        index0 = [False for i in range(len(data1))]
        scale = 1
        r0, t0 = np.eye(2), np.zeros([0,0])

        for i in range(iter):
            a = list(range(len(data1)))
            index = random.sample(a, 3)
            data10 = data1[index]
            data20 = data2[index]

            scale0 = sum_distanse(data10) / sum_distanse(data20)
            data20 = data20 * scale0
            if scale0 > 1.1 or scale0 < 0.9:
                continue

            T, r, t = best_fit_transform(data20, data10)
            data2_trans = scale0 * data2.dot(r.T) + t
            distance = data2_trans - data1
            distance = np.sqrt(np.square(distance[:, 0]) + np.square(distance[:, 1]))
            num1 = sum(distance < thread)
            if num1 > matching_nums:
                # print(num1)
                matching_nums = num1
                index0 = distance < thread
                scale = scale0
                r0, t0 = r, t

        return matching_nums, index0, 1 / scale, scale * r0.T, t0



    def get_max_matching(self, r_x, r_y, z_min):
        # print('r_x = {}, r_y = {}'.format(r_x, r_y))
        # print('%'*100)
        x, y = r_x * np.pi / 180, r_y * np.pi / 180
        mat_x = np.array([[1, 0, 0],
                          [0, np.cos(x), -np.sin(x)],
                          [0, np.sin(x), np.cos(x)]])
        mat_y = np.array([[np.cos(y), 0, np.sin(y)],
                          [0, 1, 0],
                          [-np.sin(y), 0, np.cos(y)]])
        rot_mat = mat_x.dot(mat_y)

        data2_rotate = self.data2.dot(rot_mat)

        data1_index, data2_index = 0, 0

        x_mean, y_mean = np.mean(self.data2, 0)[:2]
        z0 = np.array([x_mean, y_mean, z_min]).dot(rot_mat)[2]
        z1 = z0 + self.thickness
        data_2d_index = np.where((data2_rotate[:, 2] > z0) & (data2_rotate[:, 2] <= z1))[0]
        nums1 = 0
        match_score = 0
        scale = 1
        # r0, t0 = np.eye(2), np.zeros(2)
        if len(data_2d_index) >= 3:
            data2_2d_tmp = data2_rotate[data_2d_index]
            data2_2d_tmp = data2_2d_tmp[:, :2]
            # method = high_order_matching(self.data1, data2_2d_tmp, 500, self.theta)
            num_temp = len(data2_2d_tmp) ** 3
            index, _ = high_order_matching(self.data1, data2_2d_tmp, min(500, num_temp), self.theta)
            t0, t1 = [i for i, j in index], [j for i, j in index]
            data10, data20 = self.data1[t0], data2_2d_tmp[t1]
            nums1, index1, scale, r, t = self.remove_outliers_2d(data10, data20)
            if nums1 >= 3:
                data10, data20 = data10[index1], data20[index1]
                # method1 = high_order_matching(data10, data20, 500, self.theta)
                num_temp = len(data20) ** 3
                _, match_score = high_order_matching(data10, data20, min(500, num_temp), self.theta)
                data2_index = np.array(t1)[index1]
                data2_index = data_2d_index[data2_index]
                data1_index = np.array(t0)[index1]
                r = np.c_[r, [0,0]]
                r = np.r_[r, [[0,0,1]]]
                r = rot_mat.dot(r)
                t = np.r_[t, -z0]
                return [r_x, r_y, z_min, nums1, float(match_score)*(nums1>2), data1_index, data2_index, scale, r, t]

        return [r_x, r_y, z_min, nums1] + [0]*6


    def worker(self, input, output):
        for r_x, r_y, z_min in iter(input.get, 'STOP'):
            result = self.get_max_matching(r_x, r_y, z_min)
            output.put(result)


    def run(self, tasks):
        # 并行加速
        max_nums = multiprocessing.cpu_count()-10
        z0 = np.min(self.data2[:,2])-self.thickness
        z1 = np.max(self.data2[:,2])+self.thickness
        # tasks = [(r_x, r_y, z_min) for r_x in range(-20,21,5) for r_y in range(-20,21,5) for z_min in np.arange(z0, z1, self.step)]
        task_queue = Queue()
        result_queue = Queue()
        for task in tasks:
            task_queue.put(task)
        for i in range(max_nums):
            Process(target=self.worker, args=(task_queue, result_queue)).start()
        for i in range(max_nums):
            task_queue.put('STOP')
        rets = []
        for i in range(len(tasks)):
            rets.append(result_queue.get())
        return rets

def get_matching(tasks, data_2d, data_3d, thickness, theta, step0):
    matching = data_matching(data_2d, data_3d, step0, thickness, theta)
    ret = matching.run(tasks)
    max_score, nums0 = 0, 0
    r_x0, r_y0 = 0, 0
    index1, index2 = 0, 0
    scale0 = 1
    r0, t0 = np.eye(3), np.zeros(3)
    z1_min = 0
    for r_x, r_y, z_min, nums1, match_score, data1_index, data2_index, scale1, r, t in ret:
        # if nums1 > nums0 or (nums1 == nums0 and match_score > max_score):
        if nums1 >= nums0 and match_score > max_score:
            max_score = match_score
            index1, index2 = data1_index, data2_index
            nums0 = nums1
            r_x0, r_y0 = r_x, r_y
            scale0 = scale1
            r0, t0 = r, t
            z1_min = z_min
    return index1, index2, r0, t0, r_x0, r_y0



if __name__ == '__main__':
    theta1 = np.random.rand(600, 3)
    t1, t2, t3 = 1000, 1000, 200
    mat0 = np.array([[t1, 0, 0], [0, t2, 0], [0, 0, t3]])
    fMOST_data = theta1.dot(mat0)
    r_x0, r_y0 = np.random.rand() * 40 - 20, np.random.rand() * 40 - 20
    depth = 20
    x, y = r_x0 * np.pi / 180, r_y0 * np.pi / 180
    mat_x = np.array([[1, 0, 0],
                      [0, np.cos(x), -np.sin(x)],
                      [0, np.sin(x), np.cos(x)]])
    mat_y = np.array([[np.cos(y), 0, np.sin(y)],
                      [0, 1, 0],
                      [-np.sin(y), 0, np.cos(y)]])
    rot_mat = mat_x.dot(mat_y)
    fMOST_data1 = fMOST_data.dot(rot_mat)
    # t1, t2 = min(fMOST_data1[:, 2]), max(fMOST_data1[:, 2])
    # # 尽量选取中间的范围
    # t1, t2 = t1 + (t2 - t1) / 4, t2 - (t2 - t1) / 4
    # z_index = np.random.rand() * (t2 - t1) + t1
    # z1, z2 = z_index - depth / 2, z_index + depth / 2
    z_min = np.random.rand() * 100 + 50
    t1 = np.array([500,500,z_min]).dot(rot_mat)
    z_min0 = t1[2]
    z1, z2 = z_min0, z_min0+depth

    x1, x2 = 300, 700
    y1, y2 = 300, 700
    outliers = 5
    noise_theta = 5
    index1 = np.where((fMOST_data1[:, 2] >= z1) & (fMOST_data1[:, 2] < z2) & (fMOST_data1[:, 0] >= x1) &
                      (fMOST_data1[:, 0] < x2) & (fMOST_data1[:, 1] >= y1) & (fMOST_data1[:, 1] < y2))[0]
    print(index1)
    two_photon_data = fMOST_data1[index1]
    two_photon_data = two_photon_data[:, :2]
    noise = np.random.rand(two_photon_data.shape[0], 2) * noise_theta
    two_photon_data += noise
    out_data = np.random.rand(outliers, 2) * 400 + [x1, y1]
    two_photon_data = np.r_[two_photon_data, out_data]

    order = list(range(len(fMOST_data)))
    theta = 20
    a = data_matching(two_photon_data, fMOST_data, 20, 40, theta)
    # r_x, r_y, max_score, data1_index, data2_index = a.get_max_matching(r_x0-3, r_y0+3)
    rets = a.run()
    print(rets)
    print(r_x0, r_y0)