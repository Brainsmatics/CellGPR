import numpy as np
from multiprocessing import Process, Queue
from utils.get_data import get_csv_data, get_swc_data
# from methods.MPM import MPM
from soma_matching.methods.MPM import MPM
import multiprocessing
'''
使用图匹配算法得到最终的配对结果
输入 双光子坐标， fMOST 23 and 5坐标 以及对应原始坐标的序列
输出 配对关系， 配对结果图
liwenwei 2020/12/14
'''
# def chunks(l, n):
#     for i in range(0, len(l), n):
#         yield l[i:i + n]

class soma_matching:
    def __init__(self, fMOST_data, two_photon_data, order, step, thickness, theta):
        self.fMOST_data = fMOST_data
        self.two_photon_data = two_photon_data
        self.order = order
        self.step = step
        self.thickness = thickness
        self.theta = theta


    def get_max_matching(self, r_x, r_y, flag=1):
        # print('r_x = {}, r_y = {}'.format(r_x, r_y))
        x, y = r_x * np.pi / 180, r_y * np.pi / 180
        mat_x = np.array([[1, 0, 0],
                          [0, np.cos(x), -np.sin(x)],
                          [0, np.sin(x), np.cos(x)]])
        mat_y = np.array([[np.cos(y), 0, np.sin(y)],
                          [0, 1, 0],
                          [-np.sin(y), 0, np.cos(y)]])
        rot_mat = mat_x.dot(mat_y)
        data_3d_rotate = self.fMOST_data.dot(rot_mat)
        max_score, indexes0 = 0, 0
        data0 = []
        for i in np.arange(np.min(data_3d_rotate[:, 2]), np.max(data_3d_rotate[:, 2]), self.step):
            data_2d_tmp = []
            order_list = []
            for j in range(data_3d_rotate.shape[0]):
                if data_3d_rotate[j, 2] > i - self.thickness / 2 and data_3d_rotate[j, 2] < i + self.thickness / 2:
                    order_list.append(self.order[j])
                    data_2d_tmp = np.append(data_2d_tmp, data_3d_rotate[j, :2])
                if data_2d_tmp == []:
                    continue
            data_2d_tmp = np.array(data_2d_tmp).reshape([-1, 2])
            if data_2d_tmp.shape[0] < 3:
                continue
            method = MPM(self.two_photon_data, data_2d_tmp, self.theta)

            indexes, match_score = method.run()
            indexes = [(i, order_list[t]) for i, t in indexes]


            if match_score > max_score:
                max_score = match_score
                indexes0 = indexes
                data0 = data_2d_tmp
        if flag == 1:
            # queue.put([r_x, r_y, float(max_score)])
            return [r_x, r_y, float(max_score)]
            # print('*'*100)
        if flag == 0:
            return data0, indexes0

    def worker(self, input, output):
        for r_x, r_y in iter(input.get, 'STOP'):
            result = self.get_max_matching(r_x, r_y)
            output.put(result)


    def run(self):
        # 并行加速
        max_nums = multiprocessing.cpu_count()
        tasks = [(r_x, r_y) for r_x in range(-20,20,2) for r_y in range(-20,20,2)]
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

        print(rets)
        rets = np.array(rets)
        r_x, r_y, score = rets.T
        t = int(np.where(score == score.max())[0])
        r_x0, r_y0 = r_x[t], r_y[t]
        data0, indexes = self.get_max_matching(r_x0, r_y0, 0)
        print(data0)
        print(indexes)
        print('r_x0 = {}, r_y0 = {}'.format(r_x0, r_y0))
        # self.plot_points(self.two_photon_data, data0, indexes)



        print('*'*100)
        return indexes


if __name__ == '__main__':
    order = [3, 6, 14, 15, 16, 17, 21, 25, 35, 36, 39, 53, 55, 61, 62, 72, 73, 79, 85, 100, 101, 120, 124, 128, 129, 130]
    fMOST_data = get_swc_data('../output_dir/soma5_transformed.swc')
    two_photon_data = get_csv_data('./two_photon.csv', dim=2)
    two_photon_data = two_photon_data * 0.8
    step0 = 10
    thickness = 20
    theta = 10000
    a = soma_matching(fMOST_data, two_photon_data, order, step0, thickness, theta)
    indexes = a.run()
    print(indexes)