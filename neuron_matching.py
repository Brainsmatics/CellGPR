import os

os.chdir(r'G:\item\neuron_matching_v3')

import sys

sys.path.append('./deep_detection')
import multiprocessing
import os
import time

import numpy as np
import SimpleITK as sitk

# from deep_detection.predict1 import detection
# from matplotlib import pyplot as plt
from parameter import *
from soma_matching.data_matching3 import get_matching
from utils.data_analyse import data_analyse
from utils.func import get_closest_points1
from utils.func0 import get_transmat
from utils.get_data import *
from utils.get_image5 import get_matching_results
from utils.neuron_sort import neuron_sort
from utils.remove_outliers import remove_outliers


def run():
    output_path = os.path.join(data_path, 'results')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    # 细胞识别
    print('*' * 50 + 'doing soma detection' + '*' * 50)
    image_path = os.path.join(data_path, 'fMOST.tif')
    # a = detection(image_path=image_path, model_path=model_path, thread_distance=distance_thread,
    #               thread_conf=conf_theta, batch_size=batch_size)
    # a.run()
    fMOST_data = get_swc_data(os.path.join(data_path, 'fMOST.swc'))

    # 细胞分类
    print('*' * 50 + 'doing neuron sorting' + '*' * 50)
    sort = neuron_sort(image_path=image_path, distance_thread=180, output_path=output_path, center_points=fMOST_data)
    trans_L23_points, index_L23, trans_L5_points, index_L5, r00 = sort.run()
    print('sorting complete!')
    fMOST_image = sitk.ReadImage(image_path)
    two_photon_dir = os.path.join(data_path, '2p')

    for file_name in os.listdir(two_photon_dir):
        if file_name.endswith('csv'):
            print('*' * 50 + 'doing {} matching'.format(file_name) + '*' * 50)
            flag = 'L23' if 'L23' in file_name else 'L5'
            # b = data_analyse(os.path.join(data_path, 'fMOST.swc'), output_path, flag=flag)
            # index, data0 = b.run()
            index = index_L5 if flag == 'L5' else index_L23
            data0 = trans_L5_points if flag == 'L5' else trans_L23_points
            thickness = thickness_L23 if flag == 'L23' else thickness_L5

            two_photon_data = get_csv_data(os.path.join(two_photon_dir, file_name), 2)
            scale = scale_L23 if 'L23' in file_name else scale_L5
            two_photon_data = two_photon_data * scale

            z0_min, z0_max = np.min(data0[:, 2]), np.max(data0[:, 2])
            tasks = [(r_x, r_y, z_min) for r_x in range(-20, 21, 5) for r_y in range(-20, 21, 5) for z_min in
                     np.arange(z0_min, z0_max, step0)]

            res = get_matching(tasks, two_photon_data, data0, thickness, 500, step0)

            index1, index2, r0, t0, r_x0, r_y0 = res
            data0_trans = data0.dot(r0) + t0
            index_cub = np.where((data0_trans[:, 0] > -10) & (data0_trans[:, 0] < 420) & (data0_trans[:, 1] > -10) & (
                        data0_trans[:, 1] < 420) & (data0_trans[:, 2] > -20) & (data0_trans[:, 2] < thickness + 20))[0]
            data0_trans1 = data0_trans[index_cub]

            range0 = list(range(-8, 9, 2))
            tasks1 = [(r_x, r_y, z_min) for r_x in range0 for r_y in range0 for z_min in np.arange(-10, thickness + 20, step0)]
            res1 = get_matching(tasks1, two_photon_data, data0_trans1, thickness, 500, step0)
            ind1, ind2, r1, t1, r_x1, r_y1 = res1
            data0_trans2 = data0_trans1.dot(r1) + t1
            res_index = get_closest_points1(data0_trans2[:, :2], two_photon_data, distance_thread=15)
            p1, p2 = [i for i, _ in res_index], [i for _, i in res_index]
            f_data0 = fMOST_data[index][index_cub][p2]
            f_two = two_photon_data[p1]
            r2, t2 = get_transmat((r00, np.zeros(3)), (r0, t0), (r1, t1))
            # output_path = './temp/result/result.jpg'
            two_photon_image = sitk.ReadImage(os.path.join(two_photon_dir, file_name[:-3] + 'tif'))
            output_name = os.path.join(output_path, file_name[:-3] + 'jpg')
            print(output_name)
            a = get_matching_results(f_data0, f_two, fMOST_image, two_photon_image, thickness,
                                     output_name, r2, t2, scale, [0.65, 2])
            a.run()
            d = {}
            d['fMOST'] = fMOST_data[index][index_cub]
            d['two_photon'] = two_photon_data
            d['index'] = [p1, p2]
            d['r'] = r2
            d['t'] = t2
            np.save(os.path.join(output_path, file_name[:-3] + 'npy'), d)


if __name__ == '__main__':
    run()