# import sys
# sys.path.append('../')

import numpy as np
from utils.get_data import get_csv_data, get_swc_data
import SimpleITK as sitk
import zipfile
from sklearn import decomposition
from utils.icp import best_fit_transform
import cv2
import multiprocessing
from multiprocessing import Process, Queue, freeze_support
import os

class get_matching_results:
    def __init__(self, fMOST_data, two_photon_data, fMOST_image, two_photon_image, thinckness, output_path, r, t, scale, resolution):
        self.fMOST_data = fMOST_data
        self.two_photon_data = two_photon_data
        self.fMOST_image = fMOST_image
        self.two_photon_image = two_photon_image
        self.thinckness = thinckness
        self.output_path = output_path
        self.r, self.t = r, t
        self.scale, self.resolution = scale, resolution



    def get_image_array(self, r, t, fMOST_image, thickness, scale, resolution):
        z1 = np.expand_dims(np.kron(np.ones(512 * 512), np.arange(thickness)), 1)
        y1 = np.expand_dims(np.kron(np.ones(512), np.kron(np.arange(512), np.ones(thickness))), 1)*scale
        x1 = np.expand_dims(np.kron(np.arange(512), np.ones(thickness * 512)), 1)*scale
        # t1 = np.ones(512 * 512 * thickness)
        axis0 = np.c_[x1, y1, z1]
        axis1 = (axis0-t).dot(np.linalg.inv(r))
        axis1 = np.multiply(axis1, [1 / resolution[0], 1 / resolution[0], 1 / resolution[1]])
        axis1 = np.around(axis1[:, :3]).astype('int')
        fMOST_array = sitk.GetArrayFromImage(fMOST_image)
        fMOST_array[0, 0, 0] = 0
        axis1 = axis1[:, ::-1]

        index0 = np.where((axis1[:,0] >= fMOST_array.shape[0]) | (axis1[:,1] >= fMOST_array.shape[1]) | (
                    axis1[:,2] >= fMOST_array.shape[2]))
        axis1[index0[0]] = [0, 0, 0]

        array0 = fMOST_array[tuple(axis1.T)]
        array0 = array0.reshape([-1, thickness])
        array0 = np.max(array0, axis=1).reshape([512, -1]).T
        return array0


    def show_results(self, two_photon_data, fMOST_data, two_photon_image, image_array, r, t, scale, width=100):
        two_photon_array = sitk.GetArrayViewFromImage(two_photon_image)
        two_photon_array, image_array = two_photon_array - np.min(two_photon_array), image_array - np.min(image_array)
        two_photon_array = two_photon_array / np.max(two_photon_array) * 255
        temp = np.zeros([512,512])
        temp[:two_photon_array.shape[0],:two_photon_array.shape[1]] = two_photon_array
        two_photon_array = temp
        two_photon_array.astype('uint8')
        image_array = image_array / np.max(image_array) * 255
        image_array.astype('uint8')
        array1 = np.zeros([two_photon_array.shape[0], width]).astype('uint8')
        array2 = np.c_[two_photon_array, array1, image_array]
        # point_color = (90, 90, 90)
        point_color = (128, 128, 128)
        # # BGR
        # point_color = (135,190,255)
        # color_list = [(192,192,192), (178,48,96), (255,192,203),(176,224,230),
        #               (65,105,225),(0,255,255),(255,128,0),(240,130,140),(124,252,0),
        #               (51,161,201),(0,199,140),(255,0,255)]
        # import random



        array3 = np.zeros([array2.shape[0], array2.shape[1], 3])
        array3[:,:512,1] = two_photon_array*(60/two_photon_array.mean())
        array3[:,-512:,2] = image_array*2
        array4 = array3.copy()


        data1 = two_photon_data / scale

        data2 = fMOST_data.dot(r) + t

        # data2 = np.linalg.inv(inv_mat).dot(np.c_[fMOST_data, np.ones(fMOST_data.shape[0])].T).T
        data2 = data2[:, :2] / scale

        np.save(self.output_path[:-4] + '_data.npy', [data1, data2])

        for (i, j), (p, q) in zip(data1, data2):
            # point_color = random.choice(color_list)
            # point_color = (point_color[2], point_color[1], point_color[0])
            cv2.line(array3, (int(i), int(j)),
                     (int(p + width + two_photon_array.shape[1]), int(q)), point_color, 2)
        # cv2.imwrite(os.path.join(self.output_path), array3)
        cv2.imwrite(self.output_path[:-4] + '.jpg', array3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5

        # for t, ((i, j), (p, q)) in enumerate(zip(data1, data2)):
        #     cv2.putText(array4, str(t), (int(i)+10,int(j)+10), font, fontScale, (0, 255, 255), thickness=2)
        #     cv2.putText(array4, str(t), (int(p + width + two_photon_array.shape[1])+10, int(q)+10), font, fontScale, (0, 255, 255), thickness=2)
        cv2.imwrite(self.output_path[:-4] + '_label.jpg', array4)


    def run(self):
        # inv_mat = self.get_trans_mat(self.fMOST_data, self.two_photon_data, self.thinckness)
        image_array = self.get_image_array(self.r, self.t, self.fMOST_image, self.thinckness, self.scale, self.resolution)
        self.show_results(self.two_photon_data, self.fMOST_data, self.two_photon_image, image_array,
                           self.r, self.t, self.scale)




if __name__ == '__main__':
    import numpy as np
    from parameter import *
    from utils.get_data import get_csv_data, get_swc_data
    r, t = np.load('../temp/r.npy'), np.load('../temp/t.npy')
    f_points, tp_points = np.load('../temp/f_points.npy'), np.load('../temp/tp_points.npy')
    f_image = sitk.ReadImage('../data/202278/fMOST.tif')
    tp_image = sitk.ReadImage('../data/202278/2p/L5_site1.tif')
    output_path = '../temp/result/result.jpg'
    a = get_matching_results(f_points, tp_points, f_image, tp_image, thickness_L5, output_path, r, t, scale_L23, [0.65,2])
    a.run()
