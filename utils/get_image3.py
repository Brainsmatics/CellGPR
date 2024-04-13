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
    def __init__(self, fMOST_data, two_photon_data, fMOST_image, two_photon_image, thinckness, output_path, scale = 0.8, ):
        self.fMOST_data = fMOST_data
        self.two_photon_data = two_photon_data
        self.fMOST_image = fMOST_image
        self.two_photon_image = two_photon_image
        self.thinckness = thinckness
        # 双光子成像分辨率
        self.scale = scale
        self.output_path = output_path

    def get_trans_mat(self, fMOST_data, two_photon_data, thinkness):
        pca = decomposition.PCA(n_components=3)
        fMOST_data1 = pca.fit_transform(fMOST_data)
        mat1 = pca.components_
        print(mat1)
        # mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
        mat1 = np.array([-mat1[0], mat1[1], mat1[2]]) if np.linalg.det(mat1) < 0 else mat1
        mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
        print(mat1)
        fMOST_data1 = mat1.dot(fMOST_data.T)
        tz = -np.mean(fMOST_data1[2]) + thinkness / 2
        data1 = fMOST_data1[:2].T
        data2 = two_photon_data
        T, r, t = best_fit_transform(data1, data2)
        print('data2 = {}'.format(data2))
        print(r.dot(data1.T).T + t)

        mat1 = np.c_[mat1, [0, 0, 0]]
        mat1 = np.r_[mat1, [[0, 0, 0, 1]]]
        mat2 = np.c_[r, [0, 0], t]
        mat2 = np.r_[mat2, [[0, 0, 1, tz]], [[0, 0, 0, 1]]]
        return np.linalg.inv(mat2.dot(mat1))

    def get_image_array(self, mat, fMOST_image, thickness, scale, resolution):
        z1 = np.expand_dims(np.kron(np.ones(512 * 512), np.arange(thickness)), 1)
        y1 = np.expand_dims(np.kron(np.ones(512), np.kron(np.arange(512), np.ones(thickness))), 1) * scale
        x1 = np.expand_dims(np.kron(np.arange(512), np.ones(thickness * 512)), 1) * scale
        t1 = np.ones(512 * 512 * thickness)
        axis0 = np.c_[x1, y1, z1, t1]
        axis1 = axis0.dot(mat.T)
        axis1 = np.multiply(axis1, [1 / resolution[0], 1 / resolution[0], 1 / resolution[1], 1])
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


    def show_results(self, two_photon_data, fMOST_data, two_photon_image, image_array, inv_mat, scale, width=100):
        two_photon_array = sitk.GetArrayViewFromImage(two_photon_image)
        two_photon_array, image_array = two_photon_array - np.min(two_photon_array), image_array - np.min(image_array)
        two_photon_array = two_photon_array / np.max(two_photon_array) * 255 * 2
        temp = np.zeros([512,512])
        temp[:two_photon_array.shape[0],:two_photon_array.shape[1]] = two_photon_array
        two_photon_array = temp
        two_photon_array.astype('uint8')
        image_array = image_array / np.max(image_array) * 255
        image_array.astype('uint8')
        array1 = np.zeros([two_photon_array.shape[0], width]).astype('uint16')
        array2 = np.c_[two_photon_array, array1, image_array]
        point_color = (0, 128, 128)
        array3 = np.zeros([array2.shape[0], array2.shape[1], 3])
        for i in range(3):
            array3[:, :, i] = array2

        data1 = two_photon_data / scale
        data2 = np.linalg.inv(inv_mat).dot(np.c_[fMOST_data, np.ones(fMOST_data.shape[0])].T).T
        data2 = data2[:, :2] / scale

        for (i, j), (p, q) in zip(data1, data2):
            cv2.line(array3, (int(i), int(j)),
                     (int(p + width + two_photon_array.shape[1]), int(q)), point_color, 2)

        # a = cv2.line(array3, (0,0), (100,100), point_color, 2)
        cv2.imwrite(os.path.join(self.output_path), array3)

    def run(self):
        inv_mat = self.get_trans_mat(self.fMOST_data, self.two_photon_data, self.thinckness)
        image_array = self.get_image_array(inv_mat, self.fMOST_image, self.thinckness, self.scale, [0.65,2])
        # image = sitk.GetImageFromArray(image_array)
        self.show_results(self.two_photon_data, self.fMOST_data, self.two_photon_image, image_array,
                           inv_mat, self.scale)




def get_data(zip_path):
    namelist = zipfile.ZipFile(zip_path).namelist()
    points_3d, points_2d = [], []
    for i in range(len(namelist)):
        if i % 2 == 0:
            a, b, c = namelist[i][:-4].split('-')
            a, b, c = int(a), int(b), int(c)
            points_3d.append([c,b,a])
        else:
            a, b = namelist[i][:-4].split('-')
            a, b = int(a), int(b)
            points_2d.append([b,a])
    return points_2d, points_3d


if __name__ == '__main__':
    import numpy as np
    from utils.get_data import get_csv_data, get_swc_data
    from soma_matching.data_matching import data_matching

    fMOST_data_trans = get_swc_data('../data/202277/resultssoma5_transformed.swc')
    fMOST_data = get_swc_data('../data/202277/resultssoma5.swc')
    two_photon_data = get_csv_data('../data/202277/2p/L5_site2.csv', 2) * 0.8
    matching = data_matching(two_photon_data, fMOST_data_trans, 20, 40, 20)
    z0_min, z0_max = np.min(fMOST_data_trans[:, 2]), np.max(fMOST_data_trans[:, 2])
    tasks = [(r_x, r_y, z_min) for r_x in range(-20, 21, 5) for r_y in range(-20, 21, 5) for z_min in
             np.arange(z0_min, z0_max, 20)]
    ret = matching.run(tasks)
    max_score, nums0 = 0, 0
    index1, index2 = 0, 0
    for r_x, r_y, z_min, nums1, match_score, data1_index, data2_index in ret:
        if match_score > max_score:
            max_score = match_score
            index1, index2 = data1_index, data2_index
            nums0 = nums1
    import SimpleITK as sitk

    fMOST_image = sitk.ReadImage('../data/202277/fMOST.tif')
    two_photon_image = sitk.ReadImage(('../data/202277/2p/L5_site2.tif'))
    fMOST_data0 = fMOST_data[index2]
    two_photon_data0 = two_photon_data[index1]
    a = get_matching_results(fMOST_data0, two_photon_data0, fMOST_image, two_photon_image, 40,'./result.jpg')
    a.run()