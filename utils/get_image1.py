import sys
sys.path.append('../')
import numpy as np
from utils.get_data import get_csv_data, get_swc_data
import SimpleITK as sitk
import zipfile
from sklearn import decomposition
from utils.icp import best_fit_transform
import cv2
import multiprocessing
from multiprocessing import Process, Queue


class get_matching_results:
    def __init__(self, fMOST_data, two_photon_data, fMOST_image, two_photon_image, thinckness, scale = 0.8, resolution = [0.65,2]):
        self.fMOST_data = fMOST_data
        self.two_photon_data = two_photon_data
        self.fMOST_image = fMOST_image
        self.two_photon_image = two_photon_image
        self.thinckness = thinckness
        # 双光子成像分辨率
        self.scale = scale
        self.resolution = resolution

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

    def get_value(self, x, y, z, inv_mat):
        p = np.array([[x * self.scale], [y * self.scale], [z], [1]])
        index = inv_mat.dot(p)
        x0, y0, z0 = int(index[0, 0] / self.resolution[0]), int(index[1, 0] / self.resolution[0]), int(
            index[2, 0] / self.resolution[1])
        # print(x, y, z, self.fMOST_array[z0, y0, x0])
        return x, y, z, self.fMOST_array[z0, y0, x0]

    def worker(self, input, output):
        for x, y, z in iter(input.get, 'STOP'):
            result = self.get_value(x, y, z, self.inv_mat)
            output.put(result)


    def get_image_array(self, two_photon_image, fMOST_image, inv_mat, thickness):
        self.inv_mat = inv_mat
        size_x, size_y, size_z = two_photon_image.GetSize()[0], two_photon_image.GetSize()[1], thickness
        print(size_x, size_y, size_z)
        self.fMOST_array = sitk.GetArrayFromImage(fMOST_image)
        image_array = np.zeros([size_z, size_y, size_x], dtype='uint16')
        # 使用并行加速
        max_num = multiprocessing.cpu_count() if thinkness > multiprocessing.cpu_count() else thinkness
        tasks = [(x, y, z) for x in range(size_x) for y in range(size_y) for z in range(size_z)]
        task_queue, result_queue = Queue(), Queue()
        for task in tasks:
            task_queue.put(task)

        for i in range(max_num):
            Process(target=self.worker, args=(task_queue, result_queue)).start()
        for i in range(max_num):
            task_queue.put('STOP')
        for i in range(len(tasks)):
            x, y, z, value = result_queue.get()
            image_array[z,y,x] = value


        # for x in range(size_x):
        #     # print(x)
        #     for y in range(size_y):
        #         for z in range(size_z):
        #             p = np.array([[x * scale], [y * scale], [z], [1]])
        #             index = inv_mat.dot(p)
        #             x0, y0, z0 = int(index[0, 0] / resolution[0]), int(index[1, 0] / resolution[0]), int(index[2, 0] / resolution[1])
        #             # print(z0, y0, x0)
        #             image_array[z, y, x] = fMOST_array[z0, y0, x0]
        image_array = np.max(image_array, 0)
        # image_array = image_array.transpose((1, 0))
        return image_array

    def show_results(self, two_photon_data, fMOST_data, two_photon_image, image_array, inv_mat, scale, width=100):
        two_photon_array = sitk.GetArrayViewFromImage(two_photon_image)
        two_photon_array, image_array = two_photon_array - np.min(two_photon_array), image_array - np.min(image_array)
        two_photon_array = two_photon_array / np.max(two_photon_array) * 255
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
                     (int(p + width + two_photon_array.shape[1]), int(q)), point_color, 1)

        # a = cv2.line(array3, (0,0), (100,100), point_color, 1)
        cv2.imwrite('result.jpg', array3)

    def run(self):
        inv_mat = self.get_trans_mat(self.fMOST_data, self.two_photon_data, self.thinckness)
        image_array = self.get_image_array(self.two_photon_image, self.fMOST_image, inv_mat, self.thinckness)
        # image = sitk.GetImageFromArray(image_array)
        self.show_results(self.two_photon_data, self.fMOST_data, self.two_photon_image, image_array,
                           inv_mat, self.scale)
        # sitk.WriteImage(image, 'test1.tif')
        # cv2.imshow('test', image_array)
        # cv2.waitKey()



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
    fMOST_image_path = '../test_data/crop.tif'
    two_photon_image_path = '../test_data/two_photon1.tif'
    fMOST_image = sitk.ReadImage(fMOST_image_path)
    two_photon_image = sitk.ReadImage(two_photon_image_path)
    scale = 0.8
    two_photon_data, _ = get_data('../test_data/inverse/RoiSet.zip')
    fMOST_data = np.array(get_swc_data('../test_data/inverse/crop.swc'))
    two_photon_data, fMOST_data = np.array(two_photon_data), np.array(fMOST_data)
    two_photon_data = two_photon_data * scale
    print(two_photon_data)
    thinkness = 20
    a = get_matching_results(fMOST_data, two_photon_data, fMOST_image, two_photon_image, thinkness, scale)
    a.run()






