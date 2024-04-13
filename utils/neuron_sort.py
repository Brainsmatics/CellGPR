import numpy as np
import SimpleITK as sitk
import os
from sklearn import decomposition

class                                                                                                                                                                            neuron_sort:
    def __init__(self, image_path, distance_thread, output_path, center_points):
        self.image_path = image_path
        self.distance_thread = distance_thread
        self.output_path = output_path
        self.center_points = center_points

    def get_resample(self, image):
        orignal_resolution, new_resolution = [0.65, 0.65, 2], [5,5,5]
        image.SetSpacing([0.65,0.65,2])
        new_size = [round(image.GetSize()[i] * orignal_resolution[i]/new_resolution[i]) for i in range(3)]
        image1 = sitk.Resample(image, new_size, sitk.Transform(),
                               sitk.sitkNearestNeighbor, image.GetOrigin(), new_resolution,
                               image.GetDirection(), 0.0, image.GetPixelID())
        return image1

    def get_binary(self, image):
        thread_image = image > 0
        cleaned_thread_image = sitk.BinaryOpeningByReconstruction(thread_image, [3, 3, 3])
        cleaned_thread_image = sitk.BinaryClosingByReconstruction(cleaned_thread_image, [3, 3, 3])
        filled_image = sitk.BinaryFillhole(cleaned_thread_image, foregroundValue=0)
        for i in range(filled_image.GetSize()[2]):
            filled_image[:, :, i] = sitk.BinaryFillhole(filled_image[:, :, i])
        return filled_image

    def get_edge(self, image):
        image1 = sitk.SobelEdgeDetection(sitk.Cast(image, sitk.sitkFloat32))
        return image1

    def get_points(self, image_array):
        points = np.argwhere(image_array > 10)
        points = points[:,::-1]
        points *= 5
        return points

    def get_transform(self, data):
        # 输入坐标点，输出变换矩阵
        pca = decomposition.PCA(n_components=3)
        new_points = pca.fit_transform(data)
        mat1 = pca.components_
        mat1 = np.array([-mat1[0], mat1[1], mat1[2]]) if np.linalg.det(mat1) < 0 else mat1
        # mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
        mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
        new_points = mat1.dot(data.T).T
        center_z = np.mean(new_points[:,2])
        return mat1, center_z

    def write_swc(self, points, path):
        count = 0
        f1 = open(path, 'w')
        for point in points:
            count += 1
            str1 = '{} 1 {} {} {} 1 -1\n'.format(count, str(point[0]), str(point[1]), str(point[2]))
            f1.writelines(str1)

    def run(self):
        # 进行边缘检测
        image = sitk.ReadImage(self.image_path)
        image = self.get_resample(image)
        image = self.get_binary(image)
        image = self.get_edge(image)
        sitk.WriteImage(image, os.path.join(self.output_path, 'edge.tif'))

        # 得到变换矩阵以及z向平均值
        image_array = np.array(sitk.GetArrayFromImage(image))
        edge_points = self.get_points(image_array)
        mat1, center_z = self.get_transform(edge_points)

        # 对坐标点进行变换，记录
        transformed_points = mat1.dot(self.center_points.T).T
        distance = transformed_points[:,2] - center_z

        # 对坐标点进行分类
        L5_points = self.center_points[distance > self.distance_thread]
        L23_points = self.center_points[distance <= self.distance_thread]
        trans_L5_points = transformed_points[distance > self.distance_thread]
        trans_L23_points = transformed_points[distance <= self.distance_thread]
        self.write_swc(L5_points, os.path.join(self.output_path, 'soma5.swc'))
        self.write_swc(L23_points, os.path.join(self.output_path, 'soma2-3.swc'))
        self.write_swc(trans_L5_points, os.path.join(self.output_path, 'soma5_transformed.swc'))
        self.write_swc(trans_L23_points, os.path.join(self.output_path, 'soma2-3_transformed.swc'))
        # t = np.where(distance > self.distance_thread) if self.flag == 'L5' else np.where(distance <= self.distance_thread)
        # index = list(t[0])
        index_L23 = np.where(distance <= self.distance_thread)[0]
        index_L5 = np.where(distance > self.distance_thread)[0]

        return trans_L23_points, index_L23, trans_L5_points, index_L5, mat1.T






if __name__ == '__main__':
    swc_path = '../deep_detection/test_data/fMOST.swc'
    from utils.get_data import get_swc_data
    points = get_swc_data(swc_path)
    a = neuron_sort(image_path='../deep_detection/test_data/fMOST.tif', distance_thread=180, output_path='./', center_points=points)
    trans_L23_points, index_L23, trans_L5_points, index_L5 = a.run()
    print(trans_L23_points, index_L23, trans_L5_points, index_L5)