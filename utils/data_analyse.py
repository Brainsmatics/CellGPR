from sklearn import decomposition
import numpy as np
from matplotlib import pyplot as plt
from sklearn import cluster
from utils.get_data import get_swc_data


class data_analyse:
    def __init__(self, swc_path, dst_path, flag):
        self.data = get_swc_data(swc_path)
        self.dst_path = dst_path
        assert flag == 'L23' or 'L5'
        self.flag = flag

    def pca_data(self, data):
        pca = decomposition.PCA(n_components=3)
        new_points = pca.fit_transform(data)
        mat1 = pca.components_
        mat1 = np.array([-mat1[0], mat1[1], mat1[2]]) if np.linalg.det(mat1) < 0 else mat1
        # mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
        mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
        new_points = mat1.dot(data.T).T
        print('in data_analyse, mat1_det = {}'.format(np.linalg.det(mat1)))
        return new_points

    def K_cluster(self, data):
        y_pred = cluster.KMeans(n_clusters=2).fit_predict(data[:, 2].reshape([-1, 1]))
        pred = np.zeros(y_pred.shape)
        if y_pred.tolist().count(1) > y_pred.tolist().count(0):
            # 多的是23层 5层置为1
            pred[y_pred == 0] = 1
        else:
            pred[y_pred == 1] = 1
        return pred

    def write_swc(self, dst, data0, transformed_data, pred):
        f1 = open(dst + 'soma2-3.swc', 'w')
        f2 = open(dst + 'soma5.swc', 'w')
        f3 = open(dst + 'soma2-3_transformed.swc', 'w')
        f4 = open(dst + 'soma5_transformed.swc', 'w')

        count1, count2 = 0, 0
        for point, new_point, pred0 in zip(data0, transformed_data, pred):
            if pred0 == 0:
                count1 += 1
                str1 = '{} 1 {} {} {} 1 -1\n'.format(count1, str(point[0]), str(point[1]), str(point[2]))
                f1.writelines(str1)
                str2 = '{} 1 {} {} {} 1 -1\n'.format(count1, str(new_point[0]), str(new_point[1]), str(new_point[2]))
                f3.writelines(str2)
            else:
                count2 += 1
                str1 = '{} 1 {} {} {} 1 -1\n'.format(count2, str(point[0]), str(point[1]), str(point[2]))
                f2.writelines(str1)
                str2 = '{} 1 {} {} {} 1 -1\n'.format(count1, str(new_point[0]), str(new_point[1]), str(new_point[2]))
                f4.writelines(str2)

        f1.close()
        f2.close()
        f3.close()
        f4.close()

    def get_results(self, transformed_data, pred, flag):
        assert flag == 'L23' or 'L5'
        if flag == 'L23':
            data0 = transformed_data[pred == 0]
        else:
            data0 = transformed_data[pred == 1]

        return data0

    def get_index(self, pred, flag):
        pred = np.array(pred)
        t = np.where(pred == 0) if flag == 'L23' else np.where(pred == 1)
        return list(t[0])


    def run(self):
        points_transformed = self.pca_data(self.data)
        pred = self.K_cluster(points_transformed)
        self.write_swc(self.dst_path, self.data, points_transformed, pred)
        data0 = self.get_results(points_transformed, pred, self.flag)
        index = self.get_index(pred, self.flag)
        return index, data0

if __name__ == '__main__':
    a = data_analyse('../data_analyse/crop.swc', '../output_dir/', flag='L5')
    index, data0 = a.run()
    print(index)
    # print(data0)