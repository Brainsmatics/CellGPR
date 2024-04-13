from utils.icp import best_fit_transform
import numpy as np
from sklearn.decomposition import PCA

def remove_outliers(fMOST_data, two_photon_data, iteration=50, thread=5):
    pca = PCA(n_components=3)
    fMOST_data1 = pca.fit_transform(fMOST_data)
    mat1 = pca.components_
    # print(mat1)
    mat1 = np.array([-mat1[0], mat1[1], mat1[2]]) if np.linalg.det(mat1) < 0 else mat1
    mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] < 0 else mat1
    # mat1 = np.array([mat1[0], -mat1[1], -mat1[2]]) if mat1[2, 1] > 0 else mat1
    # print(mat1)
    fMOST_data1 = mat1.dot(fMOST_data.T).T
    matching_nums = 0
    index = np.zeros(fMOST_data.shape[0]).astype(np.bool)

    for i in range(iteration):
        index0 = np.random.randint(0,fMOST_data.shape[0],3)
        data2 = fMOST_data1[:,:2]
        data20 = data2[index0]
        data1 = two_photon_data[index0]
        T, r, t = best_fit_transform(data1, data20)
        data11 = r.dot(two_photon_data.T).T + t.T
        distance = data11 - data2
        distance = np.sqrt(np.square(distance[:, 0]) + np.square(distance[:, 1]))
        num1 = sum(distance < thread)
        # print(num1)
        if num1 > matching_nums:
            print(num1)
            matching_nums = num1
            index = distance < thread
    fMOST_data = fMOST_data[index]
    two_photon_data = two_photon_data[index]
    if matching_nums < 3:
        print('Warning! there might be an error occured!')
    return fMOST_data, two_photon_data