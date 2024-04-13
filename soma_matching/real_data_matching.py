import numpy as np
from utils.get_data import get_swc_data, get_csv_data
from soma_matching.methods.MPM import MPM

class soma_matching:


def get_max_matching(r_x, r_y, two_photon_data, fMOST_data, step0, theta = 10000):
    x, y = r_x*np.pi/180, r_y*np.pi/180
    mat_x = np.array([[1,0,0],
                      [0,np.cos(x), -np.sin(x)],
                      [0,np.sin(x), np.cos(x)]])
    mat_y = np.array([[np.cos(y),0,np.sin(y)],
                      [0,1,0],
                      [-np.sin(y),0,np.sin(y)]])
    rot_mat = mat_x.dot(mat_y)
    data_3d_rotate = fMOST_data.dot(rot_mat)
    max_score, indexes0 = 0, 0
    for i in range(np.min(data_3d_rotate[:,3]), step0, np.max(data_3d_rotate[:,3])):
        data_2d_tmp = []
        for j in range(data_3d_rotate.shape[0]):
            if data_3d_rotate[j,3]>i-thickness/2 and data_3d_rotate[j,3]<i+thickness/2:
                data_2d_tmp = np.append(data_2d_tmp, data_3d_rotate[j,:2])
            if data_2d_tmp == []:
                continue
        method = MPM(two_photon_data, data_2d_tmp, theta)
        indexes, match_score = method.run()
        if match_score > max_score:
            max_score = match_score
            indexes0 = indexes

        return indexes0, max_score

def soma_matching(fMOST_data, two_photon_data, theta, step, thickless):
    pass


if __name__ == '__main__':
    fMOST_data = get_swc_data('./soma2-3_transformed.swc')
    two_photon_data = get_csv_data('./two_photon.csv', dim=2)
    step0 = 10
    thickness = 20
    theta = 10000
    soma_matching(fMOST_data, two_photon_data, theta, step0, thickness)













