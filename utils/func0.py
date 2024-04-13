from utils.icp import best_fit_transform
import numpy as np
from utils.get_data import get_swc_data

# def get_trans_mat(r_x, r_y, data_3d, data_2d):
#     x, y = r_x * np.pi / 180, r_y * np.pi / 180
#     mat_x = np.array([[1, 0, 0],
#                   [0, np.cos(x), -np.sin(x)],
#                   [0, np.sin(x), np.cos(x)]])
#     mat_y = np.array([[np.cos(y), 0, np.sin(y)],
#                   [0, 1, 0],
#                   [-np.sin(y), 0, np.cos(y)]])
#     rot_mat = mat_x.dot(mat_y)
#     data_3d = data_3d.dot(rot_mat)
#     data_3d_pro = data_3d[:,:2]
#     z_t = -np.mean(data_3d, 0)[2]
#     print(data_2d.shape)
#     print(data_3d_pro.shape)
#
#     T, r, t = best_fit_transform(data_2d, data_3d_pro)
#     r = np.c_[r, np.zeros([2,1])]
#     r = np.r_[r, np.array([[0,0,1]])]
#     t = np.r_[t, z_t]
#     r = rot_mat.dot(r)
#     return r, t

def get_transmat(*args):
    # r0 = np.multiply(np.eye(3), scale_f)
    r0 = np.eye(3)
    t0 = np.zeros(3)
    for r, t in args:
        r0 = r0.dot(r)
        t0 = t + t0.dot(r)
    # scale_t_trans = np.multiply(np.eye(3), [1.0/scale_t, 1.0/scale_t, 1])
    # r0 = r0.dot(scale_t_trans)
    # r0 = np.c_[r0, t]
    # r0 = np.r_[r0, [[0,0,0,1]]]
    return r0, t0








if __name__ == '__main__':
    data_3d = get_swc_data('../data/202278/results/L5_site1.swc')
    # from
    data_3d = np.array(data_3d)
    data_2d = np.load('../data/202278/results/L5_site1_tp.npy')
    # data_2d = np.array([[0,0],[2,0],[0,3]])
    # data_3d = np.array([[0,0,0], [0,2,0],[-3,0,0]])
    r, t = get_trans_mat(0, 0, data_3d, data_2d)
    print(r, t)
    print(data_3d.dot(r)+t)
    print(data_2d)