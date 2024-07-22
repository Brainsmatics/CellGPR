import numpy as np
# import torch
from munkres import Munkres
from scipy import io
import ctypes

def tensor_matching(indH3, valH3, N1, N2):
    # N1, N2 = 22, 144
    # indH3 = io.loadmat(r"D:\item\tensor_matching\RRWHM_release_v1.1\indH3.mat")['indH3']
    # valH3 = io.loadmat(r"D:\item\tensor_matching\RRWHM_release_v1.1\valH3.mat")['valH3']
    indH3 = np.sort(indH3)
    _, index = np.unique(indH3, axis = 0, return_index=True)
    indH3, valH3 = indH3[index], valH3[index]

    indH3 = np.r_[indH3, indH3[:,[1,0,2]], indH3[:,[2,0,1]]]
    valH3 = np.r_[valH3, valH3, valH3]
    t0 = np.lexsort((indH3[:,2]//N2, indH3[:,1]//N2, indH3[:,0]))
    indH3 = indH3[t0]
    valH3 = valH3[t0]

    indH3 = indH3.T
    indH3 = np.array(indH3.tolist()).astype(np.uint32)
    valH3 = np.array(valH3.tolist())


    p = np.ones(N1 * N2) /np.sqrt(N1*N2)

    dll = ctypes.cdll.LoadLibrary('./matching.so')

    p_ = p.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    p_out = np.zeros(p.shape)
    p_out_ = p_out.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    pScore = ctypes.c_double(0.0)
    indH3_ = indH3.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    valH3_ = valH3.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    # dll.print_vector(indH3_, valH3_, len(valH3))
    dll.tensormatching(p_, N1, N2, indH3_, valH3_, len(valH3), p_out_, ctypes.pointer(pScore))


    m = Munkres()
    x1 = p_out.reshape(N1, N2)
    if N1 <= N2:
        index = m.compute(-x1)
    else:
        index = m.compute(-x1.T)
        index = [(j, i) for i, j in index]
    return index, pScore.value