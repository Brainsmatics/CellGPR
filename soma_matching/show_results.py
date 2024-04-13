from matplotlib import pylab as plt
import numpy as np

class show_results():
    def __init__(self, data1, data2, indexs):
        self.data1 = data1
        self.data2 = data2
        assert data1.shape[1] == 2
        assert data2.shape[1] == 2
        self.indexs = indexs

    def plot_points(self):
        plt.figure(1)
        data1, data2 = self.data1, self.data2
        plt.subplot(1,2,1)
        plt.scatter(data1[:,0], data1[:,1], c = 'r')
        plt.title('two photon image')
        for (i,j) in self.indexs:
            plt.annotate(i, (data1[i][0], data1[i][1]))
        plt.subplot(1,2,2)
        plt.scatter(data2[:,0], data2[:,1], c = 'g')
        plt.title('fMOST image')
        for (i,j) in self.indexs:
            # print(j)
            plt.annotate(i, (data2[j][0], data2[j][1]))

        plt.show()


if __name__ == "__main__":
    from SM.SM import SM
    data1 = np.array([[0,1], [0,0], [2,0]])
    data2 = np.array([[0,0], [1,0], [0, -2], [4,4]])
    method = SM(data1, data2, 1)
    indexes,_ = method.run()
    print(indexes)
    plot0 = show_results(data1, data2, indexes)
    plot0.plot_points()





