from SimpleITK import Image
from pandas import read_csv
from sklearn.metrics import euclidean_distances
import numpy as np
from sklearn.metrics.pairwise import distance
import os
import SimpleITK as sitk
import pandas as pd
from predict1 import detection

# from deep_detection.predict1 import detection


def get_csv_data(csv_path, dim):
    data = []
    df = pd.read_csv(csv_path)
    if dim == 2:
        for i in range(df.shape[0]):
            data.append([df.loc[i, 'X'], df.loc[i, 'Y']])
    else:
        for i in range(df.shape[0]):
            data.append([df.loc[i, 'X'], df.loc[i, 'Y'], df.loc[i, 'Slice']])
    return np.array(data)

def get_labeled_image(detection, image_size, image_path, out_name):
    label_image = sitk.Image(image_size, sitk.sitkUInt16)
    if detection != []:
        detection = np.round(detection)
        for i in range(detection.shape[0]):
            x, y, z = int(detection[i][0]), int(detection[i][1]), int(detection[i][2])
            for x1 in range(max(0, x-1),min(image_size[0], x+2)):
                for y1 in range(max(0, y-1),min(image_size[1], y+2)):
                    for z1 in range(max(0, z-1),min(image_size[2], z+2)):
                        label_image[x1, y1, z1] = 65535
    sitk.WriteImage(label_image, image_path[:-4]+ '_{}_label.tif'.format(out_name))


def combine(data_path):
    # 将胞体坐标进行统一
    image = sitk.ReadImage(os.path.join(data_path, 'fMOST.tif'))
    x0, y0, z0 = image.GetSize()
    x_num, y_num, z_num = x0 // 256, y0 // 256, z0 // 92
    data0 = []
    for filename in os.listdir(os.path.join(data_path, 'split')):
        if filename.endswith('.csv'):
            tmp = filename.split('.')[0]
            i = int(tmp.split('_')[-1])
            x1 = i //(y_num*z_num)
            y1 = (i % (y_num*z_num)) // z_num
            z1 = i % z_num
            data =  get_csv_data(os.path.join(data_path, 'split', filename), 3)
            data = data + [x1 * 256, y1 * 256, z1 * 92]
            data0 += data.tolist()
    return data0



class evalue_detection:
    # 输入待评估数据，以及训练好的模型，distance_tread， 以及人工标注的细胞坐标
    def __init__(self, data_path, model_path, distance_thread, conf_thread):
        self.data_path = data_path
        # self.label_path = label_path
        self.model_path = model_path
        self.distance_thread = distance_thread
        self.conf_thread = conf_thread

    def cut(self, data):
        # 去除标记了两遍的胞体
        data0 = []
        data = np.array(data)
        while len(data):
            data0.append(data[0].tolist())
            if len(data) == 1:
                break
            distance = euclidean_distances(data[:1], data[1:]).reshape(-1)
            data = data[1:][distance > 3]
        return data0


    def evalue(self, pred, label, distance_thread, image_size):
        pred, label = np.array(pred), np.array(label)
        # 把pred中冗余位置去掉
        # 转化为像素坐标
        # pred = pred / [0.65,0.65,2]
        p = pred - (image_size // [256, 256, 92])*[256, 256, 92]
        p = np.max(p, 1)
        pred = pred[np.where(p < 0)[0]]

        # 转化为空间坐标
        pred = pred * [0.65,0.65,2]
        label = label * [0.65,0.65,2]

        distance_mat = euclidean_distances(pred, label)
        # 第一行是第一个pred到所有label的距离，第一列是第一个label到所有pred的距离
        t = np.min(distance_mat, 0)
        # pred 到label的最短距离
        tp = len(np.where(t <= distance_thread)[0])
        fn = len(np.where(t > distance_thread)[0])
        t1 = np.min(distance_mat, 1)
        # label到pred的距离
        fp = len(np.where(t1 > distance_thread)[0])
        precision, recall = tp/(tp+fp), tp/(fn +tp)
        f1 = 2*precision*recall/(precision + recall + 1e-6)
        return precision, recall, f1, tp, fp, fn

    def run(self):
        # 获取检测的结果
        image_path = os.path.join(self.data_path, 'fMOST.tif')
        a = detection(image_path=image_path, model_path=self.model_path, thread_distance=8,
                  thread_conf=self.conf_thread, batch_size=2)
        detection0 = a.run()
        detection0 = np.array(detection0[:,:3])
        # detection0 = self.cut(detection0)

        # 获取标记的结果
        label = combine(self.data_path)
        label = self.cut(label)
        image = sitk.ReadImage(image_path)
        image_size0 = image.GetSize()
        image_size = np.array(image_size0)
        get_labeled_image(label, image_size0, image_path, 'label')
        get_labeled_image(detection0, image_size0, image_path, 'pred')
        precision, recall, f1, tp, fp, fn= self.evalue(detection0, label, self.distance_thread, image_size)
        return precision, recall, f1, tp, fp, fn

def evalue(model_path, data_path, name_list, thread_list):
    model_list = [os.path.join(model_path, filename) for filename in os.listdir(model_path)]
    data_list = [os.path.join(data_path,name_list0) for name_list0 in name_list]
    f = open(os.path.join(data_path, 'result.txt'), 'w')
    for model_path0 in model_list:
        for thread in thread_list:
            tp, fp, fn = 0, 0, 0
            for data_path0 in data_list:
                s = evalue_detection(data_path0, model_path0, 8, thread)
                _, _, _, tp0, fp0, fn0 = s.run()
                tp, fp, fn = tp + tp0, fp + fp0, fn + fn0
                print(tp, fp, fn)
            precision, recall = tp/(tp+fp), tp/(fn +tp)
            f1 = 2*precision*recall/(precision + recall)
            line = 'model_name : {} , thread : {} , precision : {} , recall : {} , f1 : {}'.format(
                model_path0.split('\\')[-1], thread, precision, recall, f1
            )
            print(line)
            f.writelines(line + '\n')
    f.close()




if __name__ == '__main__':
    # model_path = r'G:\item\neuron_matching_v2\deep_detection\saved_model_3/60.pth'
    # data_path = r'G:\item\neuron_matching_v2\deep_detection\train_data\202278'
    # # data = combine(r'G:\item\neuron_matching_v2\deep_detection\train_data\202279')
    # s = evalue_detection(data_path=data_path, model_path=model_path, distance_thread=8)
    # precision,recall,f1, _, _, _ = s.run()
    # print(precision, recall, f1)
    model_path = r'Z:\PubData\wwli\det\model'
    data_path = r'Z:\PubData\wwli\det\test_data'
    name_list = ['202277','202278']
    thread_list = [0.01, 0.5, 0.1, 0.2, 0.5]
    evalue(model_path, data_path, name_list, thread_list)
