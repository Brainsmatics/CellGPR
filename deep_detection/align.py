import numpy as np
import os
import pandas as pd
# 对预测结果进行归并，合并成一个大的csv或者swc文件
# 李文伟 2020.09.05

class align:
    def __init__(self, label_path, crop_nums, step=[256,256,92], resolution=[0.65,2], output_path = './crop.swc'):
        self.label_path = label_path
        self.crop_nums = crop_nums
        # crop_nums对应xyz三个方向的切片的个数
        # step对应xyz三个方向的两两数据块切片移动的像素大小
        self.step = step
        self.resolution = resolution
        self.output_path = output_path


    def get_label(self, label_path):
        count = 0
        for file_name in os.listdir(label_path):
            if file_name.endswith('.csv'):
                count += 1

        label_list = []

        for j in range(count):
            label_name = os.path.join(label_path, '{}.csv'.format(j))
            label = []
            if os.path.exists(label_name):
                df = pd.read_csv(label_name)
                for i in range(df.shape[0]):
                    label.append([df.loc[i, 'X'], df.loc[i, 'Y'], df.loc[i, 'Slice'], 1])
            else:
                print('{} not exists!'.format(label_name))
            label_list.append(label)
        return label_list

    def get_aligned(self, label_list, crop_nums, step):
        label0 = []
        # 拆分顺序 zyx
        for i, label in enumerate(label_list):
            z_nums = i % crop_nums[2]
            x_nums = i // (crop_nums[1] * crop_nums[2])
            y_nums = (i - x_nums*(crop_nums[1] * crop_nums[2])-z_nums)//crop_nums[2]

            # print(z_nums, y_nums, x_nums)
            if label != []:
                for point in label:
                    x = point[0] + x_nums * step[0]
                    y = point[1] + y_nums * step[1]
                    z = point[2] + z_nums * step[2]
                    label0.append([x, y, z])

        return label0


    def write_file(self, label, resolution, output_path):
        # 将结果保存在swc当中，或者生成tif文件
        f1 = open('crop.swc', 'w')
        for i, point in enumerate(label):
            str1 = '{} 1 {} {} {} 1 -1\n'.format(i + 1,
                                                 str(point[0]*resolution[0]), str(point[1]*resolution[0]), str(point[2]*resolution[1]))
            f1.writelines(str1)
        f1.close()



    def run(self):
        label_list = self.get_label(self.label_path)
        label = self.get_aligned(label_list, self.crop_nums, self.step)
        self.write_file(label, self.resolution, self.output_path)



if __name__ == "__main__":
     a = align(label_path='./train_data', crop_nums=[5,3,4], step=[256,256,92])
     a.run()

