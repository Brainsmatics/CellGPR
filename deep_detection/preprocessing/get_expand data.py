import numpy as np
import SimpleITK as sitk
import pandas as pd
import os
import copy

# 对数据集和label进行旋转翻转操作，得到扩充的数据集

class get_expand_data:
    def __init__(self, src):
        self.src = src
        self.image_list, self.label_list = self._get_name()
        self.image_size = sitk.ReadImage(self.image_list[0]).GetSize() #依次对应xyz

    def _get_name(self):
        files = os.listdir(self.src)
        image_list, label_list = [], []
        for f in files:
            if f.endswith('.tif'):
                image_list.append(os.path.join(self.src, f))
                label_list.append(os.path.join(self.src, f[:-3] + 'csv'))
        return image_list, label_list

    def _get_label_filp(self, label_name, image_size):
        df = pd.read_csv(label_name)
        axis = ['X', 'Y', 'Slice']
        for i, ax in enumerate(axis):
            df1 = copy.deepcopy(df)
            for j in range(df.shape[0]):
                df1.loc[j,ax] = image_size[i] - df1.loc[j,ax]
            df1.to_csv(label_name[:-4]+'_flip_{}.csv'.format(ax), index=0)

        # df_filp_x = df
    def _get_label_rot(self, label_name, image_size):
        df = pd.read_csv(label_name)
        rotate_angle = [90, 180, 270]
        rot_mat = np.array([[0,-1],[1,0]])
        trans_mat = np.array([[image_size[1], 0]]).T
        for i in rotate_angle:
            df1 = df
            for j in range(df.shape[0]):
                x, y = df1.loc[j,'X'], df1.loc[j,'Y']
                t = rot_mat.dot(np.array([[x, y]]).T) + trans_mat
                df1.loc[j, 'X'], df1.loc[j, 'Y'] = t.squeeze()[0], t.squeeze()[1]
            df1.to_csv(label_name[:-4] + '_rot_{}.csv'.format(i), index=0)
            df = df1


    # def _get_

    def run(self):
        # image_data = []
        for (image_name, label_name) in zip(self.image_list, self.label_list):
            image = sitk.ReadImage(image_name)
            # 进行翻转操作
            sitk.WriteImage(image[::-1,:,:], image_name[:-4] + '_flip_X.tif')
            sitk.WriteImage(image[:,::-1,:], image_name[:-4] + '_flip_Y.tif')
            sitk.WriteImage(image[:,:,::-1], image_name[:-4] + '_flip_Slice.tif')
            if os.path.exists(label_name):
                self._get_label_filp(label_name, self.image_size)

            # 进行旋转操作
            image_array = sitk.GetArrayFromImage(image)

            image1_array = np.rot90(image_array.swapaxes(0, 2)).swapaxes(0, 2)
            sitk.WriteImage(sitk.GetImageFromArray(image1_array), image_name[:-4] + '_rot_90.tif')

            image2_array = np.rot90(image1_array.swapaxes(0, 2)).swapaxes(0, 2)
            sitk.WriteImage(sitk.GetImageFromArray(image2_array), image_name[:-4] + '_rot_180.tif')

            image3_array = np.rot90(image2_array.swapaxes(0, 2)).swapaxes(0, 2)
            sitk.WriteImage(sitk.GetImageFromArray(image3_array), image_name[:-4] + '_rot_270.tif')

            if os.path.exists(label_name):
                self._get_label_rot(label_name, self.image_size)







if __name__ == '__main__':
    a = get_expand_data(src = r'G:\item\neuron_matching_v2\deep_detection\split')
    a.run()
    print('*'*100)

