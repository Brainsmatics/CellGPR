import torch
import os
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import numpy as np



class dataset0(Dataset):
    def __init__(self, src):
        super(dataset0, self).__init__()
        self.src = src
        self.image_list, self.label_list = self._get_name()
        self.image_size = sitk.ReadImage(self.image_list[0]).GetSize() #依次对应xyz
        # self.transform = transforms.Compose([transforms.ToTensor(),
        #                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.transform = transforms.ToTensor()

    def _get_name(self):
        files = os.listdir(self.src)
        image_list, label_list = [], []
        for f in files:
            if f.endswith('.tif'):
                image_list.append(os.path.join(self.src, f))
                label_list.append(os.path.join(self.src, f[:-3] + 'csv'))
        return image_list, label_list

    def _get_label(self, label_name):
        label = []
        df = pd.read_csv(label_name)
        for i in range(df.shape[0]):
            label.append([df.loc[i,'X'], df.loc[i,'Y'], df.loc[i,'Slice'], 1])
        return label

    def __getitem__(self, item):
        # print(self.image_list[item])
        image = sitk.ReadImage(self.image_list[item])
        image = sitk.GetArrayFromImage(image).astype('float32')
        image = self.transform(image)
        image = image.permute(1,0,2).unsqueeze(0)
        # label = torch.tensor([])
        label = []
        if os.path.exists(self.label_list[item]):
            label = self._get_label(self.label_list[item])
        # return image, torch.tensor(label).float()
        label = np.array(label, dtype='float32')
        image = np.array(image, dtype='float32')
        return image, label


    def __len__(self):
        return len(self.image_list)

def dataset0_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = np.array(images)
    bboxes = np.array(bboxes)
    return images, bboxes

if __name__ == '__main__':
    data0 = dataset0(src='./train_data1')
    data_loader = DataLoader(dataset=data0, shuffle=True, batch_size=8, collate_fn=dataset0_collate)
    for image, label in data_loader:
        print(image.shape)
        # print(label)