import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch

class predict_dataset(Dataset):
    def __init__(self, image_path, step = [128,128,46]):
        self.step = step
        image = sitk.ReadImage(image_path)
        self.image_array = sitk.GetArrayFromImage(image).astype('float32')
        self.order = self.get_order(self.image_array)

    def get_order(self, image_array):
        order = []
        z, y, x = image_array.shape
        step = self.step
        x_num, y_num, z_num = (x-256)//step[0]+1, (y-256)//step[1]+1, (z-92)//step[2]+1
        print('x_num={}, y_num={},z_num={}'.format(x_num, y_num, z_num))
        for z1 in range(z_num+1):
            z0 = z1*step[2] if z1 != z_num else z-92
            for y1 in range(y_num+1):
                y0 = y1*step[1] if y1 != y_num else y-256
                for x1 in range(x_num+1):
                    x0 = x1*step[0] if x1 != x_num else x-256
                    # image0 = image_array[z0:z0+96, y0:y0+256, x0:x0+256]
                    # image_list.append(image0)
                    order.append([x0, y0, z0])
        # print(order)
        # print(len(order))
        # print('*'*100)
        return torch.tensor(order)

    def get_image_size(self):
        return self.image_array.shape[::-1]


    def __len__(self):
        return len(self.order)

    def __getitem__(self, item):
        x0, y0, z0 = self.order[item]
        image_tensor = torch.tensor(self.image_array[z0:z0+92, y0:y0+256, x0:x0+256]).permute([0,2,1])
        return image_tensor, self.order[item]

if __name__ == "__main__":
    image_path = './level2_crop_1.tif'
    image_dataset = predict_dataset(image_path=image_path, step=[200,200,70])
    batch_size = 4
    image_size = image_dataset.get_image_size()
    print(image_size)
    dataloader = DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    # order = image_dataset.get_order()
    for image, index in dataloader:
        # pass
        print(image.shape)
        print(index)