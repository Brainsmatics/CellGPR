import SimpleITK as sitk
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class predict_dataset(Dataset):
    def __init__(self, image_path, step = [128,128,48]):
        self.image_path = image_path
        self.step = step
        self.image_list, self.order = self.get_image_list(image_path)


    def get_image_list(self, image_path):
        image = sitk.ReadImage(image_path) #读取顺序 xyz
        image_array = sitk.GetArrayFromImage(image).astype('float32') #顺序zyx
        image_list = []
        order = []
        x, y, z = image.GetSize()
        step = self.step
        x_num, y_num, z_num = (x-256)//step[0]+1, (y-256)//step[1]+1, (z-96)//step[2]+1
        print('x_num={}, y_num={},z_num={}'.format(x_num, y_num, z_num))
        for z1 in range(z_num+1):
            z0 = z1*48 if z1 != z_num else z-96
            for y1 in range(y_num+1):
                y0 = y1*128 if y1 != y_num else y-256
                for x1 in range(x_num+1):
                    x0 = x1*128 if x1 != x_num else x-256
                    image0 = image_array[z0:z0+96, y0:y0+256, x0:x0+256]
                    image_list.append(image0)
                    order.append([x0, y0, z0])
        print(order)
        print(len(order))
        print('*'*100)
        return image_list, order

    # def get_order(self):
    #     return self.order

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        return self.image_list[item], self.order[item]

if __name__ == "__main__":
    image_path = './level2_crop.tif'
    image_dataset = predict_dataset(image_path=image_path)
    batch_size = 4
    dataloader = DataLoader(dataset=image_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    # order = image_dataset.get_order()
    for image, index in dataloader:
        # pass
        print(image.shape)
        print(index)