import SimpleITK as sitk

# 对数据进行切块

image = sitk.ReadImage('../test/fMOST.tif')
image = sitk.ReadImage(r'G:\item\neuron_matching_v2\deep_detection\test\fMOST.tif')
dst = '../train_data/202279/split'
dst = r'G:\item\neuron_matching_v2\deep_detection\split'
size_x, size_y, size_z = image.GetSize()
num = 0
for i in range(size_x//256):
    for j in range(size_y//256):
        for k in range(size_z//92):
            image0 = image[i*256:(i+1)*256, j*256:(j+1)*256, k*92:(k+1)*92]
            sitk.WriteImage(image0, dst + '/{}.tif'.format(num))
            num += 1

class get_data_for_train:
    def __init__(self, data_path, dst, get_image_data=True, get_expand_data=False):
        self.data_path = data_path
        self.dst = dst
        self.get_image_data = get_image_data
        self.get_expand_data = get_expand_data

    def crop_image(self):
        for i in range(size_x // 256):
            for j in range(size_y // 256):
                for k in range(size_z // 92):
                    image0 = image[i * 256:(i + 1) * 256, j * 256:(j + 1) * 256, k * 92:(k + 1) * 92]
                    sitk.WriteImage(image0, self.dst + '/{}.tif'.format(num))
                    num += 1

    def _get_names(self):
        files = os.listdir(self.data_path)
        image_list, label_list = [], []
        for f in files:
            if f.endswith('.tif'):
                image_list.append(f)
                label_list.append(f[:-3] + 'csv')
        return image_list, label_list

    def get_expand_train_data(self, image, label):
        pass



    def run(self):
        if self.get_image_data:
            self.crop_image()

        if self.get_expand_data:
            image_list, label_list = self._get_names()
            for (image, label) in zip(image_list, label_list):
                print(image.shape)




print("*"*100)