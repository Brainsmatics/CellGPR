import SimpleITK as sitk
from torchvision import transforms

image = sitk.ReadImage('./level2_crop_1.tif')
print('SimpleITK shape is {}'.format(image.GetSize()))

image_array = sitk.GetArrayFromImage(image).astype('float')
print('numpy shape is {}'.format(image_array.shape))

transform = transforms.ToTensor()
image_tensor = transform(image_array)
print('torch tensor shape is {}'.format(image_tensor.shape))

image_tensor1 = image_tensor.permute(1,0,2)
print(image_tensor1.shape)

image_tensor2 = image_tensor.unsqueeze(1)
print(image_tensor2.shape)


