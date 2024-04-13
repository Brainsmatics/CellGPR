from torch import nn
import torch
import SimpleITK as sitk
from feature_extract_net import Model_net
import torch.backends.cudnn as cudnn
import os
from torchvision import transforms


'''
根据输入的图像进行切块并完成预测
改进1：切块方法（根据结果来改）
改进2：预测结果加速
'''

class detection:

    def __init__(self, image_path, model_path, thread_conf, thread_distance, resolution=[0.65,2]):
        self.image_path = image_path
        self.model_path = model_path
        self.thread_conf = thread_conf
        self.thread_distance = thread_distance
        self.transform = transforms.ToTensor()
        self.resolution = resolution


    def get_split(self, image):
        # 将图像进行分块计算
        pass

    def regroup(self, label_list):
        # 对计算出的label坐标进行重新组合
        pass

    def get_distance(self, max_pred, pred_list, resolution):
        distance = []
        x0,y0,z0 = max_pred[0],max_pred[1],max_pred[2]
        for i in range(pred_list.shape[0]):
            x1, y1, z1 = pred_list[i][0], pred_list[i][1], pred_list[i][2]

            distance0 = torch.sqrt(torch.sum(torch.square((x1-x0)*resolution[0])+torch.square((y1-y0)*resolution[0])
                                             +torch.square((z1-z0)*resolution[1])))
            distance.append(distance0)
        return torch.tensor(distance)
    #
    def non_max_suppression(self, pred, conf_thres=0.4960, distance_thres=12):
        # 转化为坐标
        pred0 = pred.view(pred.shape[0], -1)
        # 对置信度进行筛选
        conf_mask = pred0[3]>conf_thres
        pred0 = pred0[:,conf_mask].permute(1,0)
        # pred0 = pred0[:, conf_mask]
        # 进行非极大值抑制
        # 按照置信度排序
        _, conf_sort_index = torch.sort(pred0[:, 3], descending=True)
        pred0 = pred0[conf_sort_index]
        max_detections = []

        while pred0.size(0):
            max_detections.append(pred0[0])
            if len(pred0) == 1:
                break
            distance = self.get_distance(max_pred=pred0[0], pred_list=pred0[1:], resolution=[i*4 for i in self.resolution])
            # 判断距离，距离太小去除
            pred0 = pred0[1:][distance > distance_thres]

        max_detections = torch.cat(max_detections).data
        return max_detections

    def decode(self, pred):
        pred = pred.squeeze().cpu()
        pred_x, pred_y, pred_z = torch.sigmoid(pred[0]), torch.sigmoid(pred[1]), torch.sigmoid(pred[2])
        conf = torch.sigmoid(pred[3])
        grid_x = torch.linspace(0, 63, 64).repeat(64, 1).repeat(23,1,1)
        grid_y = torch.linspace(0, 63, 64).repeat(64, 1).T.repeat(23,1,1)
        grid_z = torch.linspace(0,22,23).repeat(64,1).repeat(64,1,1).permute(2,0,1)

        pred0 = torch.stack([pred_x+grid_x, pred_y+grid_y, pred_z+grid_z, conf], dim=0)
        return pred0

    def get_labeled_image(self, detection):
        label_image = sitk.Image([256,256,92], sitk.sitkUInt16)
        if detection != []:
            detection = torch.round(detection)
            for i in range(detection.shape[0]):
                x, y, z = int(detection[i][0]), int(detection[i][1]), int(detection[i][2])
                for x1 in range(max(0, x-1),min(255, x+2)):
                    for y1 in range(max(0, y-1),min(255, y+2)):
                        for z1 in range(max(0, z-1),min(91, z+2)):
                            label_image[x1, y1, z1] = 65535
        sitk.WriteImage(label_image, self.image_path[:-4]+ '_label.tif')

    def write_swc(self, detection):
        detection = detection.tolist()
        f1 = open(self.image_path[:-4] + '.swc', 'w')
        for i, point in enumerate(detection):
            str1 = '{} 1 {} {} {} 1 -1\n'.format(i + 1,
                                                 str(point[0]*self.resolution[0]), str(point[1]*self.resolution[0]), str(point[2]*self.resolution[1]))
            f1.writelines(str1)
        f1.close()






    def run(self):
        # 读取图片进行转化
        image = sitk.ReadImage(self.image_path) #x,y,z
        image = sitk.GetArrayFromImage(image).astype('float32') #z,y,x
        image = self.transform(image)
        image = image.permute(1,0,2).unsqueeze(0).unsqueeze(0).cuda() #z,x,y

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加快模型训练的效率
        net = Model_net([3,8])
        self.net = net.to(device)
        cudnn.benchmark = True

        print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        # self.net = nn.DataParallel(self.net.eval())
        self.net = self.net.eval()

        pred = self.net(image)

        # 进行解码
        pred0 = self.decode(pred)
        if pred0[3].max()>self.thread_conf:

            # 进行非极大值抑制
            detection = self.non_max_suppression(pred=pred0,conf_thres=self.thread_conf,
                                                 distance_thres=self.thread_distance).view(-1, 4)
            detection[:,:3] *= 4
            index = [1,0,2,3]
            detection = detection[:,index]
        else:
            detection = []

        self.get_labeled_image(detection)

        print(detection)
        self.write_swc(detection)






# def
            
if __name__ == "__main__":
    a = detection(image_path='./train_data0/58.tif', model_path='./saved_model/270.pth', thread_distance=8, thread_conf=0.3)
    a.run()
    # a.g


