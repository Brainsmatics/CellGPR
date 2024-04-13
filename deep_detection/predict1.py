'''
在原来predict的基础上进行改进，可以实现整个数据块的预测
1.修正边界处的结果，0.5个数据块为一个单位，
2.得到所有的结果，设定阈值，进行过滤，过滤掉重复的结果
'''
import sys
from numpy.lib.function_base import append
sys.path.append(r'G:\item\neuron_matching_v3')
from torch import nn
import torch
import SimpleITK as sitk
from deep_detection.feature_extract_net import Model_net
import torch.backends.cudnn as cudnn
import os
from torchvision import transforms
from deep_detection.predict_dataset1 import predict_dataset
from torch.utils.data import DataLoader


class detection:

    def __init__(self, image_path, model_path, thread_conf, thread_distance, batch_size, resolution=[0.65, 2]):
        self.image_path = image_path
        self.model_path = model_path
        self.thread_conf = thread_conf
        self.thread_distance = thread_distance
        self.transform = transforms.ToTensor()
        self.resolution = resolution
        self.batch_size = batch_size

    def get_distance(self, max_pred, pred_list, resolution):
        distance = []
        x0, y0, z0 = max_pred[0], max_pred[1], max_pred[2]
        for i in range(pred_list.shape[0]):
            x1, y1, z1 = pred_list[i][0], pred_list[i][1], pred_list[i][2]

            distance0 = torch.sqrt(
                torch.sum(torch.square((x1 - x0) * resolution[0]) + torch.square((y1 - y0) * resolution[0])
                          + torch.square((z1 - z0) * resolution[1])))
            distance.append(distance0)
        return torch.tensor(distance)

    def non_max_suppression(self, pred, conf_thres=0.5, distance_thres=12):
        # 转化为坐标
        pred0 = pred.view(pred.shape[0], -1)
        # 对置信度进行筛选
        conf_mask = pred0[3]>conf_thres
        pred0 = pred0[:,conf_mask].permute(1,0)
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

    def non_max_suppression_total(self, detection_list, distance_thres=12):
        detection_list = torch.tensor(detection_list)
        # 按照置信度排序
        _, conf_sort_index = torch.sort(detection_list[:, 3], descending=True)
        detection0 = detection_list[conf_sort_index]
        max_detections = []
        while detection0.size(0):
            max_detections.append(detection0[0])
            if len(detection0) == 1:
                break
            distance = self.get_distance(max_pred=detection0[0], pred_list=detection0[1:],
                                         resolution=self.resolution)
            # 判断距离，距离太小去除
            detection0 = detection0[1:][distance > distance_thres]

        max_detections = torch.cat(max_detections).data
        return max_detections



    def decode(self, pred):
        pred = pred.squeeze().cpu()
        pred_x, pred_y, pred_z = torch.sigmoid(pred[0]), torch.sigmoid(pred[1]), torch.sigmoid(pred[2])
        conf = torch.sigmoid(pred[3])
        grid_x = torch.linspace(0, 63, 64).repeat(64, 1).repeat(23, 1, 1)
        grid_y = torch.linspace(0, 63, 64).repeat(64, 1).T.repeat(23, 1, 1)
        grid_z = torch.linspace(0, 22, 23).repeat(64, 1).repeat(64, 1, 1).permute(2, 0, 1)

        pred0 = torch.stack([pred_x + grid_x, pred_y + grid_y, pred_z + grid_z, conf], dim=0)
        return pred0

    def get_labeled_image(self, detection, image_size):
        label_image = sitk.Image(image_size, sitk.sitkUInt16)
        if detection != []:
            detection = torch.round(detection)
            for i in range(detection.shape[0]):
                x, y, z = int(detection[i][0]), int(detection[i][1]), int(detection[i][2])
                for x1 in range(max(0, x-1),min(image_size[0], x+2)):
                    for y1 in range(max(0, y-1),min(image_size[1], y+2)):
                        for z1 in range(max(0, z-1),min(image_size[2], z+2)):
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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 加快模型训练的效率
        net = Model_net([3, 8])
        self.net = net.to(device)
        cudnn.benchmark = True

        # print('Loading weights into state dict...')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        # self.net = nn.DataParallel(self.net.eval())
        self.net = self.net.eval()

        detection_list = []


        # 读取图片进行转化
        dataset = predict_dataset(self.image_path)
        image_size = dataset.get_image_size()
        dataloader = DataLoader(dataset=dataset, shuffle=False, batch_size=1)
        for image, order in dataloader:
            # print(order)
            # image = self.(image)
            image = image.unsqueeze(1).to(device) #z,x,y
            # image = image.permute(1, 0, 2).unsqueeze(0).unsqueeze(0).cuda()
            pred = self.net(image)
            for pred0, order0 in zip(pred, order):
                # 进行解码
                pred0 = self.decode(pred0)
                if pred0[3].max() > self.thread_conf:

                    # 进行非极大值抑制
                    detection = self.non_max_suppression(pred=pred0, conf_thres=self.thread_conf,
                                                         distance_thres=self.thread_distance).view(-1, 4)
                    detection[:, :3] *= 4
                    index = [1, 0, 2, 3]
                    detection = detection[:, index]
                    # print(detection[0])
                    detection[:, :3] += order0
                    # print(order0)
                    # print(detection[0])
                    for detection0 in detection:
                        detection_list.append(detection0.tolist())
                    # print(detection)
                else:
                    detection = []

            # 对所有detection做非极大值抑制
        if detection_list is not []:
            # print(torch.max(torch.tensor(detection_list)[:,0]))
            detection_list = self.non_max_suppression_total(detection_list, distance_thres=self.thread_distance).view(-1,4)
        # print(detection_list)
        # detection_list = torch.tensor(detection_list)
        self.get_labeled_image(detection_list, image_size)
        self.write_swc(detection_list)
        print('detection complete! there are {} somas detected'.format(len(detection_list)))
        return detection_list

            # self.get_labeled_image(detection)
            #
            # print(detection)
            # self.write_swc(detection)




if __name__ == "__main__":
    a = detection(image_path='../graph/neuron_detector/fMOST.tif', model_path='saved_model_3/60.pth', thread_distance=10,
                  thread_conf=0.1, batch_size=2)
    a.run()