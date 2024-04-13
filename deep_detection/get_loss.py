import torch
from torch import nn
import numpy as np

class loss_f():
    def __init__(self, outputs, label, device, lambda_conf=10, thread=10, pred_conf_thread=0.5):
        super(loss_f, self).__init__()
        self.outputs = outputs
        self.label = label
        self.labmda_conf = lambda_conf
        self.thread = thread
        self.device = device
        self.pred_conf_thread = pred_conf_thread

    def _get_target(self, label, thread=1):
        # 根据label生成每个预测点的值，平移量，以及置信度
        mask = torch.zeros([23, 64, 64])
        noobj_mask = torch.ones([23, 64, 64])
        tx = torch.zeros([23, 64, 64])
        ty = torch.zeros([23, 64, 64])
        tz = torch.zeros([23, 64, 64])
        # tconf = torch.zeros([23, 64, 64])
        if label != []:
            label = np.array(label)
            # print(label)
            for j in range(label.shape[0]):
                if label[j][2] == 92:
                    label[j][2] -= 0.01
                if label[j][1] >256:
                    label[j][1] -= 0.1
                if label[j][0] >256:
                    label[j][0] -= 0.1
                x, y, z = int(label[j][0]/4), int(label[j][1]/4), int(label[j][2]/4)
                mask[z, x, y] = 1

                tx[z, x, y] = label[j][0]/4 -x
                ty[z, x, y] = label[j][1]/4 -y
                tz[z, x, y] = label[j][2]/4 -z
                noobj_mask[z, x, y] = 0

        return mask, noobj_mask, tx, ty, tz

    def _get_distance(self, pred, label0):
        label0_expand = label0.expand_as(pred).cuda()
        d1 = torch.square(label0_expand-pred)
        distance = torch.sqrt(torch.sum(d1, 1))
        return distance



    def _get_ignore0(self, noobj_mask, label, output, scale_factor = [0.65, 2]):
        if label != []:
            tx = torch.sigmoid(output[0])
            ty = torch.sigmoid(output[1])
            tz = torch.sigmoid(output[2])
            grid_x = torch.linspace(0, 63, 64).repeat(64, 1).repeat(23,1,1).cuda()
            grid_y = torch.linspace(0, 63, 64).repeat(64, 1).T.repeat(23,1,1).cuda()
            grid_z = torch.linspace(0,22,23).repeat(64,1).repeat(64,1,1).permute(2,0,1).cuda()
            # 预测的实际空间位置
            pred_x, pred_y, pred_z = (tx+grid_x)*4*scale_factor[0], (ty+grid_y)*4*scale_factor[0], (tz+grid_z)*4*scale_factor[1]
            pred = torch.stack([pred_x.view(-1), pred_y.view(-1), pred_z.view(-1)]).T
            label0 = torch.tensor(label)/1.0
            label0 = label0[:,:3]
            for i in range(label0.shape[0]):
                distance = self._get_distance(pred, label0[i])
                distance = distance.view(tx.shape)
                mask = distance < self.thread
                noobj_mask[mask] = 0
        return noobj_mask




    def _get_ignore(self, noobj_mask, label, output, scale_factor = [0.65, 2]):
        # 对output结果进行整合,设定一个阈值，进行过滤
        # 计算每个点的平移量
        tx = torch.sigmoid(output[0])
        ty = torch.sigmoid(output[1])
        tz = torch.sigmoid(output[2])

        # torch.linspace()


        if label != []:
            label = np.array(label)
            for j in range(label.shape[0]):
                # 计算偏移距离
                x, y, z = int(label[j][0]/4), int(label[j][1]/4), int(label[j][2]/4)
                x0, y0, z0 = label[j][0], label[j][1], label[j][2]
                for ix in range(max(0, x-6), min(x+6, self.outputs.shape[3])):
                    for iy in range(max(0, y-6), min(y+6, self.outputs.shape[4])):
                        for iz in range(max(0, z-3), min(z+3, self.outputs.shape[2])):
                            # 计算出预测点的位置
                            # print('{},{},{}'.format(ix, iy, iz))
                            x1 = (ix + tx[iz, ix, iy])*4
                            y1 = (iy + ty[iz, ix, iy])*4
                            z1 = (iz + tz[iz, ix, iy])*4


                            distance = torch.sqrt(torch.sum(torch.square((x0-x1)*scale_factor[0]) + torch.square((y0-y1)*scale_factor[0])
                                                 + torch.square((z0-z1)*scale_factor[1])))

                            if distance < self.thread:
                                noobj_mask[iz, ix, iy] = 0

        return noobj_mask



    def forward(self, input, target = None):
        pass

    def get_loss(self):
        device = self.device
        loss = 0
        total_loss_x, total_loss_y, total_loss_z = 0, 0, 0
        total_loss_conf = 0
        for i in range(self.outputs.shape[0]):
            label0 = self.label[i]
            mask, noobj_mask, tx, ty, tz = self._get_target(label0)
            noobj_mask = self._get_ignore0(noobj_mask, label0, self.outputs[i])

            tx, ty, tz = tx.to(device), ty.to(device), tz.to(device)
            mask, noobj_mask = mask.to(device), noobj_mask.to(device)

            # 对output进行处理
            pred_x, pred_y, pred_z = torch.sigmoid(self.outputs[i][0]), torch.sigmoid(self.outputs[i][1]), torch.sigmoid(self.outputs[i][2])
            conf = torch.sigmoid(self.outputs[i][3])

            # 对于有物体的地方计算xyz的损失函数
            loss_x = nn.BCELoss()(pred_x*mask, tx*mask)
            loss_y = nn.BCELoss()(pred_y*mask, ty*mask)
            loss_z = nn.BCELoss()(pred_z*mask, tz*mask)

            # 对于没有物体的地方以及有物体的地方计算置信度损失函数
            loss_conf = nn.BCELoss()(conf*mask, mask) + nn.BCELoss()(conf*noobj_mask, mask)
            loss += loss_x + loss_y + loss_z + self.labmda_conf*loss_conf
            total_loss_x += loss_x
            total_loss_y += loss_y
            total_loss_z += loss_z
            total_loss_conf += loss_conf

        # print('Total loss is {}'.format(loss))

        return loss, total_loss_x, total_loss_y, total_loss_z, total_loss_conf


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    outputs = torch.rand([2,4,23,64,64]).to(device)
    label = np.array([[[212, 204, 34,1], [137,159,12,1]], []])
    a = loss_f(outputs=outputs, label=label, device=device, lambda_conf=1, thread=10)
    a.get_loss()

