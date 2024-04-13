from torch import nn
import torch
from collections import OrderedDict


class Residual_Block(nn.Module):
    def __init__(self, inplanes, planes):
        super(Residual_Block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)

        self.conv2 = nn.Conv3d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out

class Model_net(nn.Module):
    def __init__(self, layers):
        # layers 对应每次下采样之后做了多少次残差卷积操作
        super(Model_net, self).__init__()
        self.inplanes = 16
        self.conv_net1 = nn.Sequential(
            nn.Conv3d(1, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(self.inplanes),
            nn.LeakyReLU(0.1)
        )
        self.layer1 = self._make_layer([16, 32], layers[0])
        self.layer2 = self._make_layer([32, 64], layers[1])
        self.conv_net2 = nn.Sequential(
            nn.Conv3d(64, 4, kernel_size=1),
            nn.BatchNorm3d(4),
            nn.LeakyReLU(0.1)
        )
        self.conv_net3 = nn.Conv3d(4,4,kernel_size=1)


    def _make_layer(self, planes, blocks):
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv3d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm3d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        # 加入darknet模块
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), Residual_Block(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))


    def forward(self, x):
        x = self.conv_net1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.conv_net2(x)
        # 此处需要修改，再加一层卷积，但是没有BN和relu
        x = self.conv_net3(x)
        return x



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net1 = Residual_Block(inplanes=3, planes=[3,3]).to(device)
    image = torch.rand([1, 1, 92, 256, 256]).float().to(device)
    # print(net1(image).shape)
    net2 = Model_net(layers=[3, 8]).to(device)
    print(net2(image).shape)


