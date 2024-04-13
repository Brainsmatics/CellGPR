import torch
from torch import nn
from torch import optim
from feature_extract_net import Model_net
from test_net import test_net
# 测试BCELoss在随机数据上会不会出现
from tensorboardX import SummaryWriter
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

result_dir = './result'
if os.path.exists(result_dir):
    for file_name in os.listdir(result_dir):
        os.remove(os.path.join(result_dir, file_name))
else:
    os.mkdir(result_dir)
writer = SummaryWriter(result_dir)


loss_f = nn.BCELoss()
epoch = 10000

tensor1 = torch.zeros(23,64,64).cuda()
net = nn.Sequential(
    nn.Linear(100, 100),
    nn.LeakyReLU(0.1),
    nn.Linear(100, 20),
    nn.LeakyReLU(0.1),
)
net1 = nn.Sequential(
    nn.Conv2d(1, 6, stride=1, kernel_size=1),
    nn.BatchNorm2d(6),
    nn.LeakyReLU(0.1),
    nn.Conv2d(6, 16, stride=1, kernel_size=1),
    nn.BatchNorm2d(16),
    nn.LeakyReLU(0.1),
    nn.Conv2d(16, 1, stride=1, kernel_size=1),
    nn.BatchNorm2d(1),
    nn.LeakyReLU(0.1),
    nn.Conv2d(1, 1, stride=1, kernel_size=1)
)

lr = 1e-3
# net = test_net().cuda()
net = Model_net([3,8]).cuda()
net = nn.DataParallel(net)
mask = torch.zeros([23,64,64]).cuda()
noobj_mask = torch.ones([23,64,64]).cuda()


optimizer = optim.Adam(net.parameters(), lr)

for i in range(epoch):
    image = torch.rand([1, 1, 92, 256, 256]).cuda()
    # net = Model_net([3, 8]).cuda()
    output = net(image).squeeze()
    output = torch.sigmoid(output[3])
    # output = output[3]
    loss = loss_f(output*mask, mask) + loss_f(output*noobj_mask, mask)
    print(loss.item())
    writer.add_scalar('loss', loss.item(), i)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()