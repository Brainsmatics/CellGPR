import numpy as np
from tensorboardX import SummaryWriter
import torch
import os
from feature_extract_net import Model_net
from torchvision import models

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

result_dir = './log0'

net = Model_net([2,8])

# 清理文件夹
if os.path.exists(result_dir):
    for file_name in os.listdir(result_dir):
        os.remove(os.path.join(result_dir, file_name))
else:
    os.mkdir(result_dir)

writer = SummaryWriter(result_dir, flush_secs=1)

for i in range(1000):
    writer.add_scalar('test', np.random.rand(), i)
    writer.add_scalar('test1', np.sin(np.random.rand()), i)
    writer.add_scalar('test2', torch.rand(1), i)
    # writer.add_histogram('hist', torch.rand(1000))
    # writer.add_image('image', torch.rand([100,100]))
for i in range(10):
    x = np.random.random(1000)
    writer.add_histogram('distribution centers', x + i, i)

# 显示每个layer的权重
model = models.vgg16(pretrained=True)
loss = 10   # 第0层
for i, (name, param) in enumerate(model.named_parameters()):
    if 'bn' not in name:
        writer.add_histogram(name, param, 0)
        writer.add_scalar('loss', loss, i)
        loss = loss*0.5


# 添加网络图
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image = torch.rand([1, 1, 92, 256, 256]).float().to(device)
writer.add_graph(net, (image,))

# feature map 可视化，暂时用不到，跳过

writer.close()