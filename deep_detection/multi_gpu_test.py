import torch
from torch import nn
from torch.utils.data import DataLoader
from feature_extract_net import Model_net
from load_data import dataset0, dataset0_collate
from get_loss import loss_f
import torch.backends.cudnn as cudnn
from torch import optim
import os
from torch.utils.data.dataset import Dataset
from tensorboardX import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
input_size = 5000
output_size = 200

batch_size = 3000
data_size = 1000000
epoch = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


result_dir = './result_test'
if os.path.exists(result_dir):
    for file_name in os.listdir(result_dir):
        os.remove(os.path.join(result_dir, file_name))
else:
    os.mkdir(result_dir)


writer = SummaryWriter(result_dir)

class RandomDataset(Dataset):
    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

class Model(nn.Module):
# Our model
    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
model = nn.DataParallel(model)

model.to(device)
rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),batch_size=batch_size, shuffle=True)
loss_func = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

for i in range(epoch):
    for data in rand_loader:
        input = data.to(device)
        output = model(input)
        label = torch.ones(output.shape).to(device)
        # print("Outside: input size", input.size(),
        # "output_size", output.size())
        loss = loss_func(output, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Epoch {} finished! Total loss is {}'.format(i+1, loss.cpu()))
        writer.add_scalar('loss', loss, i)


