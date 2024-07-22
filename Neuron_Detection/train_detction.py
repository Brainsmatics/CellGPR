import torch
from torch import nn
from torch.utils.data import DataLoader
from feature_extract_net import Model_net
from load_data import dataset0, dataset0_collate
from get_loss import loss_f
import torch.backends.cudnn as cudnn
from torch import optim
import os
from tensorboardX import SummaryWriter
# from predict0 import detection

result_dir = './result'
if os.path.exists(result_dir):
    for file_name in os.listdir(result_dir):
        os.remove(os.path.join(result_dir, file_name))
else:
    os.mkdir(result_dir)


writer = SummaryWriter(result_dir)



if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    Cuda = True
    batch_size = 1
    data0 = dataset0(src=r'path to train') # path to training data
    validata0 = dataset0(src=r'path to validation') # path to validation data
    lr = 1e-2 # learning rate
    epoch = 100 # train epoch
    save_path = 'Neuron_Detection/model' # path to save model

    data_loader = DataLoader(dataset=data0, shuffle=True, batch_size=batch_size, collate_fn=dataset0_collate)
    vali_data_loader = DataLoader(dataset=validata0, batch_size=torch.cuda.device_count(), collate_fn=dataset0_collate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Model_net([3,8]).train()
    net = net.to(device)
    # state_dict = torch.load('./saved_model/60_1.pth', map_location=device)
    # net.load_state_dict(state_dict)
    net = torch.nn.DataParallel(net)
    # cudnn.benchmark = True

    optimizer = optim.Adam(net.parameters(), lr)
    # optimizer = optim.SGD(net.parameters(), lr)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

    for i in range(epoch):
        total_loss = 0
        total_loss_x, total_loss_y, total_loss_z, total_loss_conf = 0,0,0,0
        for image, label in data_loader:
            image = torch.tensor(image).cuda()
            outputs = net(image)
            # print(outputs)
            a = loss_f(outputs, label, device, 0.2, 6)
            loss, loss_x, loss_y, loss_z, loss_conf = a.get_loss()
            # print(loss)
            total_loss += loss.data
            total_loss_x += loss_x.data
            total_loss_y += loss_y.data
            total_loss_z += loss_z.data
            total_loss_conf += loss_conf.data
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        lr_scheduler.step()
        writer.add_scalar('train/total_loss', total_loss, i)
        writer.add_scalar('train/total_loss_conf', total_loss_conf, i)
        writer.add_scalar('train/total_loss_x', total_loss_x, i)
        writer.add_scalar('train/total_loss_y', total_loss_y, i)
        writer.add_scalar('train/total_loss_z', total_loss_z, i)

        # for j, (name, param) in enumerate(net.named_parameters()):
        #     if 'bn' not in name:
        #         writer.add_histogram(name, param, 0)
        #         writer.add_scalar('loss', total_loss, j)
        print('Epoch {} finished! Total loss is {}'.format(i, total_loss))
        if i % 30 == 0:
            torch.save(net.module.state_dict(), save_path + '/{}.pth'.format(i))

        # do validations
        if i % 5 == 0:
            total_loss = 0
            total_loss_x, total_loss_y, total_loss_z, total_loss_conf = 0, 0, 0, 0
            label_num, pred_num = 0, 0
            for image, label in vali_data_loader:
                image = torch.tensor(image).cuda()
                outputs = net(image)
                # print(outputs)
                a = loss_f(outputs, label, device, 1, 1)
                loss, loss_x, loss_y, loss_z, loss_conf = a.get_loss()
                total_loss += loss.data
                total_loss_x += loss_x.data
                total_loss_y += loss_y.data
                total_loss_z += loss_z.data
                total_loss_conf += loss_conf.data


            writer.add_scalar('validation/validation_total_loss', total_loss, i)
            writer.add_scalar('validation/validation_total_loss_conf', total_loss_conf, i)
            writer.add_scalar('validation/validation_total_loss_x', total_loss_x, i)
            writer.add_scalar('validation/validation_total_loss_y', total_loss_y, i)
            writer.add_scalar('validation/validation_total_loss_z', total_loss_z, i)
            print('In {} epoch, validation loss is {}'.format(i, total_loss))



writer.close()