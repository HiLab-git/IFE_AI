import sys
import csv
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from tensorboardX import SummaryWriter
from libMIC.data_loader import *
from libMIC.networks.vgg_backup import VGGNet
from libMIC.parse_config import parse_config

def get_cross_entropy_loss(predict, y):
    y_soft0 = y.new_ones(y.size()) * (1.0 - y)
    y_soft1 = y.new_ones(y.size()) * (y)
    y_soft = torch.cat((y_soft0, y_soft1), -1)
    y_soft = y_soft.double()
    ce = - torch.log(predict) * y_soft
    ce = torch.sum(ce, dim = -1)
    ce = torch.mean(ce)
    return ce

def get_average_accuracy(predict, y):
    predict = torch.argmax(predict, dim = -1)
    predict = predict.int()
    y = torch.reshape(y, predict.shape)
    accuracy = torch.sum(predict == y)
    accuracy = accuracy.double()
    return accuracy/torch.numel(y)

class TrainInferAgent():
    def __init__(self, config, stage = 'train'):
        self.config = config
        self.stage  = stage
        assert(stage in ['train', 'inference', 'test'])

    def __create_dataset(self):
        root_dir  = self.config['dataset']['root_dir']
        train_csv = self.config['dataset'].get('train_csv', None)
        valid_csv = self.config['dataset'].get('valid_csv', None)
        test_csv  = self.config['dataset'].get('test_csv', None)
    
        if(self.stage == 'train'):
            transform_list = [RandomCrop((180, 230)),
                              Rescale((256, 256)),                               
                              Normalize(mean=[240.18, 238.28, 245.64],
                                        std =[16.74, 21.02, 4.59]),
                              RandomNoise(0.0, 0.05),
                              ToTensor()]
            self.train_dataset = PhrosisDataset(csv_file = train_csv,
                                           root_dir = root_dir,
                                           transform= transforms.Compose(transform_list))
            self.valid_dataset = PhrosisDataset(csv_file = valid_csv,
                                           root_dir = root_dir,
                                           transform= transforms.Compose(transform_list))
            batch_size = self.config['training']['batch_size']
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset, 
                batch_size = batch_size, shuffle=True, num_workers=batch_size * 4)
            self.valid_loader = torch.utils.data.DataLoader(self.valid_dataset, 
                batch_size = batch_size, shuffle=False, num_workers=batch_size * 4)
        else:
            transform_list = [RandomCrop((180, 230)),
                              Rescale((256, 256)),
                              Normalize(mean=[240.18, 238.28, 245.64],
                                        std =[16.74, 21.02, 4.59]),
                              ToTensor()]
            self.test_dataset = PhrosisDataset(csv_file = test_csv,
                                          root_dir = root_dir,
                                          transform= transforms.Compose(transform_list))
            batch_size = 1
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset, 
                batch_size=batch_size, shuffle=False, num_workers=batch_size)

    def __create_network(self):
        # self.net = get_network(self.config['network'])
        if(self.config['network']['net_type'] == 'VGGNet'):
            self.net = VGGNet()
        else:
            raise ValueError("undefined network {0:}".format(self.config['network']['net_type']))
        self.net.double()

    def __create_optimizer(self):
        lr = self.config['training']['learning_rate']
        momentum = self.config['training']['momentum']
        weight_decay = self.config['training']['weight_decay']
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr, weight_decay=weight_decay)
        last_iter = -1
        if(self.checkpoint is not None):
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            last_iter = self.checkpoint['iteration'] - 1
        self.schedule = optim.lr_scheduler.MultiStepLR(self.optimizer,
                self.config['training']['lr_milestones'],
                self.config['training']['lr_gamma'],
                last_epoch = last_iter)

    def __train(self):
        device = torch.device(config['training']['device_name'])
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        print('device', device)
        self.net.to(device)  

        summ_writer = SummaryWriter(config['training']['summary_dir'])
        chpt_prefx  = config['training']['checkpoint_prefix']
        iter_start  = config['training']['iter_start']
        iter_max    = config['training']['iter_max']
        iter_valid  = config['training']['iter_valid']
        iter_save   = config['training']['iter_save']

        if(iter_start > 0):
            checkpoint_file = "{0:}_{1:}.pt".format(chpt_prefx, iter_start)
            self.checkpoint = torch.load(checkpoint_file)
            assert(self.checkpoint['iteration'] == iter_start)
            self.net.load_state_dict(self.checkpoint['model_state_dict'])
        else:
            self.checkpoint = None
        self.__create_optimizer()

        # for debug
        # it = 0
        # for data in self.valid_loader:
        #     it = it + 1
        #     inputs, labels = data['image'], data['label']
        #     inputs_arr = inputs.numpy()
            
        #     print(inputs_arr.shape, labels.shape)
        #     print(inputs.min(), inputs.max(), inputs.mean())
        #     inputs, labels = inputs.to(device), labels.to(device)
        #     outputs = self.net(inputs)
        #     print(outputs.cpu())
        #     loss = get_cross_entropy_loss(outputs, labels)
        #     # img = inputs_arr[0][0]
        #     # img = np.transpose(img, [1, 2, 0])
        #     # img = np.asarray(img, np.uint8)
        #     # plt.imshow(img)
        #     # plt.show()
        #     # img_r = inputs_arr[0][0]
        #     # img_g = inputs_arr[0][1]
        #     # img_b = inputs_arr[0][2]
        #     # plt.subplot(1,3,1); plt.imshow(img_r)
        #     # plt.subplot(1,3,2); plt.imshow(img_g)
        #     # plt.subplot(1,3,3); plt.imshow(img_b)
        #     # plt.show()
        #     # print(img_r.mean(), img_g.mean(), img_b.mean())
        #     if(it ==5):
        #         break
        # #     inputs, labels = inputs.to(device), labels.to(device)
        # #     # print(labels)
        # return

        train_loss = 0.0
        train_acc  = 0.0
        trainIter = iter(self.train_loader)
        for it in range(iter_start, iter_max):  # loop over the dataset multiple times
            try:
                data = next(trainIter)
            except StopIteration:
                trainIter = iter(self.train_loader)
                data = next(trainIter)
                
            # get the inputs
            inputs, labels = data['image'], data['label']
            inputs, labels = inputs.to(device), labels.to(device)
            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.schedule.step()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = get_cross_entropy_loss(outputs, labels)
            acc  = get_average_accuracy(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # print statistics
            train_loss += loss.item()
            train_acc  += acc.item()
            if (it % iter_valid == iter_valid -1): 
                print(outputs.cpu())
                train_avg_loss = train_loss / iter_valid
                train_avg_acc  = train_acc  / iter_valid
                train_loss = 0.0
                train_acc  = 0.0

                valid_loss = 0.0
                valid_acc  = 0.0
                with torch.no_grad():
                    for data in self.valid_loader:
                        inputs, labels = data['image'], data['label']
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = self.net(inputs)
                        loss = get_cross_entropy_loss(outputs, labels)
                        acc  = get_average_accuracy(outputs, labels)
                        valid_loss += loss.item()
                        valid_acc  += acc.item()
                
                valid_avg_loss = valid_loss / len(self.valid_loader)
                valid_avg_acc  = valid_acc  / len(self.valid_loader)
                valid_loss = 0.0
                valid_acc  = 0.0
                print("[%d] loss: %.3f, %.3f, accu:  %.3f, %.3f" %
                    (it + 1, train_avg_loss, valid_avg_loss, train_avg_acc, valid_avg_acc))

                loss_scalers = {'train': train_avg_loss, 'valid': valid_avg_loss}
                accu_scalars = {'train': train_avg_acc,  'valid': valid_avg_acc}
                summ_writer.add_scalars('loss', loss_scalers, it + 1)
                summ_writer.add_scalars('acc',  accu_scalars, it + 1)
            if(it % iter_save ==  iter_save - 1):
                save_dict = {'iteration': it + 1,
                             'model_state_dict': self.net.state_dict(),
                             'optimizer_state_dict': self.optimizer.state_dict()}
                save_name = "{0:}_{1:}.pt".format(chpt_prefx, it + 1)    
                torch.save(save_dict, save_name)       

    def __infer(self):
        device = torch.device(self.config['testing']['device_name'])
        self.net.to(device)
        # laod network parameters and set the network as evaluation mode
        self.checkpoint = torch.load(self.config['testing']['checkpoint_name'])
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.net.eval()

        name_prediction = []
        with torch.no_grad():
            for data in self.test_loader:
                images = data['image']
                img_name = data['name']
                images = images.to(device)
                outputs = self.net(images).cpu()
                outputs = torch.argmax(outputs, dim = -1).numpy()
                name_prediction.append([img_name[0]] + list(outputs[0]))
        # save predictions as a csv file
        output_name   = self.config['testing']['output_name']
        header = self.test_dataset.get_csv_header()
        with open(output_name, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter = ',', lineterminator='\n',
                quotechar='"',quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header)
            for csv_item in name_prediction:
                csv_writer.writerow(csv_item)


    def run(self):
        agent.__create_dataset()
        agent.__create_network()
        if(self.stage == 'train'):
            self.__train()
        else:
            self.__infer()

if __name__ == "__main__":          
    if(len(sys.argv) < 3):
        print('Number of arguments should be 3. e.g.')
        print('    python train_test.py train config.cfg')
        exit()
    stage    = str(sys.argv[1])
    cfg_file = str(sys.argv[2])
    config   = parse_config(cfg_file)
    agent    = TrainInferAgent(config, stage)
    agent.run()
