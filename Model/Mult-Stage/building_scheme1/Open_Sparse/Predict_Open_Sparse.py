#!/usr/bin/env python
# coding: utf-8
import os
import time
import h5py
import PIL.Image as image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn.functional as F
import torch.nn.init as init
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch import nn, optim
from torchvision import transforms
from torch.autograd import Variable
from torch.utils.data import Dataset,TensorDataset, DataLoader
from torch.utils.data import SubsetRandomSampler
from tensorboardX import SummaryWriter


#writer = SummaryWriter('tensorflow')

#device = torch.device("cuda:1")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

##################模型############################
#

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1
#类中expansion =1，其表示box_block中最后一个block的channel比上第一个block的channel
    def __init__(self, in_planes, planes, stride=1, option='B'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        self.option=option
        # def tmp_func(x):
        #     return F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes // 4), "constant", 0)

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = nn.ConstantPad3d((0, 0, 0, 0, planes // 4, planes // 4),0)

            if option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.option=='A':
            out += self.shortcut(x[:,:,::2,::2])
        if self.option=='B':
            out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=2):
        self.in_planes = 64  #残差块输入层数
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.linear = nn.Linear(256, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])

#############################################测 试 集 增 强##################################################

class TTA():

    def __init__(self,net):

        self.net=net

    def data_enhance(self,image1):

        data10 = image1.copy()
        data11 = image1[:,::-1,:,:].copy()
        data12 = image1[:,:,::-1,:].copy()
        data13 = np.rot90(image1, 1, (1, 2)).copy()
        data14 = np.rot90(image1, 2, (1, 2)).copy()
        data15 = np.rot90(image1, 3, (1, 2)).copy()
        p1 = int(np.random.rand(1)[0]*2+1)
        p2 = int(np.random.rand(1)[0] * 2 + 1)
        data16 = np.concatenate((image1[:,:-p1,:,:].copy(),image1[:,-p1:,:,:].copy()),axis=1)
        data17 = np.concatenate((image1[:,:,-p2:,:].copy(),image1[:,:,:-p2,:].copy()),axis=2)
        data18 = image1.transpose(0,2,1,3)
        data19 = np.rot90(image1, 1, (1, 2)).transpose(0, 2, 1, 3).copy()

        data10 = Variable(torch.Tensor(data10.transpose(0, 3, 1, 2)).float()).cuda()
        data11 = Variable(torch.Tensor(data11.transpose(0, 3, 1, 2)).float()).cuda()
        data12 = Variable(torch.Tensor(data12.transpose(0, 3, 1, 2)).float()).cuda()
        data13 = Variable(torch.Tensor(data13.transpose(0, 3, 1, 2)).float()).cuda()
        data14 = Variable(torch.Tensor(data14.transpose(0, 3, 1, 2)).float()).cuda()
        data15 = Variable(torch.Tensor(data15.transpose(0, 3, 1, 2)).float()).cuda()
        data16 = Variable(torch.Tensor(data16.transpose(0, 3, 1, 2)).float()).cuda()
        data17 = Variable(torch.Tensor(data17.transpose(0, 3, 1, 2)).float()).cuda()
        data18 = Variable(torch.Tensor(data18.transpose(0, 3, 1, 2)).float()).cuda()
        data19 = Variable(torch.Tensor(data19.transpose(0, 3, 1, 2)).float()).cuda()

        _, output0 = torch.max(self.net.forward(data10).data, 1)
        _, output1 = torch.max(self.net.forward(data11).data, 1)
        _, output2 = torch.max(self.net.forward(data12).data, 1)
        _, output3 = torch.max(self.net.forward(data13).data, 1)
        _, output4 = torch.max(self.net.forward(data14).data, 1)
        _, output5 = torch.max(self.net.forward(data15).data, 1)
        _, output6 = torch.max(self.net.forward(data16).data, 1)
        _, output7 = torch.max(self.net.forward(data17).data, 1)
        _, output8 = torch.max(self.net.forward(data18).data, 1)
        _, output9 = torch.max(self.net.forward(data19).data, 1)

        p0 = output0.cpu().numpy().tolist()
        p1 = output1.cpu().numpy().tolist()
        p2 = output2.cpu().numpy().tolist()
        p3 = output3.cpu().numpy().tolist()
        p4 = output4.cpu().numpy().tolist()
        p5 = output5.cpu().numpy().tolist()
        p6 = output6.cpu().numpy().tolist()
        p7 = output7.cpu().numpy().tolist()
        p8 = output8.cpu().numpy().tolist()
        p9 = output9.cpu().numpy().tolist()

        p_array=np.array(p0+p1+p2+p3+p4+p5+p6+p7+p8+p9)
        result=np.argmax(np.bincount(p_array))

        return result

class TTA_soft():

    def __init__(self,net):

        self.net=net

    def data_enhance(self,image1):

        data10 = image1.copy()
        data11 = image1[:,::-1,:,:].copy()
        data12 = image1[:,:,::-1,:].copy()
        data13 = np.rot90(image1, 1, (1, 2)).copy()
        data14 = np.rot90(image1, 2, (1, 2)).copy()
        data15 = np.rot90(image1, 3, (1, 2)).copy()
        p1 = int(np.random.rand(1)[0]*2+1)
        p2 = int(np.random.rand(1)[0] * 2 + 1)
        data16 = np.concatenate((image1[:,:-p1,:,:].copy(),image1[:,-p1:,:,:].copy()),axis=1)
        data17 = np.concatenate((image1[:,:,-p2:,:].copy(),image1[:,:,:-p2,:].copy()),axis=2)
        data18 = image1.transpose(0,2,1,3)
        data19 = np.rot90(image1, 1, (1, 2)).transpose(0, 2, 1, 3).copy()

        data10 = Variable(torch.Tensor(data10.transpose(0, 3, 1, 2)).float()).cuda()
        data11 = Variable(torch.Tensor(data11.transpose(0, 3, 1, 2)).float()).cuda()
        data12 = Variable(torch.Tensor(data12.transpose(0, 3, 1, 2)).float()).cuda()
        data13 = Variable(torch.Tensor(data13.transpose(0, 3, 1, 2)).float()).cuda()
        data14 = Variable(torch.Tensor(data14.transpose(0, 3, 1, 2)).float()).cuda()
        data15 = Variable(torch.Tensor(data15.transpose(0, 3, 1, 2)).float()).cuda()
        data16 = Variable(torch.Tensor(data16.transpose(0, 3, 1, 2)).float()).cuda()
        data17 = Variable(torch.Tensor(data17.transpose(0, 3, 1, 2)).float()).cuda()
        data18 = Variable(torch.Tensor(data18.transpose(0, 3, 1, 2)).float()).cuda()
        data19 = Variable(torch.Tensor(data19.transpose(0, 3, 1, 2)).float()).cuda()

        output0 = self.net.forward(data10).data
        output1 = self.net.forward(data11).data
        output2 = self.net.forward(data12).data
        output3 = self.net.forward(data13).data
        output4 = self.net.forward(data14).data
        output5 = self.net.forward(data15).data
        output6 = self.net.forward(data16).data
        output7 = self.net.forward(data17).data
        output8 = self.net.forward(data18).data
        output9 = self.net.forward(data19).data

        p0 = output0.cpu().numpy()
        p1 = output1.cpu().numpy()
        p2 = output2.cpu().numpy()
        p3 = output3.cpu().numpy()
        p4 = output4.cpu().numpy()
        p5 = output5.cpu().numpy()
        p6 = output6.cpu().numpy()
        p7 = output7.cpu().numpy()
        p8 = output8.cpu().numpy()
        p9 = output9.cpu().numpy()

        p_array=np.array(p0+p1+p2+p3+p4+p5+p6+p7+p8+p9)
        result=np.argmax(p_array)

        return result

# ############################################预 测#####################################################

#preda_s1_new = np.load("/data/DW/Challenge/GermanAIChallenge2018/Round_2/R2tesa_s1_new.npy")
preda_s2_new = np.load("/data/DW/Challenge/GermanAIChallenge2018/Round_2/R2tesa_s2_new.npy")

#######################预测集a_S1##################

#print('预测集s1大小',preda_s1_new.shape)

#######################预测集a_S2##################

print('预测集s2大小',preda_s2_new.shape)

#######################整理########################
#(4838,32,32,14)
#preda_new=np.concatenate((preda_s1_new,preda_s2_new),axis=3)
preda_new=preda_s2_new
#del preda_s1,preda_s2
#del preda_s1_new,preda_s2_new

print('预测集大小：',preda_new.shape)

#######################预测集 Loader#################

model=torch.load("/data/DW/Challenge/GermanAIChallenge2018/Round_2/SmallClass/building_scheme1/Open_Sparse/ResNet50_optionB_building_open_sparse_nos1_fold1_40.pkl",map_location='cpu')

model=model.cuda()

pred_y = []

model.eval()

operator=TTA_soft(model)

for i in range(preda_new.shape[0]):

    temp = operator.data_enhance(preda_new[[i],:,:,:])
    pred_y+=[temp]

# def write_csv(results,file_name):
#     import csv
#     with open(file_name,'w') as f:
#         writer = csv.writer(f)
#         #writer.writerow(['id','label'])
#         writer.writerows(results)#注意读写
#
# #from keras.utils import to_categorical
# #result = to_categorical(np.array(pred_y))
#
#
# label=torch.from_numpy(np.array(pred_y)[:,np.newaxis]).long()
# print(label.shape)
# result=torch.zeros(len(pred_y), 17).scatter_(1, label, 1)
# write_csv(result.numpy().astype(int),'R2_preda_single_optionB_building_open_sparse_Res50_epoch45_fold5_0129.csv')
np.save('R2_preda_single_optionB_building_open_sparse_Res50_epoch40_fold1_0210_soft_nos1.npy',np.array(pred_y))
