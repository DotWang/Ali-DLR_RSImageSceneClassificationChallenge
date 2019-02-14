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

#import Accuracy
#
writer = SummaryWriter('tensorflow/resnet50_double_fold5')

#device = torch.device("cuda:1")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

####################标签 & 索引##########################


fold_idx=np.load('fold_5_indexes.npy')

trn_idx=fold_idx[4][0]
val_idx=fold_idx[4][1]

del fold_idx

trn_val_y=np.load('trn_val_y.npy')

trn_y=trn_val_y[trn_idx]
val_y=trn_val_y[val_idx]

del trn_val_y

################产生训练集，验证集##################

all_s1=np.load('R2trn_s1_new.npy')

trn_s1=all_s1[trn_idx]
val_s1=all_s1[val_idx]

del all_s1

all_s2=np.load('R2trn_s2_new.npy')

trn_s2=all_s2[trn_idx]
val_s2=all_s2[val_idx]

del all_s2


print(np.max(trn_y),np.min(trn_y))#最大16，最小0，没问题

############################################训  练#####################################################

#######################训练集S1##################

print('训练集s1大小,',trn_s1.shape)

#######################训练集S2##################


print('训练集s2大小,',trn_s2.shape)

#######################验证集S1##################


print('验证集s1大小,',val_s1.shape)

#######################验证集S2##################

print('验证集s2大小,',val_s2.shape)

##################整理##########################


# train_new=np.concatenate((trn_s1,trn_s2),axis=3)
# 
# del trn_s1,trn_s2
# 
# valid_new=np.concatenate((val_s1,val_s2),axis=3)
# 
# del val_s1,val_s2

#print('训练集大小：',train_new.shape)


class MyDataset(TensorDataset):

    def __init__(self, train=1, data_X1=trn_s1,data_X2=trn_s2, data_y=trn_y):

        self.train = train
        self.x1 = data_X1
        self.x2 = data_X2
        self.y = data_y

    def __getitem__(self, index):

        if self.train:
            p = np.random.rand(1)[0]

            if p < 0.1:
                data1 = np.fliplr(self.x1[index, :, :, :]).copy()
            if p >= 0.1 and p < 0.2:
                data1 = np.flipud(self.x1[index, :, :, :]).copy()
            if p >= 0.2 and p < 0.25:
                data1 = np.rot90(self.x1[index, :, :, :], 1, (0, 1)).copy()
            if p >= 0.25 and p < 0.3:
                data1 = np.rot90(self.x1[index, :, :, :], 2, (0, 1)).copy()
            if p >= 0.3 and p < 0.35:
                data1 = np.rot90(self.x1[index, :, :, :], 3, (0, 1)).copy()
            if p >= 0.35 and p < 0.4:
                data1 = self.x1[index, :, :, :].transpose(1, 0, 2).copy()
            if p >= 0.4 and p < 0.45:
                data1 = np.rot90(self.x1[index, :, :, :], 1, (0, 1)).transpose(1, 0, 2).copy()
            if p >= 0.45:
                data1 = self.x1[index, :, :, :]

            if p < 0.1:
                data2 = np.fliplr(self.x2[index, :, :, :]).copy()
            if p >= 0.1 and p < 0.2:
                data2 = np.flipud(self.x2[index, :, :, :]).copy()
            if p >= 0.2 and p < 0.25:
                data2 = np.rot90(self.x2[index, :, :, :], 1, (0, 1)).copy()
            if p >= 0.25 and p < 0.3:
                data2 = np.rot90(self.x2[index, :, :, :], 2, (0, 1)).copy()
            if p >= 0.3 and p < 0.35:
                data2 = np.rot90(self.x2[index, :, :, :], 3, (0, 1)).copy()
            if p >= 0.35 and p < 0.4:
                data2 = self.x2[index, :, :, :].transpose(1, 0, 2).copy()
            if p >= 0.4 and p < 0.45:
                data2 = np.rot90(self.x2[index, :, :, :], 1, (0, 1)).transpose(1, 0, 2).copy()
            if p >= 0.45:
                data2 = self.x2[index, :, :, :]

            #p1 = int(np.random.rand(1)[0] * 2 + 1)
            #p2 = int(np.random.rand(1)[0] * 2 + 1)
            #data7 = np.concatenate((self.x[index,:,:,:][:-p1,:,:].copy(),self.x[index,:,:,:][-p1:,:,:].copy()),axis=0)
            #data8 = np.concatenate((self.x[index,:,:,:][:, -p2:, :].copy(), self.x[index,:,:,:][:, :-p2, :].copy()), axis=1)

            #data=np.concatenate((data1,data2,data3,data4,data5,data6,data7,data8),axis=0)

        else:
            data1 = self.x1[index, :, :, :]
            data2 = self.x2[index, :, :, :]
        #print(data.shape)
        data1 = torch.from_numpy(data1.transpose(2, 0, 1))
        data2 = torch.from_numpy(data2.transpose(2, 0, 1))
        #data = torch.from_numpy(data.transpose(0,3,1,2))
        labels = self.y[index]
        return data1, data2, labels

    def __len__(self):

        return len(self.y)


train_datasets=MyDataset(train=1, data_X1=trn_s1,data_X2=trn_s2, data_y=trn_y)
valid_datasets=MyDataset(train=0, data_X1=val_s1,data_X2=val_s2, data_y=val_y)


train_loader = DataLoader(dataset=train_datasets, batch_size=768, sampler=SubsetRandomSampler(range(trn_s1.shape[0])),num_workers=3)
valid_loader = DataLoader(dataset=valid_datasets, batch_size=768, num_workers=3)

del trn_s1,trn_s2,val_s1,val_s2
del train_datasets,valid_datasets

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

    def __init__(self, block, layers, num_classes=17):
        self.in_planes1 = 64  #残差块输入层数
        self.in_planes2 = 64  # 残差块输入层数
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(5, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer1(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer1(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer1(block, 256, layers[2], stride=2)

        self.layer4 = self._make_layer2(block, 64, layers[0], stride=1)
        self.layer5 = self._make_layer2(block, 128, layers[1], stride=2)
        self.layer6 = self._make_layer2(block, 256, layers[2], stride=2)
        #self.linear = nn.Linear(256, num_classes)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.drop= nn.Dropout(p=0.1,inplace=True)

        self.apply(_weights_init)

    def _make_layer1(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers1 = []
        for stride in strides:
            layers1.append(block(self.in_planes1, planes, stride))
            self.in_planes1 = planes * block.expansion

        return nn.Sequential(*layers1)

    def _make_layer2(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers2 = []
        for stride in strides:
            layers2.append(block(self.in_planes2, planes, stride))
            self.in_planes2 = planes * block.expansion

        return nn.Sequential(*layers2)

    def forward(self, x1, x2):
        #x1
        out1 = F.relu(self.bn1(self.conv1(x1)))
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = F.avg_pool2d(out1, out1.size()[3])
        out1 = out1.view(out1.size(0), -1)

        #x2
        out2 = F.relu(self.bn2(self.conv2(x2)))
        out2 = self.layer4(out2)
        out2 = self.layer5(out2)
        out2 = self.layer6(out2)
        out2 = F.avg_pool2d(out2, out2.size()[3])
        out2 = out2.view(out2.size(0), -1)

        out=torch.cat((out1,out2),1)
        out = self.drop(out)
        out = F.relu(self.fc1(out))
        out = self.drop(out)
        out = self.fc2(out)


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


# base_model=torchvision.models.resnet50(pretrained=True)
#model=Mymodel(base_model)

#model=MyResNet(Bottleneck, [3, 4, 6, 3], num_classes=17)

# W=torch.from_numpy(np.array([124.03,25.87,8.92, 25.59,27.96,
#                               13.86,30.61,7.69,13.51, 19.35,
#                               11.77,41.7 ,20.85,10.89, 100.77,23.71,9.52])).float().cuda()

# model=resnet56()
model=torch.load("/data/DW/Challenge/GermanAIChallenge2018/Round_2/doublechannel/ResNet50_optionB_nogau_dehaze_doublechannel_fold5_50.pkl",map_location='cpu')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-5, weight_decay=1e-3)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8)

if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model,device_ids=[0])

model = model.cuda()
#
############################训练/验证#########################

def train(epoch,batch_idx_trn):
    print("Epoch [{}/10]".format(epoch))
    model.train()
    start = time.time()
    train_loss = 0.
    correct=0
    total =0
    for batch_idx,(image1,image2,labels) in enumerate(train_loader):

        image1 = image1.type(torch.FloatTensor).cuda()
        image2 = image2.type(torch.FloatTensor).cuda()
        labels = labels.type(torch.LongTensor).cuda()

        optimizer.zero_grad()
        outputs = model(image1,image2)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss+=loss.item()
        _, predicted = torch.max(outputs.data, 1)
        # val_loader total
        total += labels.size(0)
        # add correct
        correct += (predicted == labels).sum().item()

        del image1,image2,labels
        del outputs

    # if batch_idx % 10 ==0:
    #writer.add_scalar('Train', loss, batch_idx)
        writer.add_scalar('Train/Loss', loss.item(), batch_idx_trn)
        batch_idx_trn += 1
    del loss
    end = time.time()
    print("Loss: {:.8f}, Acc: {:.4f}%, correct/total:{}/{},"
          "Time: {:.1f}sec!".format(train_loss/(batch_idx+1),
                                                (100*correct / total),correct,total,
                                                end-start))
    return batch_idx_trn

def valid(epoch,batch_idx_val):
    model.eval()
    valid_loss = 0.
    correct = 0
    total = 0
    pred_val=[]
    with torch.no_grad():
        for batch_idx,(image1,image2, labels) in enumerate(valid_loader):

            image1 = image1.type(torch.FloatTensor).cuda()
            image2 = image2.type(torch.FloatTensor).cuda()
            labels = labels.type(torch.LongTensor).cuda()

            outputs = model(image1,image2)
            loss = criterion(outputs, labels)

            valid_loss+=loss.item()
            _, predicted = torch.max(outputs.data, 1)

            p = predicted.cpu().numpy().tolist()
            pred_val += p

            total += labels.size(0)
            # add correct
            correct += (predicted == labels).sum().item()

            del image1,image2,labels
            del outputs
        #scheduler.step(loss)#learning decay
            writer.add_scalar('Valid/Loss', loss.item(), batch_idx_val)
            batch_idx_val += 1
        del loss
        print("Loss: {:.8f}, Acc: {:.4f}%,"
              "correct/total:{}/{}".format(valid_loss/(batch_idx+1),100*correct / total,
                                                       correct,total))
    pred_val=np.array(pred_val)

    return batch_idx_val,pred_val

batch_idx_trn=1
batch_idx_val=1
for epoch in range(51,101):

    batch_idx_trn=train(epoch,batch_idx_trn)
    batch_idx_val,pred_val=valid(epoch,batch_idx_val)

    if epoch%5==0:
        path = "/data/DW/Challenge/GermanAIChallenge2018/Round_2/doublechannel/ResNet50_optionB_nogau_dehaze_doublechannel_fold5_"+ str(epoch)+".pkl"
        torch.save(model, path)
        print("Model save to /data/DW/Challenge/GermanAIChallenge2018/scene{}.pkl".format(epoch))

writer.close()

############################################验证集精度评价#########################################
# OA
print('Valid Dataset Overall Accuracy:',np.mean(val_y==pred_val))
#
# 分类报告
print('Valid Dataset Report')
print(classification_report(val_y,pred_val))

#############################################测 试 集 增 强##################################################
#
# class TTA():
#
#     def __init__(self,net):
#
#         self.net=net
#
#     def data_enhance(self,image1,image2):
#
#         data10 = image1
#         data11 = image1[:, ::-1, :, :].copy()
#         data12 = image1[:, :, ::-1, :].copy()
#         data13 = np.rot90(image1, 1, (1, 2)).copy()
#         data14 = np.rot90(image1, 2, (1, 2)).copy()
#         data15 = np.rot90(image1, 3, (1, 2)).copy()
#         p1 = int(np.random.rand(1)[0] * 2 + 1)
#         p2 = int(np.random.rand(1)[0] * 2 + 1)
#         data16 = np.concatenate((image1[:, :-p1, :, :].copy(), image1[:, -p1:, :, :].copy()), axis=1)
#         data17 = np.concatenate((image1[:, :, -p2:, :].copy(), image1[:, :, :-p2, :].copy()), axis=2)
#         data18 = image1.transpose(0, 2, 1, 3)
#         data19 = np.rot90(image1, 1, (1, 2)).transpose(0, 2, 1, 3).copy()
#
#         data20 = image2
#         data21 = image2[:, ::-1, :, :].copy()
#         data22 = image2[:, :, ::-1, :].copy()
#         data23 = np.rot90(image2, 1, (1, 2)).copy()
#         data24 = np.rot90(image2, 2, (1, 2)).copy()
#         data25 = np.rot90(image2, 3, (1, 2)).copy()
#         data26 = np.concatenate((image2[:, :-p1, :, :].copy(), image2[:, -p1:, :, :].copy()), axis=1)
#         data27 = np.concatenate((image2[:, :, -p2:, :].copy(), image2[:, :, :-p2, :].copy()), axis=2)
#         data28 = image2.transpose(0, 2, 1, 3)
#         data29 = np.rot90(image2, 1, (1, 2)).transpose(0, 2, 1, 3).copy()
#
#         data10 = Variable(torch.Tensor(data10.transpose(0, 3, 1, 2)).float()).cuda()
#         data11 = Variable(torch.Tensor(data11.transpose(0, 3, 1, 2)).float()).cuda()
#         data12 = Variable(torch.Tensor(data12.transpose(0, 3, 1, 2)).float()).cuda()
#         data13 = Variable(torch.Tensor(data13.transpose(0, 3, 1, 2)).float()).cuda()
#         data14 = Variable(torch.Tensor(data14.transpose(0, 3, 1, 2)).float()).cuda()
#         data15 = Variable(torch.Tensor(data15.transpose(0, 3, 1, 2)).float()).cuda()
#         data16 = Variable(torch.Tensor(data16.transpose(0, 3, 1, 2)).float()).cuda()
#         data17 = Variable(torch.Tensor(data17.transpose(0, 3, 1, 2)).float()).cuda()
#         data18 = Variable(torch.Tensor(data18.transpose(0, 3, 1, 2)).float()).cuda()
#         data19 = Variable(torch.Tensor(data19.transpose(0, 3, 1, 2)).float()).cuda()
#
#         data20 = Variable(torch.Tensor(data20.transpose(0, 3, 1, 2)).float()).cuda()
#         data21 = Variable(torch.Tensor(data21.transpose(0, 3, 1, 2)).float()).cuda()
#         data22 = Variable(torch.Tensor(data22.transpose(0, 3, 1, 2)).float()).cuda()
#         data23 = Variable(torch.Tensor(data23.transpose(0, 3, 1, 2)).float()).cuda()
#         data24 = Variable(torch.Tensor(data24.transpose(0, 3, 1, 2)).float()).cuda()
#         data25 = Variable(torch.Tensor(data25.transpose(0, 3, 1, 2)).float()).cuda()
#         data26 = Variable(torch.Tensor(data26.transpose(0, 3, 1, 2)).float()).cuda()
#         data27 = Variable(torch.Tensor(data27.transpose(0, 3, 1, 2)).float()).cuda()
#         data28 = Variable(torch.Tensor(data28.transpose(0, 3, 1, 2)).float()).cuda()
#         data29 = Variable(torch.Tensor(data29.transpose(0, 3, 1, 2)).float()).cuda()
#
#         _, output0 = torch.max(self.net.forward(data10, data20).data, 1)
#         _, output1 = torch.max(self.net.forward(data11, data21).data, 1)
#         _, output2 = torch.max(self.net.forward(data12, data22).data, 1)
#         _, output3 = torch.max(self.net.forward(data13, data23).data, 1)
#         _, output4 = torch.max(self.net.forward(data14, data24).data, 1)
#         _, output5 = torch.max(self.net.forward(data15, data25).data, 1)
#         _, output6 = torch.max(self.net.forward(data16, data26).data, 1)
#         _, output7 = torch.max(self.net.forward(data17, data27).data, 1)
#         _, output8 = torch.max(self.net.forward(data18, data28).data, 1)
#         _, output9 = torch.max(self.net.forward(data19, data29).data, 1)
#
#         p0 = output0.cpu().numpy().tolist()
#         p1 = output1.cpu().numpy().tolist()
#         p2 = output2.cpu().numpy().tolist()
#         p3 = output3.cpu().numpy().tolist()
#         p4 = output4.cpu().numpy().tolist()
#         p5 = output5.cpu().numpy().tolist()
#         p6 = output6.cpu().numpy().tolist()
#         p7 = output7.cpu().numpy().tolist()
#         p8 = output8.cpu().numpy().tolist()
#         p9 = output9.cpu().numpy().tolist()
#
#         p_array = np.array(p0 + p1 + p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9)
#         result = np.argmax(np.bincount(p_array))
#
#         return result
#
# # ############################################预 测#####################################################
#
# preda_s1_new = np.load('R2tesa_s1_new.npy')
# preda_s2_new = np.load('R2tesa_s2_new.npy')
#
# #######################预测集a_S1##################
#
# print('预测集s1大小',preda_s1_new.shape)
#
# #######################预测集a_S2##################
#
# print('预测集s2大小',preda_s2_new.shape)
#
# #######################整理########################
# #(4838,32,32,14)
# #preda_new=np.concatenate((preda_s1_new,preda_s2_new),axis=3)
#
# # del preda_s1,preda_s2
# #del preda_s1_new,preda_s2_new
#
# #print('预测集大小：',preda_new.shape)
#
# #######################预测集 Loader#################
#
# model.load_state_dict(torch.load("/data/DW/Challenge/GermanAIChallenge2018/Round_2/tooyoung/ResNet50_30.pkl",map_location='cpu'))
#
# model=model.cuda()
#
# pred_y = []
#
# model.eval()
#
# operator=TTA(model)
#
# for i in range(preda_s1_new.shape[0]):
#
#     temp = operator.data_enhance(preda_s1_new[[i],:,:,:],preda_s2_new[[i],:,:,:])
#     pred_y+=[temp]
#
#
# def write_csv(results,file_name):
#     import csv
#     with open(file_name,'w') as f:
#         writer = csv.writer(f)
#         #writer.writerow(['id','label'])
#         writer.writerows(results) #注意读写
#
# #from keras.utils import to_categorical
# #result = to_categorical(np.array(pred_y))
#
#
# label=torch.from_numpy(np.array(pred_y)[:,np.newaxis]).long()
# print(label.shape)
# result=torch.zeros(len(pred_y), 17).scatter_(1, label, 1)
# write_csv(result.numpy().astype(int),'R2_preda_doublechannel_Res50_30_fold1_0124.csv')
