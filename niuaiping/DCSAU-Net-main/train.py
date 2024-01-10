import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import time
import albumentations as A
from albumentations.pytorch import ToTensor
from torch.utils.data import random_split
from torch.optim import lr_scheduler
import seaborn as sns
import pandas as pd
import argparse
import os
from data_loading import multi_classes,binary_class
from sklearn.model_selection import GroupKFold
from pytorch_dcsaunet import DCSAU_Net
from loss import *
# from self_attention_cv import transunet


#数据增强
def get_train_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.HorizontalFlip(p=0.25),
        A.ShiftScaleRotate(shift_limit=0,p=0.25),
        A.CoarseDropout(),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])

def get_valid_transform():
   return A.Compose(
       [
        A.Resize(256, 256),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensor()
        ])


def train_model(model, criterion, optimizer, scheduler, num_epochs=5, cuda = None):
    since = time.time()  #记录训练开始的时间
    
    Loss_list = {'train': [], 'valid': []}
    Accuracy_list = {'train': [], 'valid': []}
    
    best_model_wts = model.state_dict()  #保存模型的最佳权重
    best_loss = float('inf')    #初始化最佳损失为正无穷大
    counter = 0     #计数器，用于判断是否出现连续验证集损失上升的情况
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  

            running_loss = []   #初始化用于记录每个批次损失和准确率的列表
            running_corrects = []
        
            # Iterate over data遍历数据
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for inputs,labels,image_id in dataloaders[phase]:      
            # for image_id, (inputs,labels) in dataloaders[phase]:      
                # wrap them in Variable
                if cuda:
                    if torch.cuda.is_available():
                        inputs = Variable(inputs.cuda())
                        labels = Variable(labels.cuda())
                        #label_for_ce = Variable(label_for_ce.cuda())
                    else:
                        inputs, labels = Variable(inputs), Variable(labels)
                else:
                    inputs, labels = Variable(inputs), Variable(labels) #CPU
                # zero the parameter gradients 将模型参数的梯度置零
                optimizer.zero_grad()
                #label_for_ce = label_for_ce.long()
                # forward
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                score = accuracy_metric(outputs,labels) #计算损失和准确率

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # calculate loss and IoU
                running_loss.append(loss.item())
                running_corrects.append(score.item())
             

            epoch_loss = np.mean(running_loss)      #计算当前轮次的平均损失和准确率
            epoch_acc = np.mean(running_corrects)
            
            print('{} Loss: {:.4f} IoU: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)

            # save parameters验证阶段并，则更新最佳损失、保存模型权重，并将计数器重置为0
            if phase == 'valid' and epoch_loss <= best_loss:
                best_loss = epoch_loss
                best_model_wts = model.state_dict()
                counter = 0
                if epoch > 50:
                    torch.save(model, f'save_models/epoch_{epoch}_{epoch_acc}.pth')
            elif phase == 'valid' and epoch_loss > best_loss:
                counter += 1
            if phase == 'train':
                scheduler.step()    #调用学习率调度器更新学习率
        
        print()

    time_elapsed = time.time() - since      #计算训练时间并打印训练完成的信息
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    model.load_state_dict(best_model_wts)
    return model, Loss_list, Accuracy_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='data1/', help='the path of images')
    parser.add_argument('--csvfile', type=str,default='src/test_train_data.csv', help='two columns [image_id,category(train/test)]')
    parser.add_argument('--loss', default='dice', help='loss type')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=150, help='epoches')
    parser.add_argument('--cuda', type=int, default=0, help='is cuda')
    args = parser.parse_args()  #解析命令行参数，并将解析结果存储在args对象中

    os.makedirs(f'save_models/',exist_ok=True)
    

    df = pd.read_csv(args.csvfile)
    df = df[df.category=='train']
    df.reset_index(drop=True, inplace=True) #重置DataFrame的索引，并将更改应用到原始DataFrame中
    gkf  = GroupKFold(n_splits = 5) #创建一个GroupKFold拆分器对象，将数据集分成5个折叠
    df['fold'] = -1  #在DataFrame中添加一列名为fold，并初始化为-1
    #对于每个折叠，使用GroupKFold拆分器根据image_id进行拆分，并获得训练集和验证集的索引。
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups = df.image_id.tolist())):
        df.loc[val_idx, 'fold'] = fold  #将验证集所在行的fold列的值设置为当前折叠的值
    
    fold = 0
    val_files = list(df[df.fold==fold].image_id) #获取验证集文件的标识符列表
    train_files = list(df[df.fold!=fold].image_id)
    
    #创建一个训练集的数据集对象，其中args.dataset是数据集的路径，train_files是训练集文件的标识符列表,get_train_transform()是用于数据增强的函数
    train_dataset = binary_class(args.dataset,train_files, get_train_transform())
    val_dataset = binary_class(args.dataset,val_files, get_valid_transform())
    
    #创建一个训练集的数据加载器，用于批量加载训练数据。batch_size是每个批次的样本数量，shuffle=True表示在每个轮次中对数据进行洗牌，drop_last=True表示如果最后一个批次样本数量不足一个批次，则丢弃该批次。
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch//2,drop_last=True)
    
    dataloaders = {'train':train_loader,'valid':val_loader}
    
    #创建一个名为model_ft的模型对象，img_channels=3表示输入图像的通道数为3，n_classes=1表示输出的类别数为1
    model_ft = DCSAU_Net.Model(img_channels = 3, n_classes = 1)
    if args.cuda:
        if torch.cuda.is_available():
            model_ft = model_ft.cuda()
        
    # Loss, IoU and Optimizer
    if args.loss == 'ce':
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
    if args.loss == 'dice':
        criterion = DiceLoss_binary()
        #criterion = DiceLoss_multiple()
    
    accuracy_metric = IoU_binary()
    #accuracy_metric = IoU_multiple()
    optimizer_ft = optim.Adam(model_ft.parameters(),lr = args.lr) #创建一个Adam优化器，用于优化模型的参数
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.5) #创建一个学习率调度器，使用StepLR策略。在每100个轮次时，将学习率乘以0.5
    #exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, patience=5, factor=0.1,min_lr=1e-6)
    model_ft, Loss_list, Accuracy_list = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=args.epoch, cuda = args.cuda)
    

    torch.save(model_ft, f'save_models/epoch_last.pth')
    
    #绘制验证集损失和IoU曲线：
    plt.title('Validation loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["valid"], 'IoU':Accuracy_list["valid"]})
    valid_data.to_csv(f'valid_data.csv')      #将valid_data保存为CSV文件valid_data.csv
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('valid.png')
    
    plt.figure()
    plt.title('Training loss and IoU',)
    valid_data = pd.DataFrame({'Loss':Loss_list["train"],'IoU':Accuracy_list["train"]})
    valid_data.to_csv(f'train_data.csv')
    sns.lineplot(data=valid_data,dashes=False)
    plt.ylabel('Value')
    plt.xlabel('Epochs')
    plt.savefig('train.png')

