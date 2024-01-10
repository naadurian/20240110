import argparse
import os
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from sklearn.model_selection import GroupKFold
import argparse
from dataloading import binary_class
from loss import DiceLoss_binary
import albumentations as A 
from albumentations.pytorch import ToTensor
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import model
import time
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
"""
def train_model(model, optimizer,  num_epochs=5, cuda = None):
    since = time.time()  #记录训练开始的时间
    best_model_wts = model.state_dict()  #保存模型的最佳权重
    counter = 0     #计数器，用于判断是否出现连续验证集损失上升的情况
    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)  

           
        
            for inputs,labels in dataloaders[phase]:      
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
                        # inputs, labels = Variable(inputs), Variable(labels) #CPU
                        pass
                    # zero the parameter gradients 将模型参数的梯度置零
                    optimizer.zero_grad()
                    #label_for_ce = label_for_ce.long()
                    # forward
                    # print("inputs:",inputs.shape)
                    outputs = model(inputs)
                    print("outputs:",outputs.shape)
                    optimizer.step()
        print()        
    model.load_state_dict(best_model_wts)
    return model"""
def train_model(model, criterion,optimizer,scheduler,num_epochs=5, cuda = None):
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

            running_loss = []
            running_accurate = []
            
            # Iterate over data遍历数据
            #for inputs,labels,label_for_ce,image_id in dataloaders[phase]: 
            for inputs,labels in dataloaders[phase]:   
                # print(inputs.shape)
                # print(labels)
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
                # print("outputs:",outputs.shape)
                # loss

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                
                running_loss.append(loss.item()) 
                running_accurate.append(score.item())
                
            epoch_loss = np.mean(running_loss)
            epoch_acc = np.mean(running_accurate)   
            # save parameters验证阶段并，则更新最佳损失、保存模型权重，并将计数器重置为0
            
            print('{},loss:{:.4f},Iou:{:.4f}'.format(phase,epoch_loss,epoch_acc))
            Loss_list[phase].append(epoch_loss)
            Accuracy_list[phase].append(epoch_acc)
            
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
    model.load_state_dict(best_model_wts)
    return model, Loss_list, Accuracy_list
   
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='data1/', help='the path of images')
    parser.add_argument('--csvfile', type=str,default='src/test_train_data.csv', help='two columns [image_id,category(train/test)]')
    parser.add_argument('--loss', default='dice', help='loss type')
    parser.add_argument('--batch', type=int, default=16, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epoch', type=int, default=150, help='epoches')
    parser.add_argument('--cuda', type=int, default=0, help='is cuda')
    args = parser.parse_args()  #解析命令行参数，并将解析结果存储在args对象中
    
    os.makedirs(f'save_model',exist_ok=True)
    df = pd.read_csv(args.csvfile)
    df.reset_index(drop=True, inplace=True)
    gkf  = GroupKFold(n_splits = 5)
    df.reset_index(drop=True, inplace=True) #重置DataFrame的索引，并将更改应用到原始DataFrame中
    gkf  = GroupKFold(n_splits = 5) #创建一个GroupKFold拆分器对象，将数据集分成5个折叠
    df['fold'] = -1  #在DataFrame中添加一列名为fold，并初始化为-1
    #对于每个折叠，使用GroupKFold拆分器根据image_id进行拆分，并获得训练集和验证集的索引。
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups = df.image_id.tolist())):
        df.loc[val_idx, 'fold'] = fold  #将验证集所在行的fold列的值设置为当前折叠的值
    fold = 0
    val_files = list(df[df.fold==fold].image_id) #获取验证集文件的标识符列表
    train_files = list(df[df.fold!=fold].image_id)
    val_dataset = binary_class(args.dataset,val_files, get_valid_transform())
    
        
    train_dataset = binary_class(args.dataset,train_files, get_train_transform())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch, shuffle=True,drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch//2,drop_last=True)
    dataloaders = {'train':train_loader,'valid':val_loader}
    
    model_ft = model.model(in_channels = 3, out_channels = 1)
    if args.cuda:
        if torch.cuda.is_available():
            model_ft = model_ft.cuda()
    
    if args.loss == 'ce':
        #criterion = nn.CrossEntropyLoss()
        criterion = nn.BCELoss()
    if args.loss == 'dice':
        criterion = DiceLoss_binary()
        
    accuracy_metric = IoU_binary()
    optimizer_ft = optim.Adam(model_ft.parameters(),lr = args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.5)
    
    model_ft,Loss_list, Accuracy_list = train_model(model_ft, criterion,optimizer_ft,exp_lr_scheduler,num_epochs=args.epoch, cuda = args.cuda)        
    torch.save(model_ft, f'save_model_new/epoch_last.pth')
            