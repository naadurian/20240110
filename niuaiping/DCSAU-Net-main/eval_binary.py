import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from data_loading import binary_class
import albumentations as A
from albumentations.pytorch import ToTensor
from pytorch_lightning.metrics import Accuracy, Precision, Recall, F1
import argparse
import time
import pandas as pd
import cv2
import os
from skimage import io, transform
from PIL import Image


class IoU(nn.Module):
    def __init__(self, weight=None, size_average=True): #是否对IoU进行平均
        super(IoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #将输入张量和目标张量展平为一维向量。这是为了方便后续计算交集、并集和总数
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return IoU

class Dice(nn.Module): 
    def __init__(self, weight=None, size_average=True): #指定是否对Dice系数进行平均
        super(Dice, self).__init__()

#输入张量-预测的结果 目标张量-真实的标签 平滑系数-避免除以零的情况。默认值为1，常用于平滑Dice系数的计算
    def forward(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()   #点乘                         
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice

#图像预处理和数据增强操作
def get_transform():
   return A.Compose( #创建一个操作序列，将多个图像转换操作组合在一起
       [
        A.Resize(256, 256), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),#将图像的每个通道进行归一化处理。mean和std参数指定了每个通道的均值和标准差，用于对图像进行归一化。值通常是在ImageNet数据集上计算出
        ToTensor()  #将图像转换为张量格式。将图像从NumPy数组转换为PyTorch张量，以便在训练模型时使用
        ])
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='data/',type=str, help='the path of dataset')
    parser.add_argument('--csvfile', default='src/test_train_data.csv',type=str, help='two columns [image_id,category(train/test)]')
    parser.add_argument('--model',default='save_models/epoch_last.pth', type=str, help='the path of model')
    parser.add_argument('--debug',default=True, type=bool, help='plot mask')
    args = parser.parse_args()
    
    os.makedirs('debug/',exist_ok=True)
    
    df = pd.read_csv(args.csvfile)
    df = df[df.category=='test']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_files = list(df.image_id)
    test_dataset = binary_class(args.dataset,test_files, get_transform())
    model = torch.load(args.model)

    model = model.cuda()
    
    dice_eval = Dice()
    acc_eval = Accuracy()
    pre_eval = Precision()
    recall_eval = Recall()
    # precision = 
    '''
    TP: 把正样本预测准确的个数
    TN: 把负样本预测准确的个数
    FP: 把正样本预测错误的个数
    FN: 把夫
    '''
    # recall = 
    # accuracy = 
    # f1 = 
    
    f1_eval = F1(2)
    iou_eval = IoU()
    iou_score = []
    acc_score = []
    pre_score = []
    recall_score = []
    f1_score = []
    dice_score = []
    time_cost = []
    
    since = time.time()
    
    for image_id in test_files:
        img = cv2.imread(f'{args.dataset}/images/{image_id}')
        img = cv2.resize(img, ((256,256)))
        img_id = list(image_id.split('.'))[0]
        cv2.imwrite(f'debug/{img_id}.png',img)
    
    with torch.no_grad():  #执行测试集的评估时关闭了梯度计算
        for img, mask, img_id in test_dataset:
            #将输入图像转换为浮点张量，并在维度0上添加一个额外的维度 .unsqueeze(输入, dim=0在第几维添加)
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()           
            mask = Variable(torch.unsqueeze(mask, dim=0).float(), requires_grad=False).cuda()
            torch.cuda.synchronize()  #同步GPU操作
            start = time.time()
            pred = model(img)
            torch.cuda.synchronize()
            end = time.time()
            time_cost.append(end-start)

            pred = torch.sigmoid(pred)

            #根据阈值0.5将预测输出二值化为0和1，得到二值化的预测结果
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            
            #创建预测结果的副本 detach()函数来切断一些分支的反向传播
            pred_draw = pred.clone().detach()
            mask_draw = mask.clone().detach()
            
            
            if args.debug:  #命令行传入debug为True
                img_id = list(img_id.split('.'))[0]
                img_numpy = pred_draw.cpu().detach().numpy()[0][0]  #将预测结果转换为NumPy数组，并选择第一个通道的内容。这假设预测输出是单通道的
                img_numpy[img_numpy==1] = 255 
                cv2.imwrite(f'debug/{img_id}_pred.png',img_numpy)
                
                mask_numpy = mask_draw.cpu().detach().numpy()[0][0]
                mask_numpy[mask_numpy==1] = 255
                cv2.imwrite(f'debug/{img_id}_gt.png',mask_numpy)
            iouscore = iou_eval(pred,mask)
            dicescore = dice_eval(pred,mask)
            pred = pred.view(-1)
            mask = mask.view(-1)
     
            accscore = acc_eval(pred.cpu(),mask.cpu())
            prescore = pre_eval(pred.cpu(),mask.cpu())
            recallscore = recall_eval(pred.cpu(),mask.cpu())
            f1score = f1_eval(pred.cpu(),mask.cpu())
            
            #将计算得到的评估指标值转换为NumPy数组，并添加到相应的列表中
            iou_score.append(iouscore.cpu().detach().numpy()) 
            dice_score.append(dicescore.cpu().detach().numpy())
            acc_score.append(accscore.cpu().detach().numpy())
            pre_score.append(prescore.cpu().detach().numpy())
            recall_score.append(recallscore.cpu().detach().numpy())
            f1_score.append(f1score.cpu().detach().numpy())
            torch.cuda.empty_cache()  #清空GPU缓存，释放GPU上的内存空间
            
    time_elapsed = time.time() - since
    
    print('Evaluation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('FPS: {:.2f}'.format(1.0/(sum(time_cost)/len(time_cost))))
    print('mean IoU:',round(np.mean(iou_score),4),round(np.std(iou_score),4))  #四舍五入到四位小数
    print('mean accuracy:',round(np.mean(acc_score),4),round(np.std(acc_score),4))
    print('mean precsion:',round(np.mean(pre_score),4),round(np.std(pre_score),4))
    print('mean recall:',round(np.mean(recall_score),4),round(np.std(recall_score),4))
    print('mean F1-score:',round(np.mean(f1_score),4),round(np.std(f1_score),4))
