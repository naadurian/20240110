import torch.nn as nn
import torch
from pytorch_lightning.metrics import F1
import torch.nn.functional as F
from pytorch_lightning.metrics import ConfusionMatrix
import numpy as np

cfs = ConfusionMatrix(3)

class DiceLoss_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1) #softmax 函数通常在分类问题中使用，可以将一个多分类问题转换成多个二分类问题，从而得到每个类别的概率分布
    
    def binary_dice(self, inputs, targets, smooth=1):
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        
        return 1-dice

    def forward(self, ipts, gt):
        
        ipts = self.sfx(ipts)
        c = ipts.shape[1]
        sum_loss = 0
        for i in range(c):
            tmp_inputs = ipts[:,i]
            tmp_gt = gt[:,i]
            tmp_loss = self.binary_dice(tmp_inputs,tmp_gt)
            sum_loss += tmp_loss
        return sum_loss / c
    
class IoU_multiple(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_multiple, self).__init__()
        self.sfx = nn.Softmax(dim=1)

    def forward(self, inputs, targets, smooth=1):
        inputs = self.sfx(inputs)
        c = inputs.shape[1]
        inputs = torch.max(inputs,1).indices.cpu()
        targets = torch.max(targets,1).indices.cpu()
        cfsmat = cfs(inputs,targets).numpy()
        
        sum_iou = 0
        for i in range(c):
                tp = cfsmat[i,i]
                fp = np.sum(cfsmat[0:3,i]) - tp
                fn = np.sum(cfsmat[i,0:3]) - tp
            
                tmp_iou = tp / (fp + fn + tp)
                sum_iou += tmp_iou
                
        return sum_iou / c

#计算模型预测结果与目标值之间的相似度损失
class DiceLoss_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss_binary, self).__init__()
    #inputs表示模型的预测结果，而targets表示目标值
    def forward(self, inputs, targets, smooth=1):
        
        inputs = torch.sigmoid(inputs) #new_b = b.unsqueeze(-1)
        
        #计算了inputs和targets的交集，即两者相乘并求和，结果存储在变量intersection中，这个交集表示模型预测正确的像素点的数量
        intersection = (inputs * targets).sum()   #计算Dice系数，该系数用于评估预测结果和目标值之间的相似度                         
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1-dice

class IoU_binary(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoU_binary, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)
                     
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth) / (union + smooth)
                
        return IoU
