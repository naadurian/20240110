
import torch.nn as nn
import torch

class DiceLoss_binary(nn.Module):
    def __init__(self) -> None:
        super(DiceLoss_binary,self).__init__()
    
    def forward(self,inputs,targets,smooth=1):
        inputs = torch.sigmoid(inputs)
                
        intersection = (inputs*targets).sum()
        union = targets.sum()+inputs.sum()
        dice = 2.*intersection+smooth/union+smooth
        
        return 1-dice
    
class Diceloss_multiple(nn.Module):
    def __init__(self) -> None:
        super(Diceloss_multiple,self).__init__()
        self.sfm = nn.Softmax(dim=1)
        
    def Diceloss_binary(self,inputs,targets,smooth=1):
        intersection = (inputs * targets).sum()
        toal = inputs.sum()+targets.sum()
        diceloss_binary = (2.*intersection+smooth)/(toal+smooth)
        
        return 1- diceloss_binary
    def forward(self,input,target):
        input = self.sfm(input)     #input.shape=(batch_size,类别数)
        c = input.shape[1]
        sum_loss=0
        for i in range(c):
            ipt = input[:,i]
            tgt = target[:,i]
            diceloss = self.DiceLoss_binary(ipt,tgt)
            sum_loss = diceloss +sum_loss
        
        return sum_loss/c