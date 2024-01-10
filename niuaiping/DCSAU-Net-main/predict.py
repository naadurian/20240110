import torch
import torch.nn as nn
import numpy as np
import os
import cv2
import pandas as pd
from torch.autograd import Variable
from pytorch_dcsaunet.DCSAU_Net import Model
from eval_binary import get_transform
from data_loading import predict_binary_class

def predict(test_path, out_path):
    dir_path = test_path
    files = os.listdir(dir_path)
    jpg_files = [file for file in files if (file.endswith(".jpg") or file.endswith(".png"))]
    os.makedirs('debug/',exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = predict_binary_class(test_path, jpg_files, get_transform())
    model = torch.load('save_models/epoch_last.pth')

    model.eval()
    model = model.cuda()
    for file in jpg_files:
        img = cv2.imread(f'/{dir_path}/{file}')
        img = cv2.resize(img, ((256,256)))
        cv2.imwrite(f'{out_path}/{file}',img)
    
    with torch.no_grad():
        for img, img_id in test_dataset:
            img = Variable(torch.unsqueeze(img, dim=0).float(), requires_grad=False).cuda()
            torch.cuda.synchronize()
            pred = model(img)
            torch.cuda.synchronize()
            pred = torch.sigmoid(pred)
            pred[pred >= 0.5] = 1
            pred[pred < 0.5] = 0
            pred_draw = pred.clone().detach()
            img_id = list(img_id.split('.'))[0]
            img_numpy = pred_draw.cpu().detach().numpy()[0][0]  #将预测结果转换为NumPy数组，并选择第一个通道的内容。这假设预测输出是单通道的
            img_numpy[img_numpy==1] = 255 
            cv2.imwrite(f'{out_path}/{img_id}_pred.png',img_numpy)



if __name__ == "__main__":
    predict("/gl_data/zjh/niuaiping/DCSAU-Net-main/test", "/gl_data/zjh/niuaiping/DCSAU-Net-main/debug")