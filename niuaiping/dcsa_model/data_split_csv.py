import pandas as pd
import numpy as np
import os
import argparse


def pre_csv(data_path,frac):
    np.random.seed(42)
    image_ids = os.listdir(data_path) #获取给定数据路径下的所有文件名,作为数据集的标识符
    data_size = len(image_ids)
    train_size = int(round(len(image_ids) * frac, 0)) #根据给定的分割比例frac计算训练集的大小。使用 round 函数将计算结果四舍五入为最接近的整数
    train_set = np.random.choice(image_ids,train_size,replace=False)#从所有样本中随机选择train_size个样本作为训练集，使用replace=False表示选择的样本不会重复。
    ds_split = [] #创建一个空列表，用于存储每个样本的拆分标签。标签为'train'表示该样本属于训练集，标签为'test'表示该样本属于测试集
    for img_id in image_ids:
        if img_id in train_set:
            ds_split.append('train')
        else:
            ds_split.append('test')
    
    ds_dict = {'image_id':image_ids,
               'category':ds_split 
        }
    df = pd.DataFrame(ds_dict) #使用 Pandas 库将字典转换为数据框
    df.to_csv('src/test_train_data.csv',index=False)#将数据框保存为名为'test_train_data.csv'的CSV文件，位于'src'目录下。index=False表示不将索引列保存到CSV文件中
    print('Number of train sample: {}'.format(len(train_set)))      #输出训练集的样本数量
    print('Number of test sample: {}'.format(data_size-train_size))  #输出测试集的样本数量



if __name__ == '__main__':
    #argparse模块来解析命令行参数，并调用pre_csv函数进行数据集的拆分和保存
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str, default='data/', help='the path of dataset')
    parser.add_argument('--dataset', type=str, default='../datasets/DSB2018/image', help='the path of images') # issue 16
    parser.add_argument('--size', type=float, default=0.9, help='the size of your train set')
    args = parser.parse_args()  #解析命令行参数，并将解析结果存储在args对象中
    os.makedirs('src/',exist_ok=True)
    pre_csv(args.dataset,args.size)
