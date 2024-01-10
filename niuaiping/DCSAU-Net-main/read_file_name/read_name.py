import os
import shutil

folderA = '/gl_data/zjh/niuaiping/DCSAU-Net-main/data2/masks'  # 文件夹A的路径
folderB = '/gl_data/zjh/niuaiping/0/2'  # 文件夹B的路径
folderC = '/gl_data/zjh/niuaiping/DCSAU-Net-main/data2/images'  # 文件夹C的路径

filesA = os.listdir(folderA)  # 获取文件夹A中的所有文件名

for filenameA in filesA:
    if os.path.exists(os.path.join(folderB, filenameA)):
        shutil.copy2(os.path.join(folderB, filenameA), os.path.join(folderC, filenameA))