import os
import shutil

"""
根据 json_path 中的名称在 images_path 中查找对应的原图,将其保存在 new_folder_path
    json_path: 只包含json图,从images_path中挑出来的
    images_path: 包含原图和json图
"""
 
def find_and_save_images(json_path, images_path,new_folder_path):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(json_path):
        image_name = filename.split('.')[0]
        file_path = os.path.join(images_path, image_name)
        
        #改为jpg格式
        new_image_name = image_name + ".jpg"
        if new_image_name in os.listdir(images_path):
            # 将原图复制到新文件夹中
            shutil.copy(os.path.join(images_path, new_image_name), new_folder_path)
 
# 调用函数进行操作
json_path = '/gl_data/zjh/niuaiping/data/json_labels'
images_path = '/gl_data/zjh/niuaiping/2_白斑'
new_folder_path = '/gl_data/zjh/niuaiping/data/images'
find_and_save_images(json_path,images_path, new_folder_path)