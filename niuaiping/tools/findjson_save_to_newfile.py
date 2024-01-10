import os
import shutil
 
"""
查找文件夹中的所有json文件，并保存到新的文件夹中
"""
def select_json_file(folder_path):
    # 获取文件夹中的所有文件
    files = os.listdir(folder_path)
    
    
    for file in files:
        # 检查文件是否为JSON文件
        if file.endswith(".json"):#判断字符串是否以指定字符结尾
            # 构建文件的完整路径
            file_path = os.path.join(folder_path, file)
            
            # 读取JSON文件内容
            with open(file_path, "r") as json_file:
                json_data = json_file.read()
            
            # 输出JSON文件内容到新的文件夹
            new_folder_path = "/gl_data/zjh/niuaiping/data/json_labels"
            new_file_path = os.path.join(new_folder_path, file)
            with open(new_file_path, "w") as new_file:
                new_file.write(json_data)
            
            # 可选：删除原始文件
            # os.remove(file_path)
 
# 调用函数，传入文件夹路径
select_json_file("/gl_data/zjh/niuaiping/2_白斑")