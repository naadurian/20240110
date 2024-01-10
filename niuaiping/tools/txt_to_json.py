"""
作用：将文件夹里后缀为.txt的文件改为.json后缀（并取原名字的数字部分）
25.xml.txt -> 25.json
"""
import os

folder_path = "/gl_data/zjh/niuaiping/data/txt_labels"  # 文件夹路径
# new_name = "txt_baiban"       # 新文件名
out = os.listdir(folder_path) 
# print(out)
# 遍历文件夹中所有文件
for filename in out:
    # 获取文件的完整路径
    file_path = os.path.join(folder_path, filename)
    
    # print(file_name)
    file_ext = os.path.splitext(filename)[1]# 获取文件的扩展名
    if file_ext == ".json":
        file_ext=".txt"
        file_name = filename.split(".")[0]
        # 拼接新的文件名
        new_file_path = os.path.join(folder_path, file_name+file_ext)
        # # 更改文件名
        os.rename(file_path, new_file_path)

