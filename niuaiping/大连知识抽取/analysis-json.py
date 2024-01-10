import json

#从.json文件中加载json数据
with open('/gl_data/zjh/niuaiping/大连知识抽取/sadahang.json','r') as file:
          data = json.load(file)


entity_list = []
relation_list = []

def extract_entity(obj):
          if isinstance(obj,dict):
                    for key in obj:
                              value = obj[key]
                              if key=='sourceLabel' or key=='targetLabel' :
                                        if value not in entity_list:
                                                  entity_list.append(value)
                              else:
                                        extract_entity(value)
          elif isinstance(obj, list):
                    for item in obj:
                              extract_entity(item)

def extract_relation(rel):
          for item in rel["elabels"]:
                    relation = [item["sourceLabel"], item["label"], item["targetLabel"]]
                    relation_list.append(relation)

if __name__ == "__main__":
          extract_entity(data)
          print("实体列表为：",entity_list)
          
          extract_relation(data)
          print("实体关系为：",relation_list)
          
"""
实体列表为： ['一级分行', '二级分行', '支行', '网点', '员工', '业务', '岗位', '客户', '龙易行', '设备', '高柜', '总行']
实体关系为：
[['一级分行', '下属二级分行', '二级分行'], 
 ['二级分行', '下属支行', '支行'], 
 ['支行', '下属网点', '网点'], 
 ['二级分行', '下属网点', '网点'],
 ['网点', '包含', '员工'],
 ['员工', '授权', '员工'],
 ['员工', '复核', '员工'],
 ['员工', '办理', '业务'], 
 ['员工', '任职', '岗位'], 
 ['客户', '办理', '业务'],
 ['员工', '负责', '业务'], 
 ['支行', '拥有', '龙易行'],
 ['支行', '拥有', '设备'],
 ['支行', '拥有', '高柜'], 
 ['网点', '拥有', '设备'], 
 ['网点', '拥有', '高柜'], 
 ['网点', '拥有', '龙易行'],
 ['总行', '下属一级分行', '一级分行']]
 """