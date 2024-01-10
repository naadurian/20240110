from paddlenlp import Taskflow
import paddlenlp
import sys
import re
import pymysql as pmq
import jionlp
from itertools import zip_longest

path1 = "/gl_data/"

schema = ["人名", "地址", "机构","地点"]

ie = Taskflow('information_extraction', schema=schema, task_path=path1 + "data/model/information_extraction/uie-base")
tag = Taskflow("pos_tagging", task_path=path1+"data/model/pos_tagging")

result = {"man":[], "organization":[]}


"""
@Goal: 清洗Paddle结果
"""
def clear_paddle_IE(paddle_data) -> dict:
    """清洗paddlepaddle解析文件
    经多层嵌套的list 简单保留name:text
    """
    result = dict()
    for item in paddle_data[0]:
        result[item] = paddle_data[0].get(item)[0]["text"]
    return result


def clear_paddle_IE_all(paddle_data):
    
    curr_result = dict()
    for item in paddle_data[0]:
        items = paddle_data[0].get(item)
        curr_list=[]
        items.sort(key=lambda x:x["start"])
        for elem in items:
            curr_list.append(elem['text'])
        ans = []
        for r in curr_list:
            if r not in ans:
                ans.append(r)
        curr_result[item] = ans

    return curr_result

"""
@Goal: 清洗文本
"""
def clear_baoming_info(content: str) -> str:
    content = jionlp.remove_html_tag(content)
    pattern1 = r'\{[^{}]*?\}'
    pattern2 = r'\<[^<>]*?\>'
    content = re.compile(pattern2).sub('', content)
    content = re.compile(pattern1).sub('', content)
    content = content.replace("\xa02", "").replace("\xa01", "").replace("\xa0", "").replace(" ", "").replace(";","").replace(
        "\u3000", "").replace("&nbsp", "").replace("■", "").replace("★", "").replace("_", "").replace("\\t", "").split(
        "\\n")
    content = [item for item in content if item != '' or "notice" not in item or "font" not in item]
    return "".join(content)

"""
@Goal: 邮箱
"""
def get_email(content):
    pattern = r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)'
    links = re.findall(pattern, content)
    ans = []
    for r in links:
        if r not in ans:
            ans.append(r)
    return ans

"""
@Goal: 电话
"""
def get_phone(content):
    # tel = "\d{11}"
    tel_1 = "\d{3}-\d{8}|\d{4}-\d{7,8}|\d{3}-\d{3}-\d{4}"
    tel_2 = "\d{8}"
    res_list = re.findall(tel_1, content) #+ re.findall(tel, content)
    res_list_1 = re.findall(tel_2, content)
    for phone1 in res_list_1:
        is_has = 0
        for phone in res_list:
            if phone.find(phone1) >= 0:
                is_has = 1
        if is_has == 0:
            res_list.append(phone1)
    ans = []
    for r in res_list:
        if r not in ans:
            ans.append(r)
    return ans

"""
@Goal: 手机号
"""
def get_phone_num(content):
    tel = "1[0-9]{10}"
    res_list = re.findall(tel, content)
    ans = []
    for r in res_list:
        if r not in ans:
            ans.append(r)
    return ans

"""
@Goal: 机构电话
"""
def get_jigou_tel(content):
    tel = "1[0-9]{10}"
    tel_1 = "\d{3}-\d{8}|\d{4}-\d{7,8}|\d{3}-\d{3}-\d{4}|\d{11}"
    tel_2 = "\d{8}"
    res_list = re.findall(tel_1, content) + re.findall(tel, content)
    res_list_1 = re.findall(tel_2, content)
    for phone1 in res_list_1:
        is_has = 0
        for phone in res_list:
            if phone.find(phone1) >= 0:
                is_has = 1
        if is_has == 0:
            res_list.append(phone1)
    ans = []
    for r in res_list:
        if r not in ans:
            ans.append(r)
    return ans
    
"""
@Goal: 姓名
"""
def get_name(content):
    res = []
    tag_list =  tag(content)
    for i in range(len(tag_list)):
        if tag_list[i][1] == "PER" or tag_list[i][1] == "nr":
            res.append(tag_list[i][0])
    ans = []
    for r in res:
        if r not in ans:
            ans.append(r)
    return ans

"""
@Goal: 机构
"""
def get_org(content):
    res = []
    tag_list =  tag(content)
    for i in range(len(tag_list)):
        if tag_list[i][1] == "ORG" or tag_list[i][1] == "nt":
            res.append(tag_list[i][0])
    return list(set(res))

"""
@Goal: 地址
"""
def get_address(content):
    res = []
    tag_list =  tag(content)
    for i in range(len(tag_list)):
        if tag_list[i][1] == "LOC" or tag_list[i][1] == "ns":
            res.append(tag_list[i][0])
    return list(set(res))

"""
@Goal: 评审专家
"""
def get_reviewer(content, keyword, relation, review_org):
    try:
        ans = []
        content = content.replace("\r","").replace("\n","").replace("\t","")
        review_content = content[content.find(keyword):content.find(keyword)+60]
        # print("--------------")
        # print(review_content)
        name = get_name(review_content)
        for item in name:
            
            new_entry = {
                "name": item.replace("\n",""), 
                "phone": "",
                "email": "",
                "address": "",  
                "sex": "",
                "relation": relation,
                "in_org": review_org}
            ans.append(new_entry)

        # print("评审专家")
        # print(ans)
        return ans
    except Exception as e: 
        print("Error1: ", e)
        return []


"""
@Goal: 得到每个机构的信息
"""
def get_ORG(content, relation):
    try:
        # ans = [{"name":"","tel":"","email":"","address":"","fax":"","relation":""}]
        ans = []
        if content == "":
            return []
        content = content.replace("\r","").replace("\n","").replace("\t","")
        dot_index = content.rfind(".")
        if content[dot_index-1].isdigit():
            content = content[:dot_index-1] + content[dot_index:]
        # print("--------------")
        #print(content)
        
        """
        给一个循环 拿到机构名称 电话 邮箱 地址 传真 和关系 然后append到ans  返回ans
        传真和电话形式一样  不建议取
        """
        address = []
        org = []
        # org = get_org(content)
        # address = get_address(content)
        tel = get_jigou_tel(content)
        email = get_email(content)
        extract = clear_paddle_IE_all(ie(content))
        if org == [] and extract.get("机构") is not None:
            for jigou in extract["机构"]:
                org.append(jigou)
        if address == [] and extract.get("地址") is not None:
            for dizhi in extract["地址"]:
                address.append(dizhi)
        if address == [] and extract.get("地点") is not None:
            for didian in extract["地点"]:
                address.append(didian)
        
        ## 取最长邮箱 取完删掉
        if email:
            longest_email = max(email, key=len)
            email.remove(longest_email)
        else: 
            longest_email = ""
        
        ## 取最长地址
        if address:
            longest_address = max(address, key=len)
        else: 
            longest_address = ""
            
        ## 取最长电话号
        if tel:
            longest_tel = max(tel, key=len)
        else: 
            longest_tel = ""

        for item in org:
            if item == None:
                break

            new_entry = {
                "name": item.replace("\n",""), 
                "tel": longest_tel.replace("\n",""), 
                "email": longest_email.replace("\n",""), 
                "address": longest_address.replace("\n",""), 
                "relation": relation}
            ans.append(new_entry)
        # print("ans")
        # print(ans)
        return ans
    except Exception as e: 
        print("Error2: ", e)
        return []


##添加到列表中
def tianjia(index):
            address = []
            org = []
                ## 性别
            if "先生" in index:
                sex = "男"
            elif "女士" in index or "小姐" in index:
                sex = "女"
            else:
                sex = ""
                
            ## 取最长邮箱 取完删掉
            if index:
                longest_email = max(index, key=len)
                index.remove(longest_email)
            else: 
                longest_email = ""
            
            ## 取最长地址
            if address:
                longest_address = max(address, key=len)
            else: 
                longest_address = ""
            
            ## 取最长机构名
            if org:
                longest_org = max(org, key=len)
            else: 
                longest_org = ""       
                        
            # new_entry = {
            #     "name": index.replace("\n",""), 
            #     "phone": ph.replace("\n",""), 
            #     "email": longest_email.replace("\n",""), 
            #     "tel": tel.replace("\n",""),
            #     "address": longest_address.replace("\n",""), 
            #     "sex": sex, 
            #     "relation": relation, 
            #     "in_org": longest_org.replace("\n","")}
            # ans.append(new_entry)
            return sex,longest_email,longest_address,longest_org
"""
@Goal: 得到每个人的信息
"""
def get_MAN(content, relation):
    try:
        ans = []
        i =0 
        if content == "":
            return []
            
        content = content.replace("\r","").replace("\n","").replace("\t","")
        dot_index = content.rfind(".")
        if content[dot_index-1].isdigit():
            content = content[:dot_index-1] + content[dot_index:]
        # print("--------------")
        #print("content:",content)
        
        """
        给一个循环 拿到名字 手机 邮箱 地址 性别 关系 和机构 然后append到ans  返回ans
        """
        address = []
        org = []
        name = []
        ## 词性没抽出来 用uie抽取
        extract = clear_paddle_IE_all(ie(content))
        #print(extract)
        if name == [] and extract.get("人名") is not None:
            for renming in extract["人名"]:
                name.append(renming)
        #print(name)
        if org == [] and extract.get("机构") is not None:
            for jigou in extract["机构"]:
                org.append(jigou)
        if address == [] and extract.get("地址") is not None:
            for dizhi in extract["地址"]:
                address.append(dizhi)
        if address == [] and extract.get("地点") is not None:
            for didian in extract["地点"]:
                address.append(didian)
            
        if name == []:    
            name = get_name(content)
        # org = get_org(content)
        # address = get_address(content)
        email = get_email(content)
        
        ## phone为座机号 phone_num为手机号
        ## 查重座机号 如果一样就扔掉
        phone = get_phone(content)
        phone_num = get_phone_num(content)
        filter_phone = []
        for ph in phone: 
            is_has = 0
            for ph_num in phone_num:
                if ph_num.find(ph) >= 0:
                    is_has = 1
            if is_has == 0:
                filter_phone.append(ph)

        ## 如果联系人为以下列表中字段  模型抽不出来
        name_append_list = ["客服人员", "项目四部","项目一部","项目二部","项目三部"]
        if name == []:
            for nn in name_append_list:
                if nn in content:
                    name.append(nn)
               
        ## 如果联系人为以下列表中字段  模型可能会抽出来一个姓 
        ## 这里是处理模型连姓都没抽出来的情况 
        job_title = ["助理","老师","先生","女士","小姐","主任","经理","医生","主管","总监","警官","科长","班长","干事"]
        job_content = content
        #print(job_content)       
        
        for item,ph,tel in zip_longest(name,phone_num,filter_phone):
            if ph == None:
                ph = ""
            if tel == None:
                tel = ""
            if item == None:
                break            
            
            ## 处理如果抽出来一个字 补全名字
            #以抽取到的姓（一个字）为循环,查找姓之后的两个字是否在job_title中，确保不会丢掉名字
            if len(item) == 1:
                    #print("name中只有姓：",item)
                    pos = job_content.find(item)
                     
                    #print(pos,pos_last,job_content[pos+1:pos+3])
                    while  job_content.find(item) >= 0:   #匹配"黄"字时要循环匹配，直到pos为job_content的最后一个字，因为有好几个黄字，所以匹配不到女士两个字
                        
                        if job_content[pos+1:pos+3] in job_title:
                            name_job = job_content[pos:pos+3]
                            job_content = job_content[pos+3:]
                            #print(job_content)
                            item = name_job
                            pos = len(job_content)
                            break
                        else:
                            job_content = job_content[pos+3:]
                            #print(job_content)
                            pos = job_content.find(item)
                           
            #sex,longest_email,longest_address,longest_org = tianjia(item)
            ## 性别
            if "先生" in item:
                sex = "男"
            elif "女士" in item or "小姐" in item:
                sex = "女"
            else:
                sex = ""
                
            ## 取最长邮箱 取完删掉
            if email:
                longest_email = max(email, key=len)
                email.remove(longest_email)
            else: 
                longest_email = ""
            
            ## 取最长地址
            if address:
                longest_address = max(address, key=len)
            else: 
                longest_address = ""
            
            ## 取最长机构名
            if org:
                longest_org = max(org, key=len)
            else: 
                longest_org = ""       
                        
            new_entry = {
                "name": item.replace("\n",""), 
                "phone": ph.replace("\n",""), 
                "email": longest_email.replace("\n",""), 
                "tel": tel.replace("\n",""),
                "address": longest_address.replace("\n",""), 
                "sex": sex, 
                "relation": relation, 
                "in_org": longest_org.replace("\n","")}
            ans.append(new_entry)
        #print(ans)    
        #print(job_content)
         ## 处理没抽出姓的，只有称呼的情况  ex：项目监督人：先生  
        #pos = -1
        for title in job_title:
                pos = job_content.find(title)
                #print(job_content[pos-2:],title)
                if pos >= 0:
                    if title in job_content and job_content[pos-1] == '：' :
                        #print(job_content[pos-1])
                        job_index = job_content.find(title)
                        name_job = job_content[job_index:job_index+len(title)]
                        #print("name_job",name_job)
                        job_content = job_content[job_index+len(title):]
                        #print(job_content)
                        name.append(name_job)
                    #print("name的值为：",name) 
                    if len(name) >0:
                        while i < len(name):
                            if len(name[i]) >1:
                                #print(name[i],i)
                                ## 性别
                                if "先生" in name[i]:
                                    sex = "男"
                                elif "女士" in name[i] or "小姐" in name[i]:
                                    sex = "女"
                                else:
                                    sex = ""
                                    
                                ## 取最长邮箱 取完删掉
                                if email:
                                    longest_email = max(email, key=len)
                                    email.remove(longest_email)
                                else: 
                                    longest_email = ""
                                
                                ## 取最长地址
                                if address:
                                    longest_address = max(address, key=len)
                                else: 
                                    longest_address = ""
                                
                                ## 取最长机构名
                                if org:
                                    longest_org = max(org, key=len)
                                else: 
                                    longest_org = ""       
                                new_entry = {
                                "name": title.replace("\n",""), 
                                "phone": ph.replace("\n",""), 
                                "email": longest_email.replace("\n",""), 
                                "tel": tel.replace("\n",""),
                                "address": longest_address.replace("\n",""), 
                                "sex": sex, 
                                "relation": relation, 
                                "in_org": longest_org.replace("\n","")}
                                ans.append(new_entry) 
                            i=i+1
                    

        return ans
    except Exception as e: 
        print("Error3: ", e)
        return []

"""
@Goal: 测试所有信息
"""
def get_man_org_old(content):
    man = []
    org = []
    content = clear_baoming_info(content)
    # 做分文本工作
    pos_list = ['报名联系方式','方式联系','机构联系方式','单位联系方式','人联系方式','人及联系方式','联系方式\n','、联系方式','.联系方式','联系事项','．联系方式','业务咨询：','、项目联系人','、合同主体', "、采购单位信息"] 
    pos = -1 
    for i in pos_list:
        pos = content.find(i)
        if pos > 0:
            break
    need_content = ''
    if pos > 0:
        need_content = content[pos:]
    else:
        need_content = ""
    end_pos = need_content.find('附件')
    if end_pos > 50:
        need_content = need_content[0:end_pos]
        
    ## 新疆政府采购网 截取中间一部分联系人段落
    end_pos_xinjiang = need_content.find("、成交信息")
    if end_pos_xinjiang != -1:
        need_content = need_content[0:end_pos_xinjiang]
        
    caigou_content = need_content
    #print(need_content)
    daili_content = ""
    pos1 = caigou_content.find('代理')     
    if pos1 < 0:
        pos1 = caigou_content.find('运维')       

    if pos1 > 0 and len(need_content) - pos1 < 20:
        pos1 = -1
    if pos1 > 0:
        caigou_content = need_content[0:pos1]
        daili_content = need_content[pos1:]

    out_case = ["药品"]
    jiandu_content = ""
    pos2 = need_content.find('监督')
    if pos2 < 0:
        pos2 = need_content.find('质疑')
    if pos2 >= 0:
        for element in out_case:
            if need_content[pos2-2:pos2] == element:
                pos2 = -1
    if pos2 >= 0:
        daili_content = need_content[pos1:pos2]
        jiandu_content = need_content[pos2:]
    caigou_man =  get_MAN(caigou_content, 1)
    daili_man =  get_MAN(daili_content, 2)
    jiandu_man = get_MAN(jiandu_content, 3)
    for m in caigou_man:
        man.append(m)
    for m in daili_man:
        man.append(m)
    for m in jiandu_man:
        man.append(m)
    caigou_org = get_ORG(caigou_content, 1)
    daili_org = get_ORG(daili_content, 2)
    for o in caigou_org:
        org.append(o)
    for o in daili_org:
        org.append(o)
    
    ## 获取代理机构名称
    if daili_man != []:
        review_org = daili_man[0]['in_org']
    elif daili_org != []:
        review_org = daili_org[0]['name']
    else: 
        review_org = ""
    
    reviewer = ["评审专家"]
    for review in reviewer:
        if review in content:    
            pingshen_man = get_reviewer(content, review, 4, review_org)
            for m in pingshen_man:
                man.append(m)
            break
    
    result["man"] = man
    result["organization"] = org
    # print("***************")
    #print(need_content)
    # print(content)
    new_list = []
    for i in range(len(result['organization'])):
        is_find = 0
        for man in result['man']:
            if result['organization'][i]['name'].find(man['name']) >= 0:
                is_find = 1
                break
        if is_find == 0:
            new_list.append(result['organization'][i])
    result['organization'] = new_list
    # print("===========================================================")
    return result

def is_shaoshu(name):
    symbol_index = 0
    name_index = 0
    new_name = ''
    for ch in name:
        if not (u'\u4e00' <= ch <= u'\u9fff'):
            if ch == '·' or ch == '.' or ch == '•' or ch == '﹒' or ch == '`' or ch == '▪':
                symbol_index += 1
                if symbol_index == 1 and name_index > 3:
                    new_name += '·'
                continue
            continue
        name_index += 1
        new_name += ch
    return new_name


#如果提取出的姓名最后一个字是六或七，则删去
def remove_six_seven(lst):
    modified_lst = []
    
    if lst[-1] in ['六', '七']:
        modified_name = lst[:-1]
    else:
        modified_name = lst
    modified_lst.append(modified_name)
    return modified_lst

def get_man_org(content):
    result = get_man_org_old(content)
    #print("改进里面",result)
    name = result["man"]
    #print(name)
    for na in name:
        #print(na["name"])
        mingzi = na["name"]
        if mingzi[0] =='：':
            name = mingzi[1:]    #先生
            na["name"] = name
        name = remove_six_seven(na["name"])
        na["name"] = name[0]
        name = is_shaoshu(name[0])
       # print(name)
        na["name"] = name
        #print(na,na["name"])
    return result   

        
         
