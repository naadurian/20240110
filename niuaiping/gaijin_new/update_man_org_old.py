import sys
import re
#from recognize_man_org import get_man_org
from recognize_man_org import get_man_org
import pymysql as pmq



con_mysql = pmq.connect(host='101.36.73.80',user='root',password='4b4767af8182be0e',database='pm345')
cursor_mysql = con_mysql.cursor()
sql1=''
try:
    #id = 2946269    
    #id = 2425625   
    #id = 2969236   #徐淑艳   刘海波   姜 华
    id= 3018651   #卢工 张工 薛先生
    
    #sql = f"select id, content from jdggxq where id >={2909019} and \id <={3909029} ORDER BY RAND() LIMIT 500"
    #sql = "select id, gglx,url,fb_day,created_at,content from jdggxq where id >={} and id <={}".format(3009000, 3009100)  #十万
    sql = "select id, gglx,url,fb_day,created_at,content from jdggxq where id >={} and id <={}".format(int(id), int(id))
    # sql = "select id, gglx,url,fb_day,created_at,content from jdggxq where id >={} and id <={}".format(int(id), int(id)+5000)  

    res = cursor_mysql.execute(sql)
    ans = cursor_mysql.fetchall()
    for line in ans:
        id = line[0]
        #print(line[2])
        #print(str(id)+":"+str(line[1])+":"+line[2]) 
        info = get_man_org(line[5])
        #print(info)
        info = info["man"]
        #print(info)
        for i in info:
            if "name" in i.keys():
                a = i["name"]
            else:
                a = " "
            if "phone" in i.keys():
                b = i["phone"]
            else:
                b = " "
            if "tel" in i.keys():
                c = i["tel"]
            else:
                c = " "
       
            print(id,a,b,c)
        #print(info)
    cursor_mysql.close()
    con_mysql.close()
except Exception as e:
    print(str(e))
    print("dddcddddddddddddddddddddd")

