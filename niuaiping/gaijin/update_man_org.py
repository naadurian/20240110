import sys
import re
from recognize_man_org import get_man_org
import pymysql as pmq
import time


con_mysql = pmq.connect(host='101.36.73.80',user='root',password='4b4767af8182be0e',database='pm345')
cursor_mysql = con_mysql.cursor()
sql1=''
try:
    id = 2946269 #2909221 #2911095 #1620750  
    sql = "select id, url,fb_day,created_at,content from jdggxq where  id >={} and id <={}".format(id, id) 
    #sql = "select id, url,fb_day,created_at,content from jdggxq where gglx=26 and id >={} and id <={}".format(3009000, 3019000) 
    print(sql)
    res_sql = []
    res = cursor_mysql.execute(sql)
    ans = cursor_mysql.fetchall()
    for line in ans:
        id = line[0]
        print(str(id)+":"+str(line[1])) 
        info = get_man_org(line[4])
        print(info)

    cursor_mysql.close()
    con_mysql.close()
except Exception as e:
    print(str(e))
    print("dddcddddddddddddddddddddd")

