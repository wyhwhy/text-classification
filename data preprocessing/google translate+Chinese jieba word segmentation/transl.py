import os
import pandas as pd
import urllib.request
import random   
import time
os.system("activate tensorflow")
os.system("python E:\wyh\大学资料\个人资料\大三下\暑期实习\课题\子文件\HandleJs.py")  
from HandleJs import Py4Js   
user_agents=['Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.80 Safari/537.36','Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/34.0.1847.137 Safari/537.36 LBBROWSER','Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0','Mozilla/5.0 (Windows NT 6.2; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0']
proxies={'http':'http://121.63.199.140','https':'https://1.199.31.121'}
def open_url(url,i):  
    cookies = dict(uuid='b18f0e70-8705-470d-bc4b-09a8da617e15',UM_distinctid='15d188be71d50-013c49b12ec14a-3f73035d-100200-15d188be71ffd')
    headers = {'User-Agent':random.choice(user_agents)}
    req = urllib.request.Request(url = url,headers=headers)    
    
    response = urllib.request.urlopen(req)    
    data = response.read().decode('utf-8')    
    return data    
    
def translate(content,tk,i):    
    if len(content) > 4891:    
        content=content[0:4890]    
        
    content = urllib.parse.quote(content)    
        
    url = "http://translate.google.cn/translate_a/single?client=t"+ "&sl=en&tl=zh-CN&hl=zh-CN&dt=at&dt=bd&dt=ex&dt=ld&dt=md&dt=qca"+"&dt=rw&dt=rm&dt=ss&dt=t&ie=UTF-8&oe=UTF-8&clearbtn=1&otf=1&pc=1"+"&srcrom=0&ssel=0&tsel=0&kc=2&tk=%s&q=%s"%(tk,content)    
        
    #返回值是一个多层嵌套列表的字符串形式，解析起来还相当费劲，写了几个正则，发现也很不理想，  
    #后来感觉，使用正则简直就是把简单的事情复杂化，这里直接切片就Ok了    
    result = open_url(url,i)    
        
    end = result.find("\",")    
    if end > 4:    
        result=result[4:end]
    return result
    
def main():    
    js = Py4Js()    
    words = pd.read_csv('d://processed1news.tsv', sep='\t', encoding='utf-8', index_col=False)
    word=words.head(50)    
    for i in range (len(word['content'])):
        time.sleep(3)
        content=word['content'][i]
        tk = js.getTk(content)    
        word['content'][i]=translate(content,tk,i)
        print(word['content'][i])
    #word.to_csv('d://processed2news.tsv', sep='\t', encoding='utf-8', index=False)
    word.to_csv('d://processed2news.csv')
        
if __name__ == "__main__":    
    main()
