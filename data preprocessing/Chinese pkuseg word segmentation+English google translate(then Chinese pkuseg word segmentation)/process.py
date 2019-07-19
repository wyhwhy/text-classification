#!/usr/bin/env python
# coding: utf-8
import requests
from bs4 import BeautifulSoup
import time
  
def getHTMLText(url):
    try:
        r = requests.get(url, timeout=200)
        r.raise_for_status()
        return r.text
    except Exception as e:
        
        print("Get HTML Text Failed!")
        print(e)
        return 0
  
def google_translate_EtoC(to_translate, num,from_language="en", to_language="ch-CN"):
    #根据参数生产提交的网址
    base_url = "https://translate.google.cn/m?hl={}&sl={}&ie=UTF-8&q={}"
    url = base_url.format(to_language, from_language, to_translate)
    #获取网页
    time.sleep(0.5)
    html = getHTMLText(url)
    if html:
        soup = BeautifulSoup(html, "html.parser")
      
    #解析网页得到翻译结果   
    try:
        result = soup.find_all("div", {"class":"t0"})[0].text
        result=result.strip()
    except:
#        print("Translation Failed!")
#        result = ""
        print("error")
        print(num)
        if num==0:
            n=len(to_translate)
            result=google_translate_EtoC(to_translate[0:round(n*9/10)],0)
        elif num==2:
            result=google_translate_EtoC(to_translate,1)
        elif num==1:
            result=google_translate_EtoC(to_translate,0)    
        else: 
            result=google_translate_EtoC(to_translate,2)

          
    return result
# In[4]:


# 导入需要用到的Python包
import pandas as pd
import numpy as np
#nltk.download('punkt')
import re

# In[4]:


# 加载数据
data = pd.read_csv('d://news.tsv', sep='\t', encoding='utf-8', index_col=False)
data.columns = ['label', 'content']

        # In[7]:
# 去除内容重复的行
data.drop_duplicates(['content'], inplace=True)
#data.to_csv('d://nn.tsv',sep='\t', encoding='utf-8', index=False)
#data.head(10)

data=data.reset_index(drop = True)
#%%
import pkuseg
uncn1 = re.compile(r'[\u0061-\u007a,\u0020]')
uncn2 = re.compile(r'[\u4e00-\u9fa5]')
data['cut_words']=' '        
seg = pkuseg.pkuseg()  # 程序会自动下载所对应的细领域模型

#data=data.head(2000)
for i in range(len(data)):   
    cn = "".join(uncn2.findall(data['content'][i].lower()))#取中文
    cn=seg.cut(cn)
    
    en="".join(uncn1.findall(data['content'][i].lower()))#取英文
    if len(en)>30:
        print(i)
        en=google_translate_EtoC(en,i)
        en = "".join(uncn2.findall(en.lower()))
        en=seg.cut(en)
    else:
        en=[] 
    
    data['cut_words'][i] = cn+en


# In[9]:
#data.head(10)
#print(data['cut_words'][4])
data.to_csv('d://pku+google.tsv', sep='\t', encoding='utf-8', index=False)

# In[10]:


# 将数据分成训练集，验证集，测试集三部分
train, val, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])


# In[11]:


# 保存数据
test.to_csv('d://test.tsv', sep='\t', encoding='utf-8', index=False)
train.to_csv('d://train.tsv', sep='\t', encoding='utf-8', index=False)
val.to_csv('d://dev.tsv', sep='\t', encoding='utf-8', index=False)


## In[12]:
#
#
## 查看数据分布情况
#data.groupby(['label']).count()
#
#
## In[43]:
#
#
#data['cut_words_count'] = data['cut_words'].apply(lambda x: len(x.split()))
#
#
## In[49]:
#
#
#data.describe()
#
#
## In[50]:
#
#
#data.head()


# In[ ]:





# In[ ]:


#google_translate_EtoC(words['content'][519])
#words['content'][519]


# In[57]:





# In[ ]:





# In[ ]:





# In[ ]:




