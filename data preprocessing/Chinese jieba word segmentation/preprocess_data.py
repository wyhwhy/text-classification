#!/usr/bin/env python
# coding: utf-8

# In[4]:


# 导入需要用到的Python包
import pandas as pd
import numpy as np
import jieba


# In[3]:


# 数据清洗函数
stop_words_filepath = 'd:\stop_words.txt'

def get_stop_words(filepath):
    return [w.strip('\n') for w in open(filepath, 'r',encoding='UTF-8').readlines()]

stop_words = get_stop_words(stop_words_filepath)
stop_words_map = dict(zip(stop_words, range(0,len(stop_words))))

def cut_with_remove_stop_words(text):
    text = str(text).replace('\n', '').replace('\xa0', '').replace(' ', '')
    words = jieba.cut(text, cut_all=False)
    return ' '.join([w for w in words if w not in stop_words_map])


# In[4]:


# 加载数据
data = pd.read_csv('d://news.tsv', sep='\t', encoding='utf-8', index_col=False)
data.columns = ['label', 'content']


# In[1]:


data.head(10)


# In[7]:


# 去除内容重复的行
data.drop_duplicates(['content'], inplace=True)


# In[8]:


# 给文本内容分词，取停用此，过滤无用字符
data['cut_words'] = data['content'].map(cut_with_remove_stop_words)


# In[9]:


data.head(10)
data.cut_words[464]
data.to_csv("d://3.csv",encoding="utf_8_sig")

# In[10]:


# 将数据分成训练集，验证集，测试集三部分
train, val, test = np.split(data.sample(frac=1), [int(.6*len(data)), int(.8*len(data))])


# In[11]:


# 保存数据
test.to_csv('d://test.tsv', sep='\t', encoding='utf-8', index=False)
train.to_csv('d://train.tsv', sep='\t', encoding='utf-8', index=False)
val.to_csv('d://dev.tsv', sep='\t', encoding='utf-8', index=False)


# In[12]:


# 查看数据分布情况
data.groupby(['label']).count()


# In[43]:


data['cut_words_count'] = data['cut_words'].apply(lambda x: len(x.split()))


# In[49]:


data.describe()


# In[50]:


data.head()


# In[ ]:





# In[ ]:





# In[57]:





# In[ ]:





# In[ ]:





# In[ ]:




