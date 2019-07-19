import pandas as pd
import re
import numpy as np
import pkuseg

data = pd.read_csv('d://processed4news.tsv', sep='\t', encoding='utf-8', index_col=False)
data.columns = ['label', 'content']

#data.head(10)

#%%
        
data['cut_words'] = ' '
seg = pkuseg.pkuseg()  # 程序会自动下载所对应的细领域模型
for i in range(len(data)):   
    
    cn = "".join(uncn2.findall(data['content'][i].lower()))#取中文
    cn=seg.cut(cn)
    
    data['cut_words'][i] = cn


# In[9]:
#data.head(10)
#print(data['cut_words'][4])

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




