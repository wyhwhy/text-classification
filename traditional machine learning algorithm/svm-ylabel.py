import pandas as pd
import numpy as np
from keras import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.linear_model import SGDClassifier


# In[2]:


# 加载预处理后的训练数据
train_df = pd.read_csv('d://train.tsv', sep='\t')
test_df = pd.read_csv('d://test.tsv', sep='\t')
val_df = pd.read_csv('d://dev.tsv', sep='\t')


# In[3]:


# 将训练数据转换token数组
x_train = train_df.cut_words.map(lambda x: str(x).split(' ')).tolist()
x_val = val_df.cut_words.map(lambda x: str(x).split(' ')).tolist()
x_test = test_df.cut_words.map(lambda x: str(x).split(' ')).tolist()


# In[10]:


max_features = 10000 #作为特征的单词个数
maxlen = 200          #每个样本取这么多特征词
embedding_dim = 128
n_classes = 10

tokenizer = Tokenizer(num_words=max_features) 
tokenizer.fit_on_texts(x_train)

x_train = tokenizer.texts_to_sequences(x_train)
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)

# 将整数列表转换成形状为 (samples, maxlen) 的二维整数张量
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen) 
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
x_val = preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

# In[4]:


# 标签编码
y_train = train_df.label
y_val = val_df.label
y_test = test_df.label

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)


svm_clf=SGDClassifier(loss='log',penalty='l2',alpha=1e-4,max_iter=1000,random_state=10)
svm_clf.fit(x_train,y_train)
predicted=svm_clf.predict(x_test)
print(np.mean(predicted==y_test))
