#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers
from keras import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


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
maxlen = 400          #每个样本取这么多特征词
embedding_dim = 256
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


# In[11]:


x_train


# In[4]:


# 标签one-hot编码
y_train = train_df.label
y_val = val_df.label
y_test = test_df.label

le = LabelEncoder()
y_train = le.fit_transform(y_train).reshape(-1, 1)
y_val = le.transform(y_val).reshape(-1, 1)
y_test = le.transform(y_test).reshape(-1, 1)

ohe = OneHotEncoder()
y_train = ohe.fit_transform(y_train).toarray()
y_val = ohe.transform(y_val).toarray()
y_test = ohe.transform(y_test).toarray()


# In[5]:


y_train


# In[12]:


from keras import Sequential
from keras.layers import Flatten, Dense, Embedding,LSTM,GRU,Dropout,Conv1D,GlobalMaxPooling1D,ZeroPadding1D
from keras.layers import Convolution1D, MaxPooling1D,Bidirectional,MaxPooling2D,Reshape
#from recurrent import GRU

# 训练模型
model = Sequential()

model.add(Embedding(max_features + 1, embedding_dim, input_length=maxlen))
#model.add(Embedding(max_features, embedding_dim))
'''
model.add(Conv1D(filters=64,kernel_size=3,padding='valid',activation='relu',strides=1))
#model.add(Dropout(0.2))
#model.add(LSTM(32, dropout=0.5, recurrent_dropout=0.3))

#model.add(Dense(embedding_dim, activation='relu'))

model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))

model.add(LSTM(32))
'''

#model.add(ZeroPadding1D(padding=2))

#layer.output=tf.expand_dims(layer.output,-1)
model.add(Bidirectional(LSTM(64,return_sequences=True)))#Bidirectional
#LSTM.output=tf.expand_dims(LSTM.output,-1)
#model.add(ZeroPadding1D(padding=2))
#print(LSTM.output_shape)
model.add(Convolution1D(nb_filter=128,filter_length=5,border_mode='valid',activation='relu',subsample_length=1))
model.add(MaxPooling1D())
#model.add(Reshape((6272, )))
model.add(Reshape((25344, )))
model.add(Dropout(0.3))



#model.add(Bidirectional(LSTM(32)))#,return_sequences=True
model.add(Dense(n_classes,activation='softmax'))#, input_shape=(1,8187,10)

model.compile(loss="categorical_crossentropy",
              optimizer='rmsprop',
              metrics=["accuracy"])
model.summary()


# In[ ]:





# In[13]:


# 输入数据进行训练 
history = model.fit(x_train, y_train, epochs=20, batch_size=128, validation_data=(x_val, y_val))
 

# In[15]:


# 画出精确度曲线
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[16]:


# 在测试集上的表现
from sklearn import metrics

test_result = model.predict(x_test)
print(metrics.classification_report(np.argmax(test_result, axis=1),
                                      np.argmax(y_test, axis=1)))


# In[ ]:





# In[ ]:




