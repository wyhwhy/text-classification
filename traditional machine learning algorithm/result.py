import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors.nearest_centroid import NearestCentroid
import time

# 加载预处理后的训练数据
train_df = pd.read_csv('d://train.tsv', sep='\t')
test_df = pd.read_csv('d://test.tsv', sep='\t')
val_df = pd.read_csv('d://dev.tsv', sep='\t')


x_train = train_df['cut_words']
x_val = val_df['cut_words']
x_test = test_df['cut_words']

x_train=pd.concat([x_train, x_val], axis=0)
x_train=x_train.reset_index(drop = True)
# 标签编码
y_train = train_df.label
y_val = val_df.label
y_test = test_df.label

y_train=pd.concat([y_train, y_val], axis=0)
y_train=y_train.reset_index(drop = True)

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

train_texts=x_train
train_labels=y_train
test_texts=x_test
test_labels=y_test


#v=TfidfVectorizer(decode_error='replace', encoding='utf-8')
#train_texts=v.fit_transform(train_texts.values.astype(str))
#test_texts=v.fit_transform(test_texts.values.astype(str))
#train_labels=v.fit_transform(train_labels.values.astype('U'))
#test_labels=v.fit_transform(test_labels.values.astype('U'))

# 贝叶斯
for i in range(10000,999,-1000):
    print(i)
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',NearestCentroid())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("NearestCentroid准确率为：",np.mean(predicted==test_labels))# 
    print("训练时间为%d秒" % duration)
    
#    print(i)
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',MultinomialNB())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("MultinomialNB准确率为：",np.mean(predicted==test_labels))# 
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',SGDClassifier())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("SGDClassifier准确率为：",np.mean(predicted==test_labels))# 
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',LogisticRegression())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("LogisticRegression准确率为：",np.mean(predicted==test_labels))# 
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',SVC())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("SVC准确率为：",np.mean(predicted==test_labels))
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',LinearSVC())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("LinearSVC准确率为：",np.mean(predicted==test_labels))
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',MLPClassifier())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("MLPClassifier准确率为：",np.mean(predicted==test_labels))# 
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',KNeighborsClassifier())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("KNeighborsClassifier准确率为：",np.mean(predicted==test_labels))#
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',RandomForestClassifier(n_estimators=8))])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("RandomForestClassifier准确率为：",np.mean(predicted==test_labels))# 
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',GradientBoostingClassifier())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("GradientBoostingClassifier准确率为：",np.mean(predicted==test_labels))# 
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',AdaBoostClassifier())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("AdaBoostClassifier准确率为：",np.mean(predicted==test_labels))# 
    print("训练时间为%d秒" % duration)
    
    text_clf=Pipeline([('tfidf',TfidfVectorizer(max_features=i)),
                       ('clf',DecisionTreeClassifier())])
    t0=time.time()
    text_clf=text_clf.fit(train_texts.values.astype('U'),train_labels)
    duration=time.time()-t0
    predicted=text_clf.predict(test_texts)
    print("DecisionTreeClassifier准确率为：",np.mean(predicted==test_labels))
    print("训练时间为%d秒" % duration)
