# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import jieba
train_df = pd.read_csv('/home/jovyan/shiyan_data/cnews.train.txt', sep='\t', header=None)
test_df = pd.read_csv('/home/jovyan/shiyan_data/cnews.test.txt', sep='\t', header=None)

#或者通过columns来查看
train_df.columns = ['Subject', 'Content']
train_df['Subject'].value_counts().sort_index()

print(train_df)

#%%分词
def cut_context(data):
    words=data.apply(lambda x: ' '.join(jieba.cut(x)))
    return words
#停词过滤
stopwords=open('/home/jovyan/data/7-text_classification/cnews.vocab.txt',encoding='utf-8')
stopwords_list=stopwords.readlines()
stopworsd=[x.strip() for x in stopwords_list]#去掉每行头尾空白
#%%
#TF-IDF
"""
TfidfVectorizer方法4个参数含义：
"""
tfidf=TfidfVectorizer(stop_words=stopwords,max_features=5000,lowercase=False)

X = tfidf.fit_transform(cut_context(train_df['Content']))
print('词表大小', len(tfidf.vocabulary_))
print(X.shape)



#%%建立模型
#标签编码

train_df = pd.read_csv('/home/jovyan/data/7-text_classification/cnews.train.txt', sep='\t', header=None)

labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(train_df[0])#一旦给train_df加上columns，就无法使用[0]来获取第一列了
y.shape

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
model = MultinomialNB(alpha=0.2)#参数自己选 当然也可以不特殊设置
model.fit(train_X, train_y)
model.score(test_X, test_y)


cv_split = ShuffleSplit(n_splits=5, test_size=0.3)
score_ndarray = cross_val_score(logistic_model, X, y, cv=cv_split)
print(score_ndarray)
print(score_ndarray.mean())



#%%评估
#混淆矩阵

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)
logistic_model = LogisticRegressionCV(multi_class='multinomial', solver='lbfgs')
logistic_model.fit(train_X, train_y)
predict_y = logistic_model.predict(test_X)

pd.DataFrame(confusion_matrix(test_y, predict_y),columns=labelEncoder.classes_, index=labelEncoder.classes_)

#绘制precision、recall、f1-score、support报告表：
def eval_model(y_true, y_pred, labels):
    #计算每个分类的Precision, Recall, f1, support
    p, r, f1, s = precision_recall_fscore_support( y_true, y_pred)
    #计算总体的平均Precision, Recall, f1, support
    tot_p = np.average(p, weights=s)
    tot_r = np.average(r, weights=s)
    tot_f1 = np.average(f1, weights=s)
    tot_s = np.sum(s)
    res1 = pd.DataFrame({
        u'Label': labels,
        u'Precision' : p,
        u'Recall' : r,
        u'F1' : f1,
        u'Support' : s
    })

    res2 = pd.DataFrame({
        u'Label' : ['总体'],
        u'Precision' : [tot_p],
        u'Recall': [tot_r],
        u'F1' : [tot_f1],
        u'Support' : [tot_s]
    })

    res2.index = [999]
    res = pd.concat( [res1, res2])
    return res[ ['Label', 'Precision', 'Recall', 'F1', 'Support'] ]

predict_y = model.predict(test_X)
eval_model(test_y, predict_y, labelEncoder.classes_)
#%%测试
test_df = pd.read_csv('/home/jovyan/data/7-text_classification/cnews.test.txt', sep='\t', header=None)
test_X = tfidf.transform(cut_context(test_df[1]))
test_y = labelEncoder.transform(test_df[0])
predict_y = model.predict(test_X)
eval_model(test_y, predict_y, labelEncoder.classes_)
