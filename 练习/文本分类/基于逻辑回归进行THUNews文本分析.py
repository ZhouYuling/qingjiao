# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# 从Scikit-learn中导入逻辑回归算法库，补全此处代码
from sklearn.？？？？
# 从Scikit-learn中导入划分数据集的库，补全此处代码
from sklearn.？？？？
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import numpy as np
import jieba

# 使用pandas的read_csv方法读取训练集数据，训练集位于"/home/jovyan/data/7-text_classification"文件夹中，名称为cnews.train.txt，数据以回车分行，补全此处代码
train_df = pd.？？？？
# 使用pandas的read_csv方法读取测试集数据，训练集位于"/home/jovyan/data/7-text_classification"文件夹中，名称为cnews.train.txt，数据以回车分行，补全此处代码
test_df = pd.？？？？

#或者通过columns来查看
train_df.columns = ['Subject', 'Content']
train_df['Subject'].value_counts().sort_index()
print(train_df)

# 分词
def cut_context(data):
   # 采用全模式的方法对数据进行分词，补全此处代码
    words=data.apply(lambda x: ' '.join(？？？？))
    return words

#停词过滤
stopwords=open('/home/jovyan/data/7-text_classification/cnews.vocab.txt',encoding='utf-8')
stopwords_list=stopwords.readlines()
stopworsd=[x.strip() for x in stopwords_list]#去掉每行头尾空白

#使用TfidfVectorizer方法原始文档集合转换为TF-IDF功能矩阵，要求所有停用词需要从转换结果中删除，最大特征词汇表为5000，在转换前不需要将所有字符转换为小写
tfidf=TfidfVectorizer(？？？？)
X = tfidf.fit_transform(cut_context(train_df['Content']))
print('词表大小', len(tfidf.vocabulary_))
print('数据大小',X.shape)


#建立模型
#标签编码

train_df = pd.read_csv('/home/jovyan/data/7-text_classification/cnews.train.txt', sep='\t', header=None)
labelEncoder = LabelEncoder()
y = labelEncoder.fit_transform(train_df[0])#一旦给train_df加上columns，就无法使用[0]来获取第一列了

#逻辑回归
# 使用train_test_split方法将数据集划分为训练集与测试集，按照8：2的比例进行部分，补全此处代码
train_X, test_X, train_y, test_y = ？？？？
# 使用LogisticRegression算法创建逻辑回归模型，多类选项采用multinomial，迭代优化损失函数采用lbfgs算法，补全此处代码
logistic_model = ？？？？
# 使用fit方法训练逻辑回归模型，补全此处代码
logistic_model.？？？？
# 使用score方法对训练后的逻辑回归模型对于测试集进行预测，并得出预测准确率，补全此处代码
logistic_model.？？？？


cv_split = ShuffleSplit(n_splits=5, test_size=0.3)
score_ndarray = cross_val_score(logistic_model, X, y, cv=cv_split)
print(score_ndarray)
print(score_ndarray.mean())


# 评估
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

predict_y = logistic_model.predict(test_X)
eval_model(test_y, predict_y, labelEncoder.classes_)
# 使用pandas的read_csv方法读取验证集数据，训练集位于"/home/jovyan/data/7-text_classification"文件夹中，名称为cnews.val.txt，数据以回车分行，补全此处代码
test_df = pd.？？？？
test_X = tfidf.transform(cut_context(test_df[1]))
test_y = labelEncoder.transform(test_df[0])
# 使用predict方法对验证集进行预测，补全此处代码
predict_y = logistic_model.？？？？
# 调用eval_model方法，输出验证集的报告表，补全此处代码
？？？？
