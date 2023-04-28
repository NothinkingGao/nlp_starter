"""

author： kai
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import load_corpus, stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
class datas(object):
    """
    定义的这个类别用来导入我们的数据
    读取样本中的数据/所有的数据从这文件夹中导入
    """
    def __init__(self):
        self.data_standard = 1
        pass

    def mini_data(self):
        # 读取数据文件
        TRAIN_PATH = "code/data/weibo2018/train.txt"
        TEST_PATH = "code/data/weibo2018/test.txt"
        # 分别加载训练集和测试集
        train_data = load_corpus(TRAIN_PATH)
        test_data = load_corpus(TEST_PATH)
        df_train = pd.DataFrame(train_data, columns=["words", "label"])
        df_test = pd.DataFrame(test_data, columns=["words", "label"])
        df_train.head()
        vectorizer = CountVectorizer(token_pattern='\[?\w+\]?', 
                                    stop_words=stopwords)
        X_train = vectorizer.fit_transform(df_train["words"])
        Y_train = df_train["label"]
        X_test = vectorizer.transform(df_test["words"])
        Y_test = df_test["label"]
    
        return X_train, X_test, Y_train, Y_test
    def dataload_nn(self):
        TRAIN_PATH = "./data/weibo2018/train.txt"
        TEST_PATH = "./data/weibo2018/test.txt"
        # 分别加载训练集和测试集
        train_data = load_corpus(TRAIN_PATH)
        test_data = load_corpus(TEST_PATH)
        df_train = pd.DataFrame(train_data, columns=["text", "label"])
        df_test = pd.DataFrame(test_data, columns=["text", "label"])
        df_train.head()
        wv_input = df_train['text'].map(lambda s: s.split(" "))   # [for w in s.split(" ") if w not in stopwords]
        wv_input.head()      


