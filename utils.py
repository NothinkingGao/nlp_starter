# -*- coding: utf-8 -*-
import pickle

import jieba
import re

from sklearn.feature_extraction.text import CountVectorizer

import config
import pandas as pd

TRAIN_PATH = f"{config.PROJECT_BASE_DIR}/data/weibo2018/train.txt"
TEST_PATH = f"{config.PROJECT_BASE_DIR}/data/weibo2018/test.txt"
VECTOR_NAME = f"{config.PROJECT_BASE_DIR}/model/CountVectorizer.pkl"


stopwords = []
with open(f"{config.PROJECT_BASE_DIR}\data\stopwords.txt", "r", encoding="utf8") as f:
    for w in f:
        stopwords.append(w.strip())


def load_corpus(path):
    """
    加载语料库
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing(content)
            data.append((content, int(seniment)))
    return data


def load_corpus_bert(path):
    """
    加载语料库
    """
    data = []
    with open(path, "r", encoding="utf8") as f:
        for line in f:
            [_, seniment, content] = line.split(",", 2)
            content = processing_bert(content)
            data.append((content, int(seniment)))
    return data


def processing(text):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是这个数据集中的一个bad case, 不用特别在意
    # 分词
    words = [w for w in jieba.lcut(text) if w.isalpha()]
    # 对否定词`不`做特殊处理: 与其后面的词进行拼接
    while "不" in words:
        index = words.index("不")
        if index == len(words) - 1:
            break
        words[index: index+2] = ["".join(words[index: index+2])]  # 列表切片赋值的酷炫写法
    # 用空格拼接成字符串
    result = " ".join(words)
    return result


def processing_bert(text):
    """
    数据预处理, 可以根据自己的需求进行重载
    """
    # 数据清洗部分
    text = re.sub("\{%.+?%\}", " ", text)           # 去除 {%xxx%} (地理定位, 微博话题等)
    text = re.sub("@.+?( |$)", " ", text)           # 去除 @xxx (用户名)
    text = re.sub("【.+?】", " ", text)              # 去除 【xx】 (里面的内容通常都不是用户自己写的)
    text = re.sub("\u200b", " ", text)              # '\u200b'是这个数据集中的一个bad case, 不用特别在意
    return text

def load_data():
    """
    加载数据
    """
    # 分别加载训练集和测试集
    train_data = load_corpus(TRAIN_PATH)
    test_data = load_corpus(TEST_PATH)

    df_train = pd.DataFrame(train_data, columns=["words", "label"])
    df_test = pd.DataFrame(test_data, columns=["words", "label"])

    vectorizer = CountVectorizer(token_pattern='\[?\w+\]?',
                                 stop_words=stopwords)

    # 获取训练集和测试集的特征向量和标签
    X_train = vectorizer.fit_transform(df_train["words"])
    Y_train = df_train["label"]

    X_test = vectorizer.transform(df_test["words"])
    Y_test = df_test["label"]

    with open(VECTOR_NAME, 'wb') as fw:
        pickle.dump(vectorizer.vocabulary_, fw)

    return X_train, Y_train, X_test, Y_test

def load_vectorizer():
    vectorizer = CountVectorizer(vocabulary=pickle.load(open(VECTOR_NAME,'rb')))
    return vectorizer