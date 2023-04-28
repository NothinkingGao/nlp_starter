# 贝叶斯网络的训练和保存
# 以下是使用sklearn的MultinomialNB模型训练并保存的示例代码：
# 导入所需的库和模块
# 在代码中，我们首先加载了20 Newsgroups数据集，然后使用CountVectorizer提取文本特征，并使用train_test_split将数据集分成训练集和测试集。
# 接下来，我们使用MultinomialNB模型训练了我们的分类器，然后使用score方法在测试集上评估了模型的性能。
# 最后，我们使用pickle库将训练好的模型保存到‘MultinomialNB_model.pkl’文件中。
# 需要注意的是，为了在将来使用保存的模型对新数据进行分类，我们需要用同样的方法对新数据进行特征提取。
import sys
import json
from PyQt5 import QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

import utils

sys.path.append("..")
import config
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
from utils import load_corpus, stopwords, processing
from sklearn.metrics import confusion_matrix
import seaborn as sn

class MultinomialNBLogic(object):
    def __init__(self):
        super().__init__()
        self.MODEL_NAME = f"{config.PROJECT_BASE_DIR}/model/MultinomialNB_model.pkl"
        self.image =f"{config.PROJECT_BASE_DIR}/trained/multinomial_nb.jpg"
        self.location = f"{config.PROJECT_BASE_DIR}/trained/multinomial_nb.json"
        self.score = 0.0

    def train_and_save(self):
        # 分别加载训练集和测试集
        X_train, Y_train, X_test, Y_test = utils.load_data()

        # 训练模型
        clf = MultinomialNB()
        clf.fit(X_train, Y_train)

        y_pre_linear = clf.predict(X_test)

        heatmap = sn.heatmap(confusion_matrix(y_pre_linear, Y_test), annot=True)
        # save heatmap to file
        heatmap.get_figure().savefig(self.image)

        # 评估模型
        score = clf.score(X_test, Y_test)

        self.score = score
        print('测试集准确率：%.3f' % score)

        # 保存模型
        with open(self.MODEL_NAME, 'wb') as f:
            pickle.dump(clf, f)

    # 序列化对象并保存到本地
    def save(self):
        with open(self.location, 'wb') as f:
            pickle.dump(self, f)


    def load(self):
        with open(self.location, 'rb') as f:
            return pickle.load(f)

    # load model and predict new data
    def predict(self,predict_data):
        # 加载贝叶斯模型
        with open(self.MODEL_NAME, 'rb') as f:
            clf = pickle.load(f)

        # 加载词袋模型
        vectorizer = utils.load_vectorizer()

        # 预测新数据
        words = [processing(s) for s in predict_data]
        vec = vectorizer.transform(words)
        predict_result = clf.predict(vec)
        print("预测结果：", predict_result)
        return predict_result


def test_train_and_save():
    nb = MultinomialNBLogic()
    nb.train_and_save()
    nb.save()

def test_predict():
    new_data = ["终于收获一个最好消息", "哭了, 今天怎么这么倒霉", "今天天气真好"]
    nb =  MultinomialNBLogic()
    nb.predict(new_data)

if __name__ == '__main__':
    test_train_and_save()
    test_predict()
