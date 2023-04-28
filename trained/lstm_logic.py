# import sys
# sys.path.append("..")
import os
import pickle

import numpy
import torch
from sklearn.metrics import confusion_matrix

import config
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from utils import processing
from utils import load_corpus, stopwords
import pandas as pd
import seaborn as sn
from gensim import models
from logger_config import logger
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Using {} device".format(device))



def get_words2vec(df_train = None):
    # Word2Vec
    words_path = f"{config.PROJECT_BASE_DIR}/trained/word2vec.model"
    if os.path.exists(words_path):
        word2vec = models.Word2Vec.load(words_path)
    else:
        # word2vec要求的输入格式: list(word)
        wv_input = df_train['text'].map(lambda s: s.split(" "))   # [for w in s.split(" ") if w not in stopwords]
        wv_input.head()
        word2vec = models.Word2Vec(wv_input,
                                   vector_size=64,   # 词向量维度
                                   min_count=1,      # 最小词频, 因为数据量较小, 这里卡1
                                   epochs=1000)      # 迭代轮次
        word2vec.save(words_path)
    return word2vec


# 数据集
class MyLstmDataset(Dataset):
    def __init__(self, df_train):
        self.data = []
        self.label = df_train["label"].tolist()

        word2vec = get_words2vec(df_train = df_train)
        for s in df_train["text"].tolist():
            vectors = []
            for w in s.split(" "):
                if w in word2vec.wv.key_to_index:
                    vectors.append(word2vec.wv[w])  # 将每个词替换为对应的词向量
            vectors = torch.Tensor(numpy.array(vectors))
            self.data.append(vectors)

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


def collate_fn(data):
    """
    :param data: 第0维：data，第1维：label
    :return: 序列化的data、记录实际长度的序列、以及label列表
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)  # pack_padded_sequence要求要按照序列的长度倒序排列
    data_length = [len(sq[0]) for sq in data]
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    data = pad_sequence(x, batch_first=True, padding_value=0)  # 用RNN处理变长序列的必要操作
    return data, torch.tensor(y, dtype=torch.float32), data_length



# 网络结构
class LSTM(nn.Module):
    def __init__(self, input_size = 64, hidden_size = 64, num_layers= 2):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # 双向, 输出维度要*2
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, lengths):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)  # 双向, 第一个维度要*2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        packed_input = torch.nn.utils.rnn.pack_padded_sequence(input=x, lengths=lengths, batch_first=True)
        packed_out, (h_n, h_c) = self.lstm(packed_input, (h0, c0))

        lstm_out = torch.cat([h_n[-2], h_n[-1]], 1)  # 双向, 所以要将最后两维拼接, 得到的就是最后一个time step的输出
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out

class LSTM_Logic(object):
    def __init__(self):
        super().__init__()
        self.model_path = f"{config.PROJECT_BASE_DIR}/model/lstm_model.pkl"
        self.image =f"{config.PROJECT_BASE_DIR}/trained/lstm.png"
        self.location = f"{config.PROJECT_BASE_DIR}/trained/lstm.json"
        self.score = 0.0

    # 序列化对象并保存到本地
    def save(self):
        with open(self.location, 'wb') as f:
            pickle.dump(self, f)

    def load(self):
        with open(self.location, 'rb') as f:
            return pickle.load(f)

    # 在测试集效果检验
    def test(self):

        TEST_PATH = f"{config.PROJECT_BASE_DIR}/data/weibo2018/test.txt"
        test_data = load_corpus(TEST_PATH)
        df_test = pd.DataFrame(test_data, columns=["text", "label"])



        # 定义lstm模型
        lstm = LSTM().to(device)

        batch_size = 100
        # 测试集
        test_data = MyLstmDataset(df_test)
        test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

        y_pred, y_true = [], []

        with torch.no_grad():
            for x, labels, lengths in test_loader:
                x = x.to(device)
                outputs = lstm(x, lengths)  # 前向传播
                outputs = outputs.view(-1)  # 将输出展平
                y_pred.append(outputs.cpu())
                y_true.append(labels)

        y_prob = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        y_pred = y_prob.clone()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        # dispaly y_pred, y_true by heatmap
        heatmap = sn.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='.20g')
        heatmap.get_figure().savefig(f"{config.PROJECT_BASE_DIR}/trained/lstm.png")

        print(metrics.classification_report(y_true, y_pred))

        self.score = metrics.accuracy_score(y_true, y_pred)
        print(f"准确率:{self.score}")
        print("AUC:", metrics.roc_auc_score(y_true, y_prob))

        self.save()


    def train(self):

        TRAIN_PATH = f"{config.PROJECT_BASE_DIR}/data/weibo2018/train.txt"

        # 分别加载训练集和测试集
        train_data = load_corpus(TRAIN_PATH)
        df_train = pd.DataFrame(train_data, columns=["text", "label"])

        # 超参数
        learning_rate = 5e-4

        num_epoches = 5

        batch_size = 100

        # 定义lstm模型
        lstm = LSTM()
        # 将模型移动到GPU
        lstm = lstm.to(device)


        # 训练集
        logger.info("开始加载训练集...")

        train_data = MyLstmDataset(df_train)


        train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        logger.info("训练集加载完成")



        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

        # 迭代训练
        for epoch in range(num_epoches):
            total_loss = 0
            for i, (inputs, labels, lengths) in enumerate(train_loader):
                # convert inputs to gpu
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 输出展平和前向传播
                logits = lstm(inputs, lengths)
                logits = logits.view(-1)

                loss = criterion(logits, labels)  # loss计算
                total_loss += loss
                optimizer.zero_grad()  # 梯度清零
                loss.backward(retain_graph=True)  # 反向传播，计算梯度
                optimizer.step()  # 梯度更新
                if (i + 1) % 10 == 0:
                    print("epoch:{}, step:{}, loss:{}".format(epoch + 1, i + 1, total_loss / 10))
                    total_loss = 0
        # save model
        torch.save(lstm, self.model_path)
        print("saved model: ", self.model_path)


    def predict(self,strs):
        # 这里看上去好像没什么用
        net = torch.load(f"{config.PROJECT_BASE_DIR}/model/lstm_5.model") # 加载模型

        # 定义lstm模型
        word2vec = get_words2vec()
        
        data = []
        for s in strs:
            vectors = []
            for w in processing(s).split(" "):
                if w in word2vec.wv.key_to_index:
                    vectors.append(word2vec.wv[w])   # 将每个词替换为对应的词向量
            vectors = torch.Tensor(vectors)
            data.append(vectors)
        x, _, lengths = collate_fn(list(zip(data, [-1] * len(strs))))
        with torch.no_grad():
            x = x.to(device)
            outputs = net(x, lengths)          # 前向传播
            outputs = outputs.view(-1)          # 将输出展平


        outputs = outputs.cpu().detach().numpy()
        return outputs


if __name__ == '__main__':
    lstm = LSTM_Logic()
    #lstm.train()
    #lstm.test()
    strs = ["我想说我会爱你多一点点", "日有所思梦感伤"]
    print(lstm.predict(strs))