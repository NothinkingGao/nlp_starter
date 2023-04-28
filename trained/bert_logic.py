# import sys
# sys.path.append("..")
import os
import pickle

import numpy
import torch
from sklearn.metrics import confusion_matrix
from transformers import BertTokenizer, BertModel
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

TRAIN_PATH = f"{config.PROJECT_BASE_DIR}/data/weibo2018/train.txt"
TEST_PATH = f"{config.PROJECT_BASE_DIR}/data/weibo2018/test.txt"


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"    # 在我的电脑上不加这一句, bert模型会报错
MODEL_PATH = f"{config.PROJECT_BASE_DIR}/model/chinese_wwm_pytorch"     # 下载地址见 https://github.com/ymcui/Chinese-BERT-wwm


# 加载
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)   # 分词器
bert = BertModel.from_pretrained(MODEL_PATH)           # 模型
bert = bert.to(device)

# 数据集
class MyDataset(Dataset):
    def __init__(self, df):
        self.data = df["text"].tolist()
        self.label = df["label"].tolist()

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.label)


# 网络结构
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc(x)
        out = self.sigmoid(out)
        return out


class BERT_Logic(object):
    def __init__(self):
        super().__init__()
        self.model_path = f"{config.PROJECT_BASE_DIR}/model/bert_model.pkl"
        self.image =f"{config.PROJECT_BASE_DIR}/trained/bert.png"
        self.location = f"{config.PROJECT_BASE_DIR}/trained/bert.json"
        self.score = 0.0

    # 序列化对象并保存到本地
    def save(self):
        with open(self.location, 'wb') as f:
            pickle.dump(self, f)

    def load(self):
        with open(self.location, 'rb') as f:
            return pickle.load(f)
    # 测试集效果检验
    def test(self):
        # 超参数

        input_size = 768
        batch_size = 100

        # 分别加载训练集和测试集
        test_data = load_corpus(TEST_PATH)
        df_test = pd.DataFrame(test_data, columns=["text", "label"])
        # 测试集
        test_data = MyDataset(df_test)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

        net = torch.load(f"{config.PROJECT_BASE_DIR}/model/bert_dnn_8.model")

        y_pred, y_true = [], []
        with torch.no_grad():
            for words, labels in test_loader:
                tokens = tokenizer(words, padding=True)
                input_ids = torch.tensor(tokens["input_ids"]).to(device)
                attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
                last_hidden_states = bert(input_ids, attention_mask=attention_mask)
                bert_output = last_hidden_states[0][:, 0]
                outputs = net(bert_output)  # 前向传播
                outputs = outputs.view(-1)  # 将输出展平
                y_pred.append(outputs.cpu())
                y_true.append(labels)

        y_prob = torch.cat(y_pred)
        y_true = torch.cat(y_true)
        y_pred = y_prob.clone()
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0

        heatmap = sn.heatmap(confusion_matrix(y_true, y_pred), annot=True, fmt='.20g')
        heatmap.get_figure().savefig(f"{config.PROJECT_BASE_DIR}/trained/bert.png")

        print(metrics.classification_report(y_true, y_pred))
        self.score = metrics.accuracy_score(y_true, y_pred)
        print(f"准确率:{self.score}")
        print("AUC:", metrics.roc_auc_score(y_true, y_prob))
        self.save()

    def train(self):
        # 超参数
        learning_rate = 1e-3
        input_size = 768
        num_epoches = 10
        batch_size = 100
        decay_rate = 0.9
        train_data = load_corpus(TRAIN_PATH)

        # 训练集
        df_train = pd.DataFrame(train_data, columns=["text", "label"])
        train_data = MyDataset(df_train)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

        net = Net(input_size).to(device)

        # 定义损失函数和优化器
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=decay_rate)

        # 迭代训练
        for epoch in range(num_epoches):
            total_loss = 0
            for i, (words, labels) in enumerate(train_loader):
                tokens = tokenizer(words, padding=True)
                input_ids = torch.tensor(tokens["input_ids"]).to(device)
                attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
                labels = labels.float().to(device)
                with torch.no_grad():
                    last_hidden_states = bert(input_ids, attention_mask=attention_mask)
                    bert_output = last_hidden_states[0][:, 0]
                optimizer.zero_grad()  # 梯度清零
                outputs = net(bert_output)  # 前向传播
                logits = outputs.view(-1)  # 将输出展平
                loss = criterion(logits, labels)  # loss计算
                total_loss += loss
                loss.backward()  # 反向传播，计算梯度
                optimizer.step()  # 梯度更新
                if (i + 1) % 10 == 0:
                    print("epoch:{}, step:{}, loss:{}".format(epoch + 1, i + 1, total_loss / 10))
                    total_loss = 0

            # learning_rate decay
            scheduler.step()

            # save model
            model_path = f"{config.PROJECT_BASE_DIR}/model/bert_dnn_{epoch + 1}.model"
            torch.save(net, model_path)
            print("saved model: ", model_path)


    def predict(self,strings):
        net = torch.load(f"{config.PROJECT_BASE_DIR}/model/bert_dnn_8.model")  # 训练过程中的巅峰时刻
        tokens = tokenizer(strings, padding=True)
        input_ids = torch.tensor(tokens["input_ids"]).to(device)
        attention_mask = torch.tensor(tokens["attention_mask"]).to(device)
        last_hidden_states = bert(input_ids, attention_mask=attention_mask)
        bert_output = last_hidden_states[0][:, 0]
        outputs = net(bert_output)
        outputs = outputs.cpu().detach().numpy()
        return outputs[0]


if __name__ == '__main__':
    bert_logic = BERT_Logic()
    #bert_logic.train()
    bert_logic.test()
    bert_logic.predict(["我爱你"])

