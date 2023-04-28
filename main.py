import sys
import json
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QTextEdit, \
    QLineEdit, QMainWindow, QDialog
from PyQt5.QtGui import QPixmap

from trained.bert_logic import BERT_Logic,Net,MyDataset
from trained.lstm_logic import LSTM_Logic,MyLstmDataset,LSTM
from trained.multinomial_nb import MultinomialNBLogic
from trained.svm_logic import SVMLogic
from trained.xgboost_logic import XgbboostLogic

class SentimentAnalysisWidget(QWidget):
    def __init__(self):
        super().__init__()

        # 设置窗口标题和大小
        self.setWindowTitle('中文文本情感分类')
        self.resize(500, 600)

        # 创建垂直布局
        vbox = QVBoxLayout()
        vbox.setSpacing(5)

        # 添加标题标签
        title_label = QLabel('中文文本情感分类')
        title_label.setStyleSheet('font-size: 24px; font-weight: bold;')
        vbox.addWidget(title_label)

        # 添加方向标签
        direction_label = QLabel('方向：')
        direction_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        hbox = QHBoxLayout()
        hbox.addWidget(direction_label)
        hbox.addWidget(QLabel('机器学习'))
        hbox.addStretch()
        vbox.addLayout(hbox)

        # 添加数据集选项
        # dataset_label = QLabel('数据集：')
        # dataset_combo = QComboBox()
        # dataset_combo.addItems(['数据集1', '数据集2', '数据集3'])
        # dataset_hbox = QHBoxLayout()
        # dataset_hbox.addWidget(dataset_label)
        # dataset_hbox.addWidget(dataset_combo)
        # dataset_hbox.addStretch()
        # vbox.addLayout(dataset_hbox)

        # 输入分析文本
        input_text_label = QLabel('输入分析文本：')

        input_text_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        vbox.addWidget(input_text_label)
        self.input_text_edit = QTextEdit()
        # 设置输入文本高度
        self.input_text_edit.setFixedHeight(100)
        vbox.addWidget(self.input_text_edit)

        # 添加算法选项
        self.algorithm_label = QLabel('算法：')
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(['贝叶斯', 'SVM', 'XGboost', 'LSTM', 'bert'])

        # add a buttong to pop up a window to dispaly the algorithm
        button = QPushButton('测试结果')
        button.clicked.connect(self.show_result)
        algorithm_hbox = QHBoxLayout()
        algorithm_hbox.addWidget(self.algorithm_label)
        algorithm_hbox.addWidget(self.algorithm_combo)
        algorithm_hbox.addWidget(button)
        algorithm_hbox.addStretch()
        vbox.addLayout(algorithm_hbox)

        # 添加运行按钮
        run_button = QPushButton('分析')
        run_button.setStyleSheet('font-size: 16px; font-weight: bold;')
        run_button.clicked.connect(self.run_algorithm)
        vbox.addWidget(run_button)

        # 添加运行结果
        # running_result_label = QLabel('运行结果：')
        # running_result_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        # running_result_hbox = QHBoxLayout()
        # running_result_hbox.addWidget(running_result_label)
        # self.running_result_box = QLineEdit('')
        # running_result_hbox.addWidget(self.running_result_box)
        # running_result_hbox.addStretch()
        # vbox.addLayout(running_result_hbox)


        # 添加情感分析结果标签
        emotion_label = QLabel('情感分析结果：')
        emotion_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        emotion_layout = QHBoxLayout()
        emotion_layout.addWidget(emotion_label)
        self.emotion_result = QLabel('')
        emotion_layout.addWidget(self.emotion_result)
        emotion_layout.addStretch()
        vbox.addLayout(emotion_layout)
        vbox.addStretch()

        # 添加文本框
        # result_textedit = QTextEdit('运行效果')
        # result_textedit.setReadOnly(True)
        # result_textedit.setStyleSheet('font-size: 14px;')
        # vbox.addWidget(result_textedit)
        #
        # # 添加显示图像的标签
        # self.image_label = QLabel()
        # vbox.addWidget(self.image_label)

        # 设置窗口布局
        self.setLayout(vbox)

    def show_result(self):
        algorithme = self.algorithm_combo.currentText()
        # popup AlgorithmWindow
        algorithm_window = AlgorithmWindow(algorithme)
        algorithm_window.show()
        algorithm_window.exec_()

    def run_algorithm(self):
        text = self.input_text_edit.toPlainText()
        words = text.split(",")

        algorithme = self.algorithm_combo.currentText()
        if algorithme == '贝叶斯':
            try:
                nb = MultinomialNBLogic()
                predict_result = nb.predict(words)
                self.emotion_result.setText(str(predict_result))
            except Exception as e:
                print(e)
        elif algorithme == 'SVM':
            svm_logic = SVMLogic()
            predict_result = svm_logic.predict(words)
            self.emotion_result.setText(str(predict_result))
        elif algorithme == 'XGboost':
            try:
                xgboost_logic = XgbboostLogic()
                predict_result = xgboost_logic.predict(words)
                self.emotion_result.setText(str(predict_result))
            except Exception as e:
                print(e)
        elif algorithme == 'LSTM':
            try:
                lstm_logic = LSTM_Logic()
                predict_result = lstm_logic.predict(words)
                self.emotion_result.setText(str(predict_result))
            except Exception as e:
                print(e)
        elif algorithme == 'bert':
            try:
                print("use bert to predict...")
                bert_logic = BERT_Logic()
                predict_result = bert_logic.predict(words)
                print(predict_result)
                self.emotion_result.setText(str(predict_result))
            except Exception as e:
                print(e)

# create new window to display the algorithm,it has two line,one is the test result,another is the image
class AlgorithmWindow(QDialog):
    def __init__(self, algorithme = '贝叶斯'):
        super().__init__()
        self.algorithme = algorithme
        self.setWindowTitle('算法测试结果')
        self.resize(500, 600)

        print(self.algorithme)

        vbox = QVBoxLayout()
        vbox.setSpacing(5)

        # 添加标题标签
        title_label = QLabel(f'{self.algorithme}测试结果')
        title_label.setStyleSheet('font-size: 24px; font-weight: bold;')
        vbox.addWidget(title_label)

        # 添加运行结果
        running_result_label = QLabel('运行结果：')
        running_result_label.setStyleSheet('font-size: 16px; font-weight: bold;')
        running_result_hbox = QHBoxLayout()
        running_result_hbox.addWidget(running_result_label)

        self.running_result_box = QLineEdit('')
        running_result_hbox.addWidget(self.running_result_box)
        running_result_hbox.addStretch()
        vbox.addLayout(running_result_hbox)

        # 添加显示图像的标签
        self.image_label = QLabel()
        vbox.addWidget(self.image_label)
        vbox.addStretch()

        # 设置窗口布局
        self.setLayout(vbox)

        # show the result
        self.show_result()

    def show_result(self):
        if self.algorithme == '贝叶斯':
            nb = MultinomialNBLogic()
            nb = nb.load()
            try:
                self.running_result_box.setText(str(nb.score))
                print(nb.image)
                pixmap = QPixmap(nb.image)  # 这里需要指定图片路径
                self.image_label.setPixmap(pixmap)
            except Exception as e:
                print(e)
        elif self.algorithme == 'SVM':
            svm_logic = SVMLogic()
            svm_logic = svm_logic.load()
            try:
                self.running_result_box.setText(str(svm_logic.score))
                print(svm_logic.image)
                pixmap = QPixmap(svm_logic.image)  # 这里需要指定图片路径
                self.image_label.setPixmap(pixmap)
            except Exception as e:
                print(e)
        elif self.algorithme == 'XGboost':
            xgboost_logic = XgbboostLogic()
            xgboost_logic = xgboost_logic.load()
            try:
                self.running_result_box.setText(str(xgboost_logic.score))
                print(xgboost_logic.image)
                pixmap = QPixmap(xgboost_logic.image)  # 这里需要指定图片路径
                self.image_label.setPixmap(pixmap)
            except Exception as e:
                print(e)
        elif self.algorithme == 'LSTM':
            lstm_logic = LSTM_Logic()
            lstm_logic = lstm_logic.load()
            try:
                self.running_result_box.setText(str(lstm_logic.score))
                print(lstm_logic.image)
                pixmap = QPixmap(lstm_logic.image)  # 这里需要指定图片路径
                self.image_label.setPixmap(pixmap)
            except Exception as e:
                print(e)
        elif self.algorithme == 'bert':
            bert_logic = BERT_Logic()
            bert_logic = bert_logic.load()
            try:
                self.running_result_box.setText(str(bert_logic.score))
                print(bert_logic.image)
                pixmap = QPixmap(bert_logic.image)  # 这里需要指定图片路径
                self.image_label.setPixmap(pixmap)
            except Exception as e:
                print(e)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = SentimentAnalysisWidget()
    widget.show()
    sys.exit(app.exec_())