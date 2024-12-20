import urllib.request  # 下载文件
import os
import tarfile   # 解压缩文件
import re
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer   # 建立字典
from keras.preprocessing import sequence  # 截长补短
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Embedding, LSTM


def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')  # 剔除掉html标签
    return re_tag.sub('', text)


def read_file(filetype):  # 读取文件
    path = "./aclImdb/"
    file_list = []

    positive_path = path + filetype + '/pos/'   # 正面评价的文件路径
    for f in os.listdir(positive_path):
        file_list += [positive_path + f]   # 存储到文件列表中

    negative_path = path + filetype + '/neg/'   # 负面评价的文件路径
    for f in os.listdir(negative_path):
        file_list += [negative_path + f]

    print(f'读取 {filetype} 文件: {len(file_list)}')   # 打印文件个数

    all_labels = ([1] * 12500 + [0] * 12500)  # 前12500是正面都为1;后12500是负面都为0
    all_texts = []

    for fi in file_list:  # 读取所有文件
        with open(fi, encoding='utf8') as file_input:
            # 先读取文件,使用join连接所有字符串,然后使用rm_tags剔除tag最后存入列表all_texts
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    return all_labels, all_texts


y_train, train_text = read_file("train")
y_test, train_test = read_file("test")

y_train = np.array(y_train)
y_test = np.array(y_test)
test_text = train_test


# 建立 token
token = Tokenizer(num_words=2000)  # 词典的单词数为2000
# 建立token词典
token.fit_on_texts(train_text)  # 按单词出现次数排序 取前2000个

# 将影评文字转化为数字列表（一条影评文字转化为一条数字列表）
x_train_seq = token.texts_to_sequences(train_text)
x_test_seq = token.texts_to_sequences(test_text)

# 截长补短操作
x_train = sequence.pad_sequences(x_train_seq, maxlen=380)
x_test = sequence.pad_sequences(x_test_seq, maxlen=380)

# LSTM 模型
model = Sequential()
model.add(Embedding(output_dim=32, input_dim=3800, input_length=380))
# 一个单词用32维词向量表示;字典词数(维数)为3800;每个数字列表有100个数字，相当于用100个数字去表示一条评论
model.add(Dropout(0.2))
model.add(LSTM(32))  # LSTM
model.add(Dense(units=256, activation='relu'))  # 神经元节点数为256，激活函数为relu
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # 输出1表示正面评价,0表示负面评价,激活函数为sigmoid
model.summary()  # 模型汇总


# 配置
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])  # 定义损失函数、优化器以及评估
# 训练
train_history = model.fit(x=x_train, y=y_train, validation_split=0.2, epochs=10, batch_size=300, verbose=1)
# 训练10个epoch，每一批次训练300项数据


# 展示训练结果
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


show_train_history(train_history, 'accuracy', 'val_accuracy')   # 准确率折线图
show_train_history(train_history, 'loss', 'val_loss')   # 损失函数折线图

scores = model.evaluate(x_test, y_test)  # 评估
print(f'测试集损失: {scores[0]}')
print(f'测试集准确率: {scores[1]}')
