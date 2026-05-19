import numpy as np
import pandas as pd
import joblib
from bokeh.layouts import column
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from common.functions import sigmoid, softmax

# 读取数据
def get_data():
    # 加载数据
    data = pd.read_csv("../data/train.csv")

    # 划分数据集
    x = data.drop(columns = 'label',axis = 1)
    y = data['label']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

    # 归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_test,  y_test

# 初始化神经网络
def init_network():
    # 从文件中加载训练好的模型
    network = joblib.load('../data/nn_sample')
    return network

# 前向传播
def forward(network,x):
    w1 = network['W1']
    b1 = network['b1']
    w2 = network['W2']
    b2 = network['b2']
    w3 = network['W3']
    b3 = network['b3']
    a1 = x @ w1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ w2 + b2
    z2 = sigmoid(a2)
    a3 = z2 @ w3 + b3

    y = softmax(a3)
    return y

# 主流程
network = init_network()
x_test ,y_test = get_data()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x_test), batch_size):
    x_batch = x_test[i:i+batch_size]
    y_batch = forward(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == y_test[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))