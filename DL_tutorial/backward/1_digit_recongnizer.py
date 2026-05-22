import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet
from common.data import get_data

x_test,y_test,x_train, y_train = get_data()
print(x_test.shape, y_test.shape, x_train.shape, y_train.shape)

# 定义模型
network = TwoLayerNet(n_input=784,n_hidden=50,n_output=10)

# 设置超参数
lr = 0.1
batch_size = 100
num_epochs = 10
# 计算总迭代次数
iter_per_epoch = np.ceil(x_train.shape[0] // batch_size)
iter_num = int(iter_per_epoch * num_epochs)

# 用梯度下降法训练模型
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

# 循环迭代，用梯度下降法训练模型
for i in range(iter_num):
    # 随机选择批数据
    batch_index = np.random.choice(x_train.shape[0],batch_size)
    x_batch = x_train[batch_index]
    y_batch = y_train[batch_index]

    # grads = network.numerical_gradient(x_batch,y_batch)
    grads = network.gradient(x_batch,y_batch)
    for key in network.params.keys():
        network.params[key] -= lr * grads[key]
    train_loss_list.append(network.loss(x_batch,y_batch))

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train,y_train)
        test_acc = network.accuracy(x_test,y_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(network.loss(x_batch, y_batch))
        print(train_acc, test_acc)

# 画图
plt.plot(train_acc_list,label='train_acc')
plt.plot(test_acc_list,label='test_acc')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('acc')
plt.show()