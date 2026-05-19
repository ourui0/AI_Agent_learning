import numpy as np
from numpy import identity

from DL_tutorial.common.functions import sigmoid, identity_function


# 初始化网络
def init_network():
    network = {}
    # 第一层参数
    network['w1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    # 第二层参数
    network['w2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    # 第三层参数
    network['w3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network
# 前向传播
def forward(network, x):
    w1 = network['w1']
    b1 = network['b1']
    w2 = network['w2']
    b2 = network['b2']
    w3 = network['w3']
    b3 = network['b3']
    a1 = x @ w1 + b1
    z1 = sigmoid(a1)
    a2 = z1 @ w2 + b2
    z2 = sigmoid(a2)
    a3 = z2 @ w3 + b3

    y = identity_function(a3)
    return y

# 主流程

x = np.array([1.0,0.5])

# 定义神经网络
network = init_network()

# 前向传播
y = forward(network, x)

print(y)