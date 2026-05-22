from collections import OrderedDict

import numpy as np
from common.functions import sigmoid,softmax,cross_entropy_error
from common.gradient import numerical_gradient
from common.layers import Affine, ReLU, SoftmaxWithLoss


class TwoLayerNet():
    # 初始化方法
    def __init__(self,n_input,n_hidden,n_output, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(n_input,n_hidden) * weight_init_std
        self.params['b1'] = np.zeros(n_hidden)
        self.params['W2'] = np.random.randn(n_hidden,n_output) * weight_init_std
        self.params['b2'] = np.zeros(n_output)

        # 构建神经网络的层
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'],self.params['b1'])
        self.layers['ReLU1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'],self.params['b2'])
        self.last_layer = SoftmaxWithLoss()
    # 前向传播
    def forward(self,x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    # 计算损失
    def loss(self, x,t):
        y = self.forward(x)
        return self.last_layer.forward(y,t)
    # 预测准确率
    def accuracy(self,x,t):
        y = self.forward(x)
        y_pred = np.argmax(y,axis=1)
        acc = np.sum(y_pred==t) / x.shape[0]
        return acc
    def numerical_gradient(self, x, t):
        grads = {}
        def loss_f(w):
            return self.loss(x,t)
        grads['W1'] = numerical_gradient(loss_f,self.params['W1'])
        grads['b1'] = numerical_gradient(loss_f,self.params['b1'])
        grads['W2'] = numerical_gradient(loss_f,self.params['W2'])
        grads['b2'] = numerical_gradient(loss_f,self.params['b2'])
        return grads

    # 反向传播法计算梯度
    def gradient(self, x, t):
        # 前向传播
        self.loss(x,t)
        # 反向传播
        layers = list(self.layers.values())
        layers.reverse()
        dy = self.last_layer.backward()
        for layer in layers:
            dy = layer.backward(dy)
        # 提取全连接层的梯度
        grads = {}
        grads['W1'],grads['b1'] = self.layers['Affine1'].dW,self.layers['Affine1'].db
        grads['W2'],grads['b2'] = self.layers['Affine2'].dW,self.layers['Affine2'].db
        return grads