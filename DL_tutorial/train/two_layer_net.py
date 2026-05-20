import numpy as np
from common.functions import sigmoid,softmax,cross_entropy_error
from common.gradient import numerical_gradient
class TwoLayerNet():
    # 初始化方法
    def __init__(self,n_input,n_hidden,n_output, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = np.random.randn(n_input,n_hidden) * weight_init_std
        self.params['b1'] = np.zeros(n_hidden)
        self.params['W2'] = np.random.randn(n_hidden,n_output) * weight_init_std
        self.params['b2'] = np.zeros(n_output)
    # 前向传播
    def forward(self,x):
        w1,w2 = self.params['W1'],self.params['W2']
        b1,b2 = self.params['b1'],self.params['b2']
        a1 = np.dot(x,w1)+b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1,w2)+b2
        y = softmax(a2)
        return y
    # 计算损失
    def loss(self, x,t):
        y = self.forward(x)
        return cross_entropy_error(y,t)
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
