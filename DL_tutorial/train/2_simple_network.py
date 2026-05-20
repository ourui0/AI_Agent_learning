import numpy as np
from common.functions import softmax,cross_entropy_error
from common.gradient import numerical_gradient


# 定义一个单层简单网络类
class SimpleNet:
    # 初始化
    def __init__(self):
        self.W = np.random.randn(2,3)
        self.b = np.random.randn(3)
    # 前向传播
    def forward(self,x):
        a = np.dot(x,self.W) + self.b
        y = softmax(a)
        return y

    # 计算损失
    def loss(self,y,t):
        y = self.forward(x)
        loss_value = cross_entropy_error(y,t)
        return loss_value

if __name__=="__main__":
    x = np.array([0.6,0.9])
    t = np.array([0,0,1])
    net = SimpleNet()
    def loss_f(w):
        return net.loss(x, t)
    dw = numerical_gradient(loss_f,net.W)
    print(dw)

