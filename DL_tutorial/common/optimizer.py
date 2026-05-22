import numpy as np

# SGD
class SGD:
    # 初始化
    def __init__(self,lr = 0.01):
        self.lr = lr
    # 更新参数：传入参数字典和梯度字典，更新参数字典
    def update(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
# 动量法
class Momentum:
    # 初始化
    def __init__(self,lr=0.01,momentum=0.9):
        self.momentum = momentum
        self.lr = lr
        self.v = None
    def update(self,params,grads):
        # 判断是否是第一次迭代，如果是对v做全0优化
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        # 遍历参数，进行更新
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
# AdaGrad
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr = lr
        self.h = None
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] = grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)
# RMSProp
class RMSProp:
    def __init__(self,lr=0.01,decay=0.99):
        self.lr = lr
        self.decay = decay
        self.h = None
    def update(self,params,grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] = self.decay * self.h[key] + (1 - self.decay) * grads[key] * grads[key]
            self.h[key] += self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-8)

# Adam
class Adam:
    def __init__(self,lr=0.01,beta1=0.9,beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0 #迭代次数
        self.v = None
        self.h = None
    # 更新参数
    def update(self,params,grads):
        if self.h is None:
            self.v,self.h = {},{}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.h[key] = np.zeros_like(val)
        # 迭代次数加一
        self.t += 1
        # 将t对公式的修正影响，直接添加到学习率上
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)
        # 遍历所有参数，迭代更新
        for key in params.keys():
            # self.v[key] = self.beta1 * self.v[key] + (1 - self.beta1) * grads[key]
            # self.h[key] = self.beta2 * self.h[key] + (1 - self.beta2) * (grads[key] ** 2)
            self.v[key] += (1 - self.beta1) * (grads[key] - self.v[key])
            self.h[key] += (1 - self.beta2) * (grads[key] ** 2 - self.h[key])
            params[key] -= lr_t * self.v[key] / (np.sqrt(self.h[key]) + 1e-8)