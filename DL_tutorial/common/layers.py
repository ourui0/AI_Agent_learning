from common.functions import sigmoid,softmax,cross_entropy_error
import numpy as np
 # RelU
class ReLU:
     def __init__(self):
         self.mask = None
     # 前向传播
     def forward(self, x):
         self.mask = (x <= 0)
         y = x.copy()
         y[self.mask] = 0
         return y
     # 反向传播
     def backward(self, dout):
         dx = dout.copy()
         dx[self.mask] = 0
         return dx

# Sigmoid
class Sigmoid:
     def __init__(self):
         self.y = None
     def forward(self, x):
         y = sigmoid(x)
         self.y = y
         return y
     def backward(self, dout):
         dx = dout * (1 - self.y)
         return dx

class Affine:
     def __init__(self,W,b):
         self.W = W
         self.b = b
         # 将输入X保存为属性，方便反向传播计算
         self.x = None
         self.original_x_shape = None
         self.dW = None
         self.db = None
     # 前向传播
     def forward(self, x):
         self.x = x.reshape(x.shape[0],-1)
         y = self.x @ self.W + self.b
         return y

     # 反向传播
     def backward(self, dout):
         # 保存原始形状用于最后恢复
         self.original_x_shape = self.x.shape

         # 1. 计算当前层参数的梯度（全部基于传回来的原始梯度 dout）
         self.dW = self.x.T @ dout  # 形状：(784, 100) @ (100, 50) = (784, 50) -> 完美匹配！
         self.db = np.sum(dout, axis=0)  # 偏置的梯度也是对 dout 求和

         # 2. 计算准备传给前一层的梯度 dx
         dx = dout @ self.W.T

         # 3. 恢复成前向传播输入时的形状并返回
         dx = dx.reshape(*self.original_x_shape)
         return dx

# 带损失的输出层
class SoftmaxWithLoss:
     def __init__(self):
         self.loss = None
         self.y = None
         self.t = None
     # 前向传播
     def forward(self, x,t):
         self.t = t
         self.y = softmax(x)
         self.loss = cross_entropy_error(self.y,self.t)
         return self.loss

     # 反向传播
     def backward(self,dout = 1):
        n = self.t.shape[0]
        # 分情况讨论：如果t是独热编码标签，直接带入公式
        if self.t.size == self.y.size:
            dx = self.y - self.t
        # 如果t是顺序编码标签
        else:
            dx = self.y.copy()
            dx[ np.arange(n), self.t ] -= 1
        return dx * dout/ n
