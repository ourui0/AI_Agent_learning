import numpy as np
from common.optimizer import Adam,SGD,Momentum,AdaGrad
import matplotlib.pyplot as plt
from collections import OrderedDict

# 定义目标函数
def f(x,y):
    return x ** 2 / 20 + y ** 2

# 定义计算梯度的函数，返回梯度向量
def f_grad(x,y):
    return x / 10,2 * y

# 参数初始化
init_pos = (-7.0, 2.0)

params, grads = {},{}

# 定义优化器需要的参数字典和梯度字典
optimizers = OrderedDict()
optimizers['SGD'] = SGD(lr = 0.9)
optimizers['AdaGrad'] = AdaGrad(lr = 1.5)
optimizers['Momentum'] = Momentum(lr = 0.11,momentum=0.85)
optimizers['Adam'] = Adam(lr = 0.5,beta1= 0.5)

# 定义子图序列
idx = 1

# 遍历优化器，完成梯度下降法求解最小值
for key in optimizers:
    optimizer = optimizers[key]
    # 将初始参数放入参数字典
    params['x'], params['y'] = init_pos[0],init_pos[1]
    # 用列表保存参数变化轨迹
    x_history, y_history = [],[]
    # 用迭代进行梯度下降法
    for epoch in range(30):
        # 计算梯度
        grads['x'], grads['y'] = f_grad(params['x'],params['y'])
        # 更新参数
        optimizer.update(params,grads)
        x_history.append(params["x"])
        y_history.append(params["y"])
    # 画图
    plt.subplot(2,2,idx)
    idx += 1
    x = np.arange(-10.0, 10.0, 0.01)
    y = np.arange(-5.0, 5.0, 0.01)
    X,Y = np.meshgrid(x,y)
    Z = f(X,Y)
    Z[ Z > 7] = 0
    plt.contour(X,Y,Z)
    # 单独画出最低点
    plt.plot(x_history,y_history,'o-',c = 'red',markersize = 2,label = key)
    plt.legend(loc = 'best')
plt.show()