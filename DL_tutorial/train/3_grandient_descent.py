import numpy as np
import matplotlib.pyplot as plt
from common.gradient import gradient_descent

def f(x):
    return x[0] ** 2 + x[1] ** 2

# 定义初始点
init_x = np.array([-3.0,4.0])

# 定义超参数
lr = 0.1
num_iter = 20

# 梯度下降法求最小值点
x,x_history = gradient_descent(f,init_x,lr,num_iter)
print(x)
print(x_history)

# 画图
plt.scatter(x_history[:,0],x_history[:,1])

plt.plot([-5,5],[0,0],"--b")
plt.plot([0,0],[-5,5],"--b")
plt.show()