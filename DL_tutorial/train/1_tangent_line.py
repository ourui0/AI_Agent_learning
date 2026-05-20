import numpy as np
import matplotlib.pyplot as plt
from common.gradient import numerical_diff

def f(x):
    return 0.01 * x ** 2 + 0.1 * x

def tangent_line(f,x):
    a = numerical_diff(f,x)
    print("切线斜率为",a)
    # 计算截距
    b = f(x) - a * x
    return lambda x : a * x + b

# 定义
x0 = 5.0

# 得到切线方程
tf = tangent_line(f,x0)

# 画图
x = np.arange(0.0,20.0,0.1)
y1 = f(x)
y2 = tf(x)
plt.plot(x,y1)
plt.plot(x,y2)
plt.xlabel("x")
plt.ylabel("y")
plt.show()