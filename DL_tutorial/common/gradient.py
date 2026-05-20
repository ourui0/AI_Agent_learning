import numpy as np
# 数值微分
def numerical_diff0(f, x):
    h = 1e-4
    return (f(x + h) - f(x)) /h
# 中心差分实现
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x -h)) / (h * 2)

# 利用数值微分计算梯度
def _numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    
    # 遍历x向量中的每个自变量xi
    for i in range (x.size):
        xi = x[i]
        # 对当前自变量做微小改变，并计算函数值
        x[i] = xi + h
        fxh1 = f(x)
        x[i] = xi - h
        fxh2 = f(x)
        # 中心差分
        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = xi
    return grad

# 扩展到二维情况：X为n * m矩阵，表示n个输入数据
def numerical_gradient(f, x):
    if x.ndim == 1:
        return _numerical_gradient(f, x)
    else:
        grad = np.zeros_like(x)
        for i ,x in enumerate(x):
            grad[i] = _numerical_gradient(f, x)
        return grad
# 梯度下降法,返回最小值点以及更新路径
def gradient_descent(f,init_x,lr = 0.01,step_num = 100):
    x = init_x
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x = x - lr * grad
    return x,np.array(x_history)

