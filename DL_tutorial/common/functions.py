# 激活函数
import numpy as np
# 阶跃函数
def step_function0(x):
    return 1 if x >= 0 else 0

def step_function(x):
    return np.array(x > 0, dtype=int)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
if __name__ == '__main__':
    print(step_function(np.array([0, 1,-1,-5,10,6,7,8])))
    print(sigmoid(np.array([0,1,-1,-5,10,6,7,8])))
    print(step_function0(-5))