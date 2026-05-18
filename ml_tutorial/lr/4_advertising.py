import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor
from sklearn.metrics import  mean_squared_error


# 加载数据集
dataset = pd.read_csv("../data/advertising.csv")

dataset.drop(dataset.columns[0], axis=1, inplace=True)
dataset.dropna(inplace=True)

dataset.info()
print(dataset.head())
# 划分数据集
x = dataset.drop("Sales", axis=1)
y = dataset["Sales"]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

# 数据标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义模型
model_lr = LinearRegression()
model_SGD = SGDRegressor()

# 训练
model_SGD.fit(x_train,y_train)
print("SGD模型系数",model_SGD.coef_)
print("SGD模型截距",model_SGD.intercept_)
model_lr.fit(x_train,y_train)
print("正规方程法模型系数",model_lr.coef_)
print("正规方程法截距",model_lr.intercept_)

# 验证
y_pred_lr = model_lr.predict(x_test)
y_pred_sgd = model_SGD.predict(x_test)

# 使用评价指标评价模型

print("正规方程法 MSE",mean_squared_error(y_test,y_pred_lr))
print("SGD MSE",mean_squared_error(y_test,y_pred_sgd))

print("正规方程法决定系数",model_lr.score(x_test,y_test))
print("SGD 决定系数",model_SGD.score(x_test,y_test))


