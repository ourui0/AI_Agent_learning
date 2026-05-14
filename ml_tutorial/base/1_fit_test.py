import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.preprocessing import PolynomialFeatures # 构建多项式特征
from sklearn.linear_model import LinearRegression # 线性回归模型
from sklearn.metrics import mean_squared_error # 均方误差

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
#构建训练集
x = np.linspace(-3,3,300).reshape(-1,1)
y = np.sin(x) + np.random.uniform(low = -0.5,high = 0.5,size= 300).reshape(-1,1)

# print(x.shape)
# print(y.shape)

fig,ax = plt.subplots(1,3,figsize=(15,4))
ax[0].scatter(x,y,color='blue')
ax[1].scatter(x,y,color='blue')
ax[2].scatter(x,y,color='blue')

#划分训练集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
x_train1 = x_train
x_test1 = x_test
#定义线性回归方程
model = LinearRegression()

#分三种情况

# 欠拟合

model.fit(x_train1,y_train)

# 查看训练完成的模型参数

print("斜率为：",model.coef_)
print("截距为：",model.intercept_)

#画出拟合直线
ax[0].plot(x,model.predict(x),color='r')

# 测试
y_pred1 = model.predict(x_test1)
# 计算训练误差和测试误差
test_loss1 = mean_squared_error(y_test,y_pred1)
train_loss1 = mean_squared_error(y_train,model.predict(x_train1))
ax[0].text(-3,1,f"测试误差：{test_loss1:.4f}")
ax[0].text(-3,1.3,f"训练误差：{train_loss1:.4f}")


# 恰好拟合
ploy5 = PolynomialFeatures(degree=5)

x_train2 = ploy5.fit_transform(x_train)
x_test2 = ploy5.transform(x_test)

model.fit(x_train2,y_train)

# 画出拟合曲线
ax[1].plot(x,model.predict(ploy5.fit_transform(x)),color='r')
# 测试
y_pred2 = model.predict(x_test2)
# 计算训练误差和测试误差
test_loss2 = mean_squared_error(y_test,y_pred2)
train_loss2 = mean_squared_error(y_train,model.predict(x_train2))
ax[1].text(-3,1,f"测试误差：{test_loss2:.4f}")
ax[1].text(-3,1.3,f"训练误差：{train_loss2:.4f}")
# 过拟合
ploy20 = PolynomialFeatures(degree=40)

x_train3 = ploy20.fit_transform(x_train)
x_test3 = ploy20.transform(x_test)

model.fit(x_train3,y_train)

# 画出拟合曲线
ax[2].plot(x,model.predict(ploy20.fit_transform(x)),color='r')
# 测试
y_pred3 = model.predict(x_test3)
# 计算训练误差和测试误差
test_loss3 = mean_squared_error(y_test,y_pred3)
train_loss3 = mean_squared_error(y_train,model.predict(x_train3))
ax[2].text(-3,1,f"测试误差：{test_loss3:.4f}")
ax[2].text(-3,1.3,f"训练误差：{train_loss3:.4f}")
plt.show()