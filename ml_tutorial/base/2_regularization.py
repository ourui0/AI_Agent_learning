import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split # 划分训练集和测试集
from sklearn.preprocessing import PolynomialFeatures # 构建多项式特征
from sklearn.linear_model import LinearRegression, Lasso,  Ridge# 线性回归模型,Lasso回归,岭回归
from sklearn.metrics import mean_squared_error # 均方误差

plt.rcParams['font.sans-serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
#构建训练集
x = np.linspace(-3,3,300).reshape(-1,1)
y = np.sin(x) + np.random.uniform(low = -0.5,high = 0.5,size= 300).reshape(-1,1)

# print(x.shape)
# print(y.shape)

fig,ax = plt.subplots(2,3,figsize=(15,8))
ax[0,0].scatter(x,y,color='blue')
ax[0,1].scatter(x,y,color='blue')
ax[0,2].scatter(x,y,color='blue')

#划分训练集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)

# 准备数据：构建21维特征
ploy20 = PolynomialFeatures(degree=20)

x_train = ploy20.fit_transform(x_train)
x_test = ploy20.transform(x_test)

# 分三种情况：定义不同的模型
# 不加正则化线性回归方程
model = LinearRegression()

model.fit(x_train,y_train)

# 画出拟合曲线
ax[0,0].plot(x,model.predict(ploy20.fit_transform(x)),color='r')
# 测试
y_pred1 = model.predict(x_test)
# 计算测试误差
test_loss1 = mean_squared_error(y_test,y_pred1)
ax[0,0].text(-3,1,f"测试误差：{test_loss1:.4f}")

#系数直方图
ax[1,0].bar(np.arange(21),model.coef_.reshape(-1),color='blue')
# 加L1正则化,Lasso模型
lasso = Lasso(alpha=0.01)

lasso.fit(x_train,y_train)

# 画出拟合曲线
ax[0,1].plot(x,lasso.predict(ploy20.fit_transform(x)),color='r')
# 测试
y_pred2 = lasso.predict(x_test)
# 计算训练误差和测试误差
test_loss2 = mean_squared_error(y_test,y_pred2)
ax[0,1].text(-3,1,f"测试误差：{test_loss2:.4f}")

#系数直方图
ax[1,1].bar(np.arange(21),lasso.coef_.reshape(-1),color='blue')
# 加L2正则化,岭模型
ridge = Ridge(alpha=0.01)

ridge.fit(x_train,y_train)

# 画出拟合曲线
ax[0,2].plot(x,ridge.predict(ploy20.fit_transform(x)),color='r')
# 测试
y_pred3 = lasso.predict(x_test)
# 计算训练误差和测试误差
test_loss3 = mean_squared_error(y_test,y_pred3)
ax[0,2].text(-3,1,f"测试误差：{test_loss3:.4f}")

#系数直方图
ax[1,2].bar(np.arange(21),ridge.coef_.reshape(-1),color='blue')
plt.show()
