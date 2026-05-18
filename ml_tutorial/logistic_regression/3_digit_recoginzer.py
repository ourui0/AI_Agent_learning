import pandas as pd
import matplotlib.pyplot as plt
from imblearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

# 加载数据集
dataset = pd.read_csv('../data/train.csv')

# 划分数据集
x = dataset.drop("label", axis=1)
y = dataset["label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# digit = x_train.iloc[10,:].values.reshape(28,28)
# plt.imshow(digit, cmap="gray")
# plt.show()

# 归一化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 定义模型
model = LogisticRegression(max_iter=500)

# 模型训练
model.fit(x_train, y_train)

# 模型评估
accuracy = model.score(x_test, y_test)
print("预测准确率为：",accuracy)

# 预测
digit = x_test[123]
pred = model.predict(digit.reshape(1, -1))
print(pred)
print(y_test.iloc[123])

plt.imshow(digit.reshape(28, 28), cmap='Greys_r')
plt.show()


