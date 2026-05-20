import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def get_data():
    # 加载数据
    data = pd.read_csv("../data/train.csv")

    # 划分数据集
    x = data.drop(columns="label", axis=1)
    y = data["label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # 归一化
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # 转换成ndarray
    y_train = y_train.values
    y_test = y_test.values
    return x_test, y_test, x_train, y_train
