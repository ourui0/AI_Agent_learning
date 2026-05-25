import torch
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader

# 读取数据
fashion_mnist_train = pd.read_csv("../data/fashion-mnist_train.csv")
fashion_mnist_test = pd.read_csv("../data/fashion-mnist_test.csv")
# 将数据转换为张量，原数据形状为n×1×784，转换为n×1×28×28的张量
X_train = torch.tensor(fashion_mnist_train.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_train = torch.tensor(fashion_mnist_train.iloc[:, 0].values, dtype=torch.int64)
X_test = torch.tensor(fashion_mnist_test.iloc[:, 1:].values, dtype=torch.float32).reshape(-1, 1, 28, 28)
y_test = torch.tensor(fashion_mnist_test.iloc[:, 0].values, dtype=torch.int64)
plt.imshow(X_train[12345, 0, :, :], cmap="gray")
plt.show()
# 构建数据集
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 构建CNN模型
model = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5,padding=2),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10),
)

# 给模型一个输入数据做前向传播
x = torch.rand(10,1,28,28)

for layer in model:
    x = layer(x)
    print(f'{layer.__class__.__name__:<12}: output shape:{x.shape}')


# 模型训练

def train(model, train_dataset, test_dataset, lr, epoch_num, batch_size, device):
    def init_weights(layer):
        # 对线性层和卷积层使用Xavier均匀分布初始化参数
        if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
            nn.init.xavier_uniform_(layer.weight)
    model.apply(init_weights)  # 初始化参数
    model.to(device)  # 将模型加载到设备
    loss = nn.CrossEntropyLoss()  # 损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 优化器
    for epoch in range(epoch_num):
        # 训练过程
        model.train()  # 将模型设置为训练模式
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        loss_accumulate = 0
        train_correct_accumulate = 0
        for batch_count, (X, y) in enumerate(train_loader):
            # 前向传播
            X, y = X.to(device), y.to(device)
            output = model(X)
            # 反向传播
            loss_value = loss(output, y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            # 累加损失
            loss_accumulate += loss_value.item()
            # 累加正确输出的数量
            _, pred = output.max(1)
            train_correct_accumulate += pred.eq(y).sum()
            # 打印进度条
            print(f"\repoch:{epoch:0>2}[{'=' * (int((batch_count + 1) / len(train_loader) * 50)):<50}]", end="")
        this_loss = loss_accumulate / len(train_loader)  # 计算平均损失
        this_train_correct = train_correct_accumulate / len(train_dataset)  # 计算训练准确率

        # 验证过程

        model.eval()  # 将模型设置为评估模式
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
        test_correct_accumulate = 0
        with torch.no_grad():  # 关闭梯度计算
            for X, y in test_loader:
                # 前向传播
                X, y = X.to(device), y.to(device)
                output = model(X)
                # 累加正确输出的数量
                _, pred = output.max(1)
                test_correct_accumulate += pred.eq(y).sum()
        this_test_correct = test_correct_accumulate / len(test_dataset)  # 计算验证准确率
        # 打印损失，训练准确率，验证准确率
        print(f" loss:{this_loss:.6f}, train_acc:{this_train_correct:.6f}, test_acc:{this_test_correct:.6f}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 如果cude可用则使用cuda，否则使用cpu
train(model, train_dataset, test_dataset, lr=0.9, epoch_num=20, batch_size=256, device=device)