import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib
# 加载数据集
dataset = pd.read_csv('../data/heart_disease.csv')

# 处理缺失值
dataset.dropna(inplace=True)

dataset.info()

# 数据集划分
x = dataset.drop(['是否患有心脏病'], axis=1)
y = dataset["是否患有心脏病"]

# 将数据集按7:3划分为训练数据与测试数据
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# 数值型特征

numerical_features = ["年龄", "静息血压", "胆固醇", "最大心率", "运动后的ST下降", "主血管数量"]

# 类别型特征

categorical_features = ["胸痛类型", "静息心电图结果", "峰值ST段的斜率", "地中海贫血"]

# 二元特征

binary_features = ["性别", "空腹血糖", "运动性心绞痛"]

# 创建列转换器

preprocessor = ColumnTransformer(

    transformers=[
        # 对数值型特征进行标准化

        ("num", StandardScaler(), numerical_features),

        # 对类别型特征进行独热编码，使用drop="first"避免多重共线性

        ("cat", OneHotEncoder(drop="first"), categorical_features),

        # 二元特征不进行处理

        ("binary", "passthrough", binary_features),

    ]

)

# 执行特征转换

x_train = preprocessor.fit_transform(x_train) # 计算训练集的统计信息并进行转换
x_test = preprocessor.transform(x_test) # 使用训练集计算的信息对测试集进行转换

# 使用K近邻分类模型，K=3
knn = KNeighborsClassifier(n_neighbors=3)
# 模型训练
knn.fit(x_train, y_train)
# 模型评估，计算准确率
knn.score(x_test, y_test)

joblib.dump(knn, "knn_heart_disease")
# 加载模型
knn_loaded = joblib.load("knn_heart_disease")
# 预测
y_pred = knn_loaded.predict(x_test[10:11])
# 打印真实值与预测值
print(y_test.iloc[10], y_pred)

knn = KNeighborsClassifier()

# 网格搜索参数，K值设置为1到10

param_grid = {"n_neighbors": list(range(1, 10))}

# GridSearchCV(estimator=模型, param_grid=网格搜索参数, cv=k折交叉验证)

knn = GridSearchCV(estimator=knn, param_grid=param_grid, cv=10)

# 模型训练

knn.fit(x_train, y_train)

print(pd.DataFrame(knn.cv_results_))  # 所有交叉验证结果

print(knn.best_estimator_)  # 最佳模型

print(knn.best_score_)  # 最佳得分

# 使用最佳模型进行评估

knn = knn.best_estimator_

print(knn.score(x_test, y_test))