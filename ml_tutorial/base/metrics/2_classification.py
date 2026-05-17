from sklearn.datasets import make_classification #自动生成分类数据集
from sklearn.model_selection import train_test_split #划分数据集
from sklearn.linear_model import LogisticRegression #逻辑回归分类模型
from sklearn.metrics import classification_report #分类评估报告

#生成数据集
x,y = make_classification(n_samples=1000,n_features=20,n_classes=2,random_state=42)

#划分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)

#定义模型
model = LogisticRegression()

#训练模型
model.fit(x_train,y_train)

#预测
y_pred = model.predict(x_test)

#生成分类报告
report = classification_report(y_test,y_pred)

print(report)

from sklearn.metrics import roc_auc_score
# 得到预测概率值
y_pred_proba = model.predict_proba(x_test)[:,1]
# 计算AUC值
auc = roc_auc_score(y_test,y_pred_proba)
print(auc)
