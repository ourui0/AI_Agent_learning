import os
os.environ["OMP_NUM_THREADS"] = "2"
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 或者用 'Heiti TC'
plt.rcParams['axes.unicode_minus'] = False              # 顺便修复负号 [-] 显示为方块的问题
# 随机生成数据样本点
x,y = make_blobs(n_samples=300,n_features=2,centers=3,cluster_std=2,random_state=42)

fig, ax = plt.subplots(2,figsize =(8,8))
ax[0].scatter(x[:,0],x[:,1],c = y,s=50,label = "原始数据")
ax[0].set_title("原始数据")
ax[0].legend()
# plt.show()

# 定义模型和训练
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

# 获取聚类的簇中心
centers = kmeans.cluster_centers_

print(centers)

# 预测：得到每个样本点的簇标签
y_pred = kmeans.predict(x)

# 画图
ax[1].scatter(x[:,0],x[:,1],c = y_pred,s=50,label = "聚类数据")
ax[1].scatter(centers[:,0],centers[:,1],c = 'r',s=200,label = "簇中心")
ax[1].set_title("聚类数据")
ax[1].legend()
plt.show()