import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score# 引入轮廓系数来评估

# 定义KMeans类
class KMeans:
    def __init__(self, n=3, max_num=100):  # n:聚类数目，max_num:迭代极限
        self.n = n
        self.max_num = max_num

    def fit(self, X):
        # 初始化聚类中心
        np.random.seed(21) # 设置随机种子
        self.centers = X[np.random.choice(len(X), self.n, replace=False)]  # 从样本数据 X 中无放回地选择 self.n 个样本作为初始聚类中心

        for _ in range(self.max_num):
            # 分配样本到最近的聚类中心
            labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2), axis=1)  # 计算欧氏距离

            # 更新聚类中心为每个簇的均值
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n)])

            # 检查是否收敛，收敛则结束训练
            if np.allclose(new_centers, self.centers):
                break

            self.centers = new_centers

    # 预测函数
    def predict(self, X):
        return np.argmin(np.linalg.norm(X[:, np.newaxis] - self.centers, axis=2), axis=1)


# 加载鸢尾花数据集
iris = load_iris()
X, Y = iris.data, iris.target

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=21)

# 训练K-means模型
kmeans = KMeans(n=3)
kmeans.fit(X_train)

# 预测训练集和测试集
train_hat = kmeans.predict(X_train)
test_hat = kmeans.predict(X_test)

# 评估模型
train_silhouette_score = silhouette_score(X_train, train_hat)
test_silhouette_score = silhouette_score(X_test, test_hat)

print("训练集轮廓系数:", train_silhouette_score)
print("测试集轮廓系数:", test_silhouette_score)