#导入相关库和函数
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#由于伦理问题，波士顿数据集已经被移除，我们用加利福尼亚房屋数据集替代
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()



#加载数据集
set = fetch_california_housing()
X = set.data#特征矩阵
Y = set.target#目标变量

#划分训练集和测试集(测试集占20%)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


#由于波士顿房价数据集特征尺度差异较大，我们进行特征缩放
#套公式
def standardize(X):
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_train = standardize(X_train)
X_test = standardize(X_test)

#添加一列1，使模型能够计算偏置截距项，提高拟合能力
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

#最小二乘法求解析解
def solve(X, Y):
    return np.linalg.inv(X.T @ X) @ X.T @ Y

w = solve(X_train, Y_train)


#代价函数
def cost(X, Y, w):
    return np.mean((X @ w - Y) ** 2)

#计算梯度
def gradient(X, Y, w):
    return 2 * X.T @ (X @ w - Y) / X.shape[0]


#梯度下降法
def gradient_descent(X, Y, learning_rate=0.01, num=1000):
    w = np.zeros(X.shape[1])
    losses = []

    for i in range(num):
        gradient1 = gradient(X, Y, w)
        w -= learning_rate * gradient1
        loss = cost(X, Y, w)
        losses.append(loss)

    return w, losses


res, losses = gradient_descent(X_train, Y_train)

#分析评估模型性能
# 计算测试集上的预测结果
Y_pred = X_test @ w

# 计算测试集上的均方根误差（RMSE）
rmse = np.sqrt(np.mean((Y_pred - Y_test) ** 2))
print("RMSE:", rmse)

#RMSE较小，说明预测结果与实际结果相差较小，拟合效果较好