import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification

X, y = make_classification(random_state=2)
# print("X", X.shape,y.shape)
# 只取两个特征值, 二维特征值方便可视化
X = X.T[:2, :]
y = np.expand_dims(y, axis=0)
print("X", X.shape)
print("y", y.shape)


interval = 0.2
x_min, x_max = X[0, :].min() - .5, X[0, :].max() + .5
y_min, y_max = X[1, :].min() - .5, X[1, :].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, interval),
                     np.arange(y_min, y_max, interval))


cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=cm_bright, edgecolors='k')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.xlabel("theta_1")
plt.ylabel("theta_2")
plt.show()


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 进行正向计算并求出损失
def forward(X, theta, bias):
    z = np.dot(theta.T, X) + bias
    y_hat = sigmoid(z)
    return y_hat
def compute_loss(y, y_hat):
    e = 1e-8
    return - y * np.log(y_hat + e) - (1 - y) * np.log(1 - y_hat + e)

# 梯度下降, 参见(eq.3), (eq.4), (eq.5), (eq.6)
def backward(X, y, y_hat, theta):
    m = X.shape[-1]
    # 求theta的梯度
    delta_theta = np.dot(X, (y_hat-y).T) / m #(eq.3)(eq.5)
    # 求bias的梯度
    delta_bias = np.mean(y_hat-y) #(eq.4)(eq.6)
    return delta_theta, delta_bias


# 初始化theta为全0
theta = np.zeros([2, 1])
# 初始化偏置为0
bias = np.zeros([1])

for i in range(1000):
    # 正向
    y_hat = forward(X, theta, bias)
    # 计算损失
    loss = np.mean(compute_loss(y, y_hat))
    if i%100 == 0:
        print("step:",i,"loss:",loss)
    # 梯度下降
    delta_theta, delta_bias = backward(X, y, y_hat, theta)
    # 更新参数
    theta -= 0.1 * delta_theta
    bias -= 0.1 * delta_bias


    # 画等高线图
data = np.c_[xx.ravel(), yy.ravel()].T
# 计算出区域内每一个点的模型预测值
Z = forward(data, theta, bias)
Z = Z.reshape(xx.shape)

# 定义画布大小
plt.figure(figsize=(10,8))
# 画等高线
plt.contourf(xx, yy, Z, 10, cmap=plt.cm.RdBu, alpha=.8)
# 画轮廓
contour = plt.contour(xx, yy, Z, 10, colors="k", linewidths=.5)
plt.scatter(X[0, :], X[1, :], c=y.ravel(), cmap=cm_bright, edgecolors='k')
# 标出等高线对应的数值
plt.clabel(contour, inline=True, fontsize=10)
plt.show()