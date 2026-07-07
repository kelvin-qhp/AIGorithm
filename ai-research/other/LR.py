import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class LogisticRegressionManual:
    """
    手动实现的二分类逻辑回归
    使用梯度下降优化，损失函数为交叉熵
    """

    def __init__(self, learning_rate=0.1, n_iterations=1000):
        self.learning_rate = learning_rate  # 学习率
        self.n_iterations = n_iterations  # 迭代次数
        self.weights = None  # 权重系数
        self.bias = None  # 偏置项
        self.loss_history = []  # 记录损失变化

    def sigmoid(self, z):
        """Sigmoid激活函数：将线性输出映射到0-1之间"""
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        训练模型：使用梯度下降最小化交叉熵损失
        X: 特征矩阵 (n_samples, n_features)
        y: 标签向量 (n_samples,)
        """
        n_samples, n_features = X.shape

        # 初始化参数为小的随机值
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0

        # 梯度下降迭代
        for i in range(self.n_iterations):
            # 1. 前向传播：计算线性输出和预测概率
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_output)

            # 2. 计算交叉熵损失
            loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
            self.loss_history.append(loss)

            # 3. 计算梯度（对交叉熵损失求导）
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # 4. 更新参数（梯度下降）
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 每100次迭代打印一次损失
            if i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """预测概率"""
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    def predict(self, X, threshold=0.5):
        """预测类别（0或1）"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def score(self, X, y):
        """计算准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


# ========== 数据准备 ==========
# 生成一个二分类数据集
X, y = make_classification(
    n_samples=1000,  # 样本数
    n_features=2,  # 2个特征，方便可视化
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# 标准化特征（加速收敛）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# ========== 训练手动实现的LR ==========
print("=" * 50)
print("训练手动实现的逻辑回归")
print("=" * 50)

lr_manual = LogisticRegressionManual(learning_rate=0.1, n_iterations=1000)
lr_manual.fit(X_train, y_train)

# 测试集评估
train_acc = lr_manual.score(X_train, y_train)
test_acc = lr_manual.score(X_test, y_test)
print(f"\n手动LR - 训练集准确率: {train_acc:.4f}")
print(f"手动LR - 测试集准确率: {test_acc:.4f}")

# ========== 绘制损失下降曲线 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(lr_manual.loss_history)
plt.xlabel('迭代次数')
plt.ylabel('交叉熵损失')
plt.title('损失下降曲线')
plt.grid(True)

# ========== 绘制决策边界 ==========
plt.subplot(1, 2, 2)
# 生成网格点
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = lr_manual.predict(grid_points).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu)
plt.scatter(X_scaled[y == 0, 0], X_scaled[y == 0, 1], c='blue', label='类别0', alpha=0.5)
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1], c='red', label='类别1', alpha=0.5)
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('逻辑回归决策边界')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()