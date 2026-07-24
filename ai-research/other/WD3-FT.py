import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# ==================== 1. 优化后的 Wide & Deep 模型 ====================
class WideDeepModel(nn.Module):
    def __init__(self, num_continuous, num_categories, embedding_dims, hidden_dims,
                 cross_features_indices=None, dropout_rate=0.3):
        """
        Args:
            num_continuous: 连续特征数量
            num_categories: list, 每个离散特征的类别数
            embedding_dims: list, 每个离散特征的 embedding 维度
            hidden_dims: list, Deep 部分隐藏层维度
            cross_features_indices: list of tuples, 手工交叉特征的特征对索引
            dropout_rate: Dropout 比例
        """
        super(WideDeepModel, self).__init__()

        # 1. Embedding 层
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories[i], embedding_dims[i])
            for i in range(len(num_categories))
        ])

        # 计算总特征维度
        self.embedding_total_dim = sum(embedding_dims)
        self.total_dim = num_continuous + self.embedding_total_dim

        # 2. Wide 部分 - 使用稀疏特征（离散特征的一阶项 + 交叉特征）
        # Wide 部分只使用离散特征（原始索引）和交叉特征
        self.wide_linear = nn.Linear(len(num_categories), 1, bias=True)

        # 如果有交叉特征，额外添加交叉特征的线性层
        self.cross_features_indices = cross_features_indices
        if cross_features_indices:
            self.cross_linear = nn.Linear(len(cross_features_indices), 1, bias=False)

        # 3. Deep 部分 - MLP
        deep_input_dim = self.total_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(deep_input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            deep_input_dim = hidden_dim
        layers.append(nn.Linear(deep_input_dim, 1))
        self.deep = nn.Sequential(*layers)

        # 4. 输出层合并权重（可学习）
        self.final_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, continuous, categorical):
        """
        Args:
            continuous: [batch, num_continuous]
            categorical: [batch, num_categories]
        """
        batch_size = continuous.size(0)

        # ---- Embedding 处理 ----
        embedded = [self.embeddings[i](categorical[:, i]) for i in range(categorical.shape[1])]
        embedded = torch.cat(embedded, dim=1)  # [batch, embedding_total_dim]

        # ---- Deep 部分的输入 ----
        deep_input = torch.cat([continuous, embedded], dim=1)  # [batch, total_dim]
        deep_output = self.deep(deep_input)  # [batch, 1]

        # ---- Wide 部分的输入（使用离散特征原始索引） ----
        # 注意：这里传入的是 categorical 的原始索引值
        wide_input = categorical.float()  # [batch, num_categories]
        wide_output = self.wide_linear(wide_input)  # [batch, 1]

        # ---- 交叉特征（Wide 部分的核心） ----
        if self.cross_features_indices:
            cross_features = []
            for idx1, idx2 in self.cross_features_indices:
                # 特征交叉：两个离散特征的乘积（指示同时出现）
                cross = (categorical[:, idx1] == categorical[:, idx2]).float().unsqueeze(1)
                cross_features.append(cross)
            cross_tensor = torch.cat(cross_features, dim=1)  # [batch, num_cross]
            cross_output = self.cross_linear(cross_tensor)  # [batch, 1]
            wide_output = wide_output + cross_output

        # ---- 合并输出 ----
        # 使用 sigmoid 进行二分类
        output = torch.sigmoid(wide_output + deep_output)
        return output.squeeze()


# ==================== 2. 数据预处理函数 ====================
def preprocess_data(continuous_data, categorical_data, labels, test_size=0.2):
    """预处理数据：归一化、编码、划分"""

    # 1. 连续特征标准化
    scaler = StandardScaler()
    continuous_scaled = scaler.fit_transform(continuous_data)

    # 2. 离散特征 Label Encoding
    categorical_encoded = categorical_data.copy()
    label_encoders = []
    for i in range(categorical_data.shape[1]):
        le = LabelEncoder()
        categorical_encoded[:, i] = le.fit_transform(categorical_data[:, i])
        label_encoders.append(le)

    # 3. 划分数据集（保持类别分布）
    X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(
        continuous_scaled, categorical_encoded, labels,
        test_size=test_size, random_state=42, stratify=labels
    )

    return {
        'train': (X_cont_train, X_cat_train, y_train),
        'val': (X_cont_val, X_cat_val, y_val),
        'scaler': scaler,
        'label_encoders': label_encoders
    }


# ==================== 3. 数据加载器 ====================
def create_dataloaders(X_cont, X_cat, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(
        torch.tensor(X_cont, dtype=torch.float32),
        torch.tensor(X_cat, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32)
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ==================== 4. 训练函数（改进版） ====================
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer,
                scheduler=None, device='cpu', early_stop_patience=5):
    """增强版训练函数，包含 Early Stopping 和 Learning Rate Scheduler"""

    model = model.to(device)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        # ---- 训练阶段 ----
        model.train()
        train_loss, train_acc = 0.0, 0.0
        train_steps = 0

        for continuous, categorical, labels in train_loader:
            continuous = continuous.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(continuous, categorical)
            loss = criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_acc += (preds == labels).float().mean().item()
            train_steps += 1

        avg_train_loss = train_loss / train_steps
        avg_train_acc = train_acc / train_steps
        train_losses.append(avg_train_loss)
        train_accs.append(avg_train_acc)

        # ---- 验证阶段 ----
        val_loss, val_acc = evaluate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # ---- 学习率调度 ----
        if scheduler:
            scheduler.step(val_loss)

        # ---- Early Stopping ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_wide_deep_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # ---- 打印日志 ----
        print(f'Epoch {epoch + 1:2d}/{num_epochs} | '
              f'Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | '
              f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_wide_deep_model.pth'))
    return train_losses, train_accs, val_losses, val_accs


# ==================== 5. 评估函数 ====================
def evaluate_model(model, val_loader, criterion, device):
    model.eval()
    val_loss, val_acc = 0.0, 0.0
    val_steps = 0

    with torch.no_grad():
        for continuous, categorical, labels in val_loader:
            continuous = continuous.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)

            outputs = model(continuous, categorical)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            preds = (outputs > 0.5).float()
            val_acc += (preds == labels).float().mean().item()
            val_steps += 1

    return val_loss / val_steps, val_acc / val_steps


# ==================== 6. 可视化函数 ====================
def plot_curves(train_losses, val_losses, train_accs, val_accs):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
    axes[0].plot(epochs, val_losses, 'r-o', label='Val Loss', linewidth=2, markersize=6)
    axes[0].set_title('Training and Validation Loss', fontsize=14)
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, 'b-o', label='Train Accuracy', linewidth=2, markersize=6)
    axes[1].plot(epochs, val_accs, 'r-o', label='Val Accuracy', linewidth=2, markersize=6)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14)
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ==================== 7. 主程序 ====================
if __name__ == "__main__":

    # ---- 7.1 生成有意义的合成数据 ----
    # 使用 make_classification 生成有模式的数据
    num_samples = 5000
    num_continuous = 5
    num_categorical = 3
    num_classes = 2

    # 生成连续特征 + 标签（有真实模式）
    X_cont, y = make_classification(
        n_samples=num_samples,
        n_features=num_continuous,
        n_informative=3,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        random_state=42
    )

    # 生成离散特征（与标签有一定相关性）
    np.random.seed(42)
    cat_features = []
    for i in range(num_categorical):
        # 让离散特征与标签相关
        cat_feat = np.random.randint(0, 10 + i * 5, num_samples)
        # 对部分样本根据标签调整
        mask = y == 1
        cat_feat[mask] = (cat_feat[mask] + np.random.randint(0, 3, mask.sum())) % (10 + i * 5)
        cat_features.append(cat_feat)
    X_cat = np.column_stack(cat_features)

    # ---- 7.2 数据预处理 ----
    data = preprocess_data(X_cont, X_cat, y, test_size=0.2)

    # ---- 7.3 创建 DataLoader ----
    batch_size = 128
    train_loader = create_dataloaders(
        data['train'][0], data['train'][1], data['train'][2],
        batch_size=batch_size, shuffle=True
    )
    val_loader = create_dataloaders(
        data['val'][0], data['val'][1], data['val'][2],
        batch_size=batch_size, shuffle=False
    )

    # ---- 7.4 模型参数 ----
    num_categories = [10, 15, 20]  # 每个离散特征的类别数
    embedding_dims = [4, 5, 6]  # embedding 维度
    hidden_dims = [128, 64, 32]  # Deep 部分隐藏层

    # 定义交叉特征（Wide 部分的关键）
    cross_features = [(0, 1), (0, 2), (1, 2)]  # 特征对交叉

    # ---- 7.5 初始化模型 ----
    model = WideDeepModel(
        num_continuous=num_continuous,
        num_categories=num_categories,
        embedding_dims=embedding_dims,
        hidden_dims=hidden_dims,
        cross_features_indices=cross_features,
        dropout_rate=0.3
    )

    # 打印模型参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")

    # ---- 7.6 训练配置 ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    # ---- 7.7 训练 ----
    num_epochs = 50
    train_losses, train_accs, val_losses, val_accs = train_model(
        model, train_loader, val_loader, num_epochs, criterion, optimizer,
        scheduler=scheduler, device=device, early_stop_patience=10
    )

    # ---- 7.8 结果展示 ----
    print(f"\n最佳验证准确率: {max(val_accs):.4f}")
    print(f"最终训练准确率: {train_accs[-1]:.4f}")
    print(f"最终验证准确率: {val_accs[-1]:.4f}")

    plot_curves(train_losses, val_losses, train_accs, val_accs)