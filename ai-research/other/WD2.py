
# Wide & Deep Learning for Recommender Systems
# 使用PyTorch实现，包含完整的训练示例

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Wide & Deep 模型定义
# ============================================================

class WideAndDeep(nn.Module):
    """
    Wide & Deep Learning 模型

    论文: Cheng et al., "Wide & Deep Learning for Recommender Systems", 2016

    架构:
    - Wide部分: 线性模型，处理交叉特征（cross-product features）
    - Deep部分: 深度神经网络，处理embedding后的稠密特征
    - 输出: Wide + Deep 的联合预测
    """

    def __init__(self,
                 num_dense_features,      # 连续特征数量
                 num_categories,          # 类别特征数量
                 category_sizes,          # 每个类别特征的取值数量列表
                 embedding_dim=8,          # Embedding维度
                 deep_hidden_units=[128, 64, 32],  # Deep部分隐藏层
                 wide_input_dim=None,      # Wide部分输入维度（交叉特征）
                 dropout_rate=0.2):
        super(WideAndDeep, self).__init__()

        self.num_categories = num_categories
        self.embedding_dim = embedding_dim

        # ---------- Deep 部分 ----------
        # Embedding层：将稀疏类别特征映射为稠密向量
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=size, embedding_dim=embedding_dim)
            for size in category_sizes
        ])

        # Deep部分的输入维度 = 连续特征数 + 所有embedding维度之和
        deep_input_dim = num_dense_features + num_categories * embedding_dim

        # 构建Deep部分的MLP
        deep_layers = []
        input_dim = deep_input_dim
        for hidden_dim in deep_hidden_units:
            deep_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.BatchNorm1d(hidden_dim)
            ])
            input_dim = hidden_dim

        self.deep_network = nn.Sequential(*deep_layers)

        # ---------- Wide 部分 ----------
        # Wide部分是一个线性模型
        if wide_input_dim is not None:
            self.wide_linear = nn.Linear(wide_input_dim, 1, bias=True)
        else:
            # 如果没有显式wide特征，使用一个可学习的偏置
            self.wide_linear = None
            self.wide_bias = nn.Parameter(torch.zeros(1))

        # ---------- 最终输出层 ----------
        # 合并Wide和Deep的输出
        self.output_layer = nn.Linear(deep_hidden_units[-1] + 1, 1)

    def forward(self, dense_features, category_features, wide_features=None):
        """
        前向传播

        Args:
            dense_features: [batch_size, num_dense_features] 连续特征
            category_features: [batch_size, num_categories] 类别特征索引
            wide_features: [batch_size, wide_input_dim] 交叉特征（可选）

        Returns:
            predictions: [batch_size, 1] 预测值
        """
        # Deep部分
        # 1. Embedding
        embedded = []
        for i, emb_layer in enumerate(self.embeddings):
            # [batch_size, 1, embedding_dim]
            emb = emb_layer(category_features[:, i])
            embedded.append(emb)

        # 拼接所有embedding: [batch_size, num_categories * embedding_dim]
        embedded = torch.cat(embedded, dim=1)

        # 2. 拼接连续特征和embedding
        deep_input = torch.cat([dense_features, embedded], dim=1)

        # 3. 通过Deep网络
        deep_output = self.deep_network(deep_input)  # [batch_size, last_hidden]

        # Wide部分
        if self.wide_linear is not None and wide_features is not None:
            wide_output = self.wide_linear(wide_features)  # [batch_size, 1]
        else:
            wide_output = self.wide_bias.expand(dense_features.size(0), 1)

        # 合并Wide和Deep
        combined = torch.cat([deep_output, wide_output], dim=1)  # [batch_size, last_hidden + 1]

        # 最终输出
        output = torch.sigmoid(self.output_layer(combined))

        return output


# ============================================================
# 2. 数据准备（模拟推荐系统数据）
# ============================================================

def create_synthetic_data(n_samples=10000, random_state=42):
    """
    创建模拟的推荐系统数据（类似CTR预测场景）

    特征说明:
    - 连续特征: 用户年龄、收入、观看时长等
    - 类别特征: 用户ID、物品ID、类别、时段、设备等
    - 交叉特征: 用户ID × 物品ID 等（Wide部分使用）
    """
    np.random.seed(random_state)

    n_users = 1000
    n_items = 500
    n_categories = 20

    data = {
        # 连续特征
        'user_age': np.random.randint(18, 65, n_samples),
        'user_income': np.random.normal(50000, 15000, n_samples).astype(int),
        'item_price': np.random.normal(100, 30, n_samples),
        'watch_time': np.random.exponential(30, n_samples),
        'click_count': np.random.poisson(5, n_samples),

        # 类别特征
        'user_id': np.random.randint(0, n_users, n_samples),
        'item_id': np.random.randint(0, n_items, n_samples),
        'category': np.random.randint(0, n_categories, n_samples),
        'hour_bucket': np.random.randint(0, 6, n_samples),  # 0-5时段
        'device_type': np.random.randint(0, 4, n_samples),   # 0-3设备类型
        'location': np.random.randint(0, 50, n_samples),      # 0-49地区
    }

    df = pd.DataFrame(data)

    # 生成标签（模拟CTR，加入一些非线性关系和交互效应）
    # 基础概率
    base_prob = 0.1

    # 年龄效应（非线性）
    age_effect = np.exp(-((df['user_age'] - 35) ** 2) / 500) * 0.1

    # 收入效应
    income_effect = (df['user_income'] - 50000) / 100000 * 0.05

    # 价格效应（负相关）
    price_effect = -df['item_price'] / 500 * 0.1

    # 观看时长效应
    watch_effect = np.log1p(df['watch_time']) / 5 * 0.1

    # 类别交互效应（某些类别组合更容易点击）
    category_effect = ((df['category'] == 5) | (df['category'] == 12)).astype(float) * 0.15

    # 时段效应
    hour_effect = (df['hour_bucket'].isin([2, 3])).astype(float) * 0.08

    # 用户-物品交互效应（模拟某些用户对某些物品有偏好）
    user_item_effect = np.sin(df['user_id'] / 100 + df['item_id'] / 50) * 0.05

    # 合并所有效应
    logit = base_prob + age_effect + income_effect + price_effect + \
            watch_effect + category_effect + hour_effect + user_item_effect

    # 加入噪声
    logit += np.random.normal(0, 0.05, n_samples)

    # 转换为概率
    prob = 1 / (1 + np.exp(-logit * 10))

    # 生成标签
    df['label'] = (np.random.random(n_samples) < prob).astype(float)

    print(f"数据集信息:")
    print(f"  样本数: {n_samples}")
    print(f"  正样本比例: {df['label'].mean():.3f}")
    print(f"\n前5行数据:")
    print(df.head())

    return df, n_users, n_items, n_categories


class RecommendationDataset(Dataset):
    """PyTorch Dataset for recommendation data"""

    def __init__(self, df, dense_cols, category_cols, wide_cols=None):
        self.dense_features = torch.FloatTensor(df[dense_cols].values)
        self.category_features = torch.LongTensor(df[category_cols].values)
        self.labels = torch.FloatTensor(df['label'].values).unsqueeze(1)

        if wide_cols is not None:
            self.wide_features = torch.FloatTensor(df[wide_cols].values)
        else:
            self.wide_features = None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.wide_features is not None:
            return (self.dense_features[idx],
                    self.category_features[idx],
                    self.wide_features[idx],
                    self.labels[idx])
        else:
            return (self.dense_features[idx],
                    self.category_features[idx],
                    self.labels[idx])


# ============================================================
# 3. 构建交叉特征（Wide部分）
# ============================================================

def create_cross_features(df, category_cols, max_combinations=1000):
    """
    创建交叉特征用于Wide部分
    使用哈希技巧处理高维交叉特征
    """
    from sklearn.feature_extraction import FeatureHasher

    # 构建交叉特征字符串
    cross_strs = []
    for _, row in df[category_cols].iterrows():
        # 创建两两交叉特征
        crosses = []
        for i in range(len(category_cols)):
            for j in range(i+1, len(category_cols)):
                cross = f"{category_cols[i]}={row[category_cols[i]]}&{category_cols[j]}={row[category_cols[j]]}"
                crosses.append(cross)
        cross_strs.append(' '.join(crosses))

    # 使用FeatureHasher进行哈希编码
    hasher = FeatureHasher(n_features=max_combinations, input_type='string')
    wide_features = hasher.transform(cross_strs).toarray()

    wide_df = pd.DataFrame(
        wide_features,
        columns=[f'wide_{i}' for i in range(max_combinations)]
    )

    return wide_df


# ============================================================
# 4. 训练流程
# ============================================================

def train_wide_and_deep(model, train_loader, val_loader, epochs=20, lr=0.001, device='cpu'):
    """训练Wide & Deep模型"""

    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_val_auc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_auc': []}

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_losses = []

        for batch in train_loader:
            if len(batch) == 4:
                dense, category, wide, labels = batch
                wide = wide.to(device)
            else:
                dense, category, labels = batch
                wide = None

            dense = dense.to(device)
            category = category.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(dense, category, wide)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # 验证阶段
        model.eval()
        val_losses = []
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 4:
                    dense, category, wide, labels = batch
                    wide = wide.to(device)
                else:
                    dense, category, labels = batch
                    wide = None

                dense = dense.to(device)
                category = category.to(device)
                labels = labels.to(device)

                outputs = model(dense, category, wide)
                loss = criterion(outputs, labels)

                val_losses.append(loss.item())
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算AUC
        from sklearn.metrics import roc_auc_score
        val_auc = roc_auc_score(all_labels, all_preds)

        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)

        if val_auc > best_val_auc:
            best_val_auc = val_auc

        scheduler.step()

        if (epoch + 1) % 2 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f}")

    print(f"\n最佳验证AUC: {best_val_auc:.4f}")
    return history


# ============================================================
# 5. 主程序：完整示例
# ============================================================

print("=" * 60)
print("Wide & Deep Learning for Recommender Systems")
print("=" * 60)

# 1. 创建数据
print("\n【步骤1】生成模拟数据...")
df, n_users, n_items, n_categories = create_synthetic_data(n_samples=20000)

# 2. 定义特征列
DENSE_COLS = ['user_age', 'user_income', 'item_price', 'watch_time', 'click_count']
CATEGORY_COLS = ['user_id', 'item_id', 'category', 'hour_bucket', 'device_type', 'location']

# 3. 标准化连续特征
print("\n【步骤2】特征预处理...")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[DENSE_COLS] = scaler.fit_transform(df[DENSE_COLS])

# 4. 构建交叉特征（Wide部分）
print("\n【步骤3】构建交叉特征（Wide部分）...")
wide_df = create_cross_features(df, CATEGORY_COLS[:4], max_combinations=500)
wide_cols = list(wide_df.columns)
df = pd.concat([df, wide_df], axis=1)

# 5. 划分训练集和验证集
print("\n【步骤4】划分数据集...")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

# 类别特征的取值数量
category_sizes = [df[col].nunique() for col in CATEGORY_COLS]
print(f"类别特征取值数量: {category_sizes}")

# 6. 创建Dataset和DataLoader
train_dataset = RecommendationDataset(train_df, DENSE_COLS, CATEGORY_COLS, wide_cols)
val_dataset = RecommendationDataset(val_df, DENSE_COLS, CATEGORY_COLS, wide_cols)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

# 7. 初始化模型
print("\n【步骤5】初始化Wide & Deep模型...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {device}")

model = WideAndDeep(
    num_dense_features=len(DENSE_COLS),
    num_categories=len(CATEGORY_COLS),
    category_sizes=category_sizes,
    embedding_dim=16,
    deep_hidden_units=[256, 128, 64],
    wide_input_dim=len(wide_cols),
    dropout_rate=0.3
)

print(f"\n模型结构:")
print(model)
print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

# 8. 训练模型
print("\n【步骤6】开始训练...")
history = train_wide_and_deep(model, train_loader, val_loader, epochs=15, lr=0.001, device=device)

# 9. 可视化训练过程
print("\n【步骤7】绘制训练曲线...")
