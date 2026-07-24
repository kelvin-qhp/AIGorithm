import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# 定义 Wide & Deep 模型
class WideDeepModel(nn.Module):
    def __init__(self, num_continuous,  c, embedding_dims, hidden_dims,drop_out=0.3):
        super(WideDeepModel, self).__init__()

        # 为分类变量定义 embedding 层
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_categories[i], embedding_dims[i]) for i in range(len(num_categories))])

        # Wide 部分 (线性层)
        self.linear = nn.Linear(num_continuous + sum(embedding_dims), 1)

        # Deep 部分 (多层感知器，用于处理连续特征和嵌入后的分类特征)
        deep_input_dim = num_continuous + sum(embedding_dims)
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(deep_input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(drop_out))
            deep_input_dim = hidden_dim
        layers.append(nn.Linear(deep_input_dim, 1))  # 输出层
        self.deep = nn.Sequential(*layers)

    def forward(self, continuous, categorical):
        # 对分类变量进行 embedding 查找
        embedded = [self.embeddings[i](categorical[:, i]) for i in range(categorical.shape[1])]
        embedded = torch.cat(embedded, dim=1)

        # 将连续特征和 embedding 特征拼接
        x = torch.cat([continuous, embedded], dim=1)

        # Wide 部分 (线性层)
        wide_output = self.linear(x)

        # Deep 部分
        deep_output = self.deep(x)

        # 结合 Wide 和 Deep 部分的输出 (用于二分类任务)
        output = torch.sigmoid(wide_output + deep_output)  # 二分类任务使用 sigmoid 激活函数
        return output


# 训练和验证模型的函数
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    model = model.to(device)
    best_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0

        # 训练循环
        for continuous, categorical, labels in train_loader:
            continuous = continuous.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(continuous, categorical)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            preds = (outputs.squeeze() > 0.5).long()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / total_preds
        epoch_acc = correct_preds / total_preds
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc)

        # 验证循环
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        # 保存表现最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_wide_deep_model.pth')

    return train_losses, train_accs, val_losses, val_accs


# 验证模型的函数
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0

    with torch.no_grad():
        for continuous, categorical, labels in val_loader:
            continuous = continuous.to(device)
            categorical = categorical.to(device)
            labels = labels.to(device)

            outputs = model(continuous, categorical)
            loss = criterion(outputs.squeeze(), labels.float())
            running_loss += loss.item() * labels.size(0)
            preds = (outputs.squeeze() > 0.5).long()
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

    val_loss = running_loss / total_preds
    val_acc = correct_preds / total_preds
    return val_loss, val_acc


# 单个样本预测的函数
def predict_single_sample(model, continuous, categorical, device):
    model.eval()
    with torch.no_grad():
        continuous = torch.tensor(continuous, dtype=torch.float32).unsqueeze(0).to(device)
        categorical = torch.tensor(categorical, dtype=torch.long).unsqueeze(0).to(device)
        output = model(continuous, categorical)
        return (output.item() > 0.5)  # 返回二分类预测结果


# 可视化训练和验证损失、准确率
def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Train Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'bo-', label='Train Accuracy')
    plt.plot(epochs, val_accs, 'ro-', label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()


# 参数定义
num_continuous = 5  # 连续特征的数量
num_categories = [10, 20, 15]  # 每个离散特征的分类数量
embedding_dims = [4, 5, 3]  # 每个分类特征的embedding维度
hidden_dims = [64, 32]  # 深层模型的隐藏层维度
num_samples = 1000  # 样本数量

# 生成连续变量（使用正态分布随机生成）
continuous_data = np.random.randn(num_samples, num_continuous)

# 生成离散分类变量
# 根据num_categories中的值，为每个分类特征生成样本
categorical_data = np.column_stack([np.random.randint(0, cat, num_samples) for cat in num_categories])

# field_idx = [i for i in range(len(num_categories))]
# for i in field_idx:
#     lbe = LabelEncoder()
#     categorical_data[:,i] = lbe.fit_transform(categorical_data[:,i])
# mms = MinMaxScaler(feature_range=(0, 1))
# categorical_data[:,field_idx] = mms.fit_transform(categorical_data[:,field_idx])


# 生成二分类标签 (0 或 1)
labels = np.random.randint(0, 2, num_samples)

# 将numpy数据转换为torch.Tensor，以便在PyTorch模型中使用
continuous_data = torch.tensor(continuous_data, dtype=torch.float32)
categorical_data = torch.tensor(categorical_data, dtype=torch.long)
labels = torch.tensor(labels, dtype=torch.long)

# 检查生成的数据形状
print("连续特征数据形状:", continuous_data.shape)
print("分类特征数据形状:", categorical_data.shape)
print("标签数据形状:", labels.shape)

# 划分训练集和验证集
train_continuous, val_continuous, train_categorical, val_categorical, train_labels, val_labels = train_test_split(
    continuous_data.numpy(), categorical_data.numpy(), labels.numpy(), test_size=0.2, random_state=42,stratify=labels.numpy())

# 创建 DataLoader
batch_size = 32
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_continuous, dtype=torch.float32),
                                               torch.tensor(train_categorical, dtype=torch.long),
                                               torch.tensor(train_labels, dtype=torch.long))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_continuous, dtype=torch.float32),
                                             torch.tensor(val_categorical, dtype=torch.long),
                                             torch.tensor(val_labels, dtype=torch.long))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 模型、损失函数和优化器
model = WideDeepModel(num_continuous, num_categories, embedding_dims, hidden_dims)
criterion = nn.BCELoss()  # 二分类任务使用 Binary Cross Entropy 损失
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

# 判断是否有GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 训练模型
num_epochs = 10
train_losses, train_accs, val_losses, val_accs = train_model(model, train_loader, val_loader, num_epochs, criterion,
                                                             optimizer, device)

# 加载表现最好的模型
model.load_state_dict(torch.load('best_wide_deep_model.pth'))

# 对单个样本进行预测
sample_continuous = [0.5, -0.2, 0.1, 0.7, -1.2]  # 示例连续数据
sample_categorical = [3, 5, 2]  # 示例分类数据
prediction = predict_single_sample(model, sample_continuous, sample_categorical, device)
print("单个样本的预测类别:", prediction)

# 可视化训练过程
plot_training_curves(train_losses, val_losses, train_accs, val_accs)