import torch

# 假设有一个概率分布
probs = torch.tensor([0.1, 0.2, 0.7])

# 找到最大概率的索引
max_idx = torch.argmax(probs)
print(f"Max index: {max_idx}")  # 输出最大概率的索引

# 对于多维张量，可以指定dim参数
probs_2d = torch.tensor([[0.1, 1.2,0.1], [0.3, 0.9,1.0]])
max_idx_col = torch.argmax(probs_2d, dim=0)  # 在列上寻找最大值的索引
max_idx_row = torch.argmax(probs_2d, dim=1)  # 在行上寻找最大值的索引
print(f"Max indices in columns: {max_idx_col}")
print(f"Max indices in rows: {max_idx_row}")



print("*"*20,"softmax:");

# 假设有一些原始分数
logits = torch.tensor([2.0, 1.0, 0.1])

# 应用softmax函数
probs = torch.softmax(logits, dim=0)
print(f"Probabilities: {probs}")

# 对于多维张量，softmax通常在最后一个维度上应用
logits_2d = torch.tensor([[2.0, 1.0], [0.1, 3.0]])
probs_2d = torch.softmax(logits_2d, dim=1)  # 在每个行（类别）上应用softmax
print(f"Probabilities in 2D: {probs_2d}")