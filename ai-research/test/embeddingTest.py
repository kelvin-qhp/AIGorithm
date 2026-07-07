import torch

# batch_size=2, seq_len=4, hidden_dim=3
token_embeddings = torch.tensor([
    # 样本 1: 4 个 token，每个 3 维
    [[0.1, 0.2, 0.3],  # token 1
     [0.4, 0.5, 0.6],  # token 2
     [0.7, 0.8, 0.9],  # token 3
     [1.0, 1.1, 1.2]],  # token 4

    # 样本 2: 4 个 token，每个 3 维
    [[0.1, 0.1, 0.1],  # token 1
     [0.2, 0.2, 0.2],  # token 2
     [0.3, 0.3, 0.3],  # token 3
     [0.4, 0.4, 0.4]]  # token 4
])

print(token_embeddings.shape)  # torch.Size([2, 4, 3])

output0 = torch.sum(token_embeddings, dim=0)
print(output0.shape)

# 沿 dim=1 求和（把 4 个 token 向量加起来）
output1 = torch.sum(token_embeddings, dim=1)
print(output1.shape)  # torch.Size([2, 3])

print(output1)

output2 = torch.sum(token_embeddings, dim=2)
print(output2.shape)


output4 = token_embeddings.sum(dim=1)
print(output4.shape)