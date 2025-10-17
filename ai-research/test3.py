import torch
import torch.nn as nn
import torch.nn.functional as F
from d2l import torch as d2l


# 设置随机种子确保可重复性
torch.manual_seed(42)

def init_xavier(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
def init_42(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 42)
def init_constant(m):
    if type(m) == nn.Linear:
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)

class MyLinear(nn.Module):
    def __init__(self, in_units, units):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(in_units, units))
        self.bias = nn.Parameter(torch.randn(units,))
    def forward(self, X):
        linear = torch.matmul(X, self.weight.data) + self.bias.data
        return F.relu(linear)

# 多次创建相同的线性层
linear1 = nn.Linear(5, 3)
linear2 = nn.Linear(5, 3)

linear1.apply(init_xavier)
linear2.apply(init_xavier)
# linear1.apply(init_42)
# linear2.apply(init_42)
print(f"权重是否相同:{torch.allclose(linear1.weight, linear2.weight)} -----model1:{linear1.weight},model2:{linear2.weight}")
print("偏置是否相同:", torch.allclose(linear1.bias, linear2.bias))


linear1.apply(init_constant)
linear2.apply(init_constant)

print(f"权重是否相同:{torch.allclose(linear1.weight, linear2.weight)} ")
print("偏置是否相同:", torch.allclose(linear1.bias, linear2.bias))


linear = MyLinear(5, 3)
print(type(linear))
for k,v in linear.named_parameters():
    print(f"-----------key:{k},value:{v}")



T = 1000  # 总共产生1000个点
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))


A = torch.rand(3, 4)
B = torch.rand(4, 5)

C1 = torch.mm(A, B)
C2 = A @ B                # 推荐用法
C3 = torch.matmul(A, B)   # 推荐用法

print(f"c1:{C1},c2:{C2},c3:{C3}")

matrix = torch.zeros(2, 3,1, 4)

n = torch.squeeze(matrix)
print(f"defaul sequeeze for matrix:{n.shape}")

n2 = torch.squeeze(matrix, 1)

print(f"defaul sequeeze 1 for matrix:{n2.shape}")
