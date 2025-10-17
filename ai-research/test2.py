import torch
from torch import nn
from torch.nn import functional as F


def f(x):
    return 3 * x ** 2 - 4 * x


def numerical_lim(f, x, h):
    return (f(x + h) - f(x)) / h

# h = 0.1
# for i in range(5):
#     print(f'h={h:.5f}, numerical limit={numerical_lim(f, 1, h):.5f}')
#     h *= 0.1


class MLP(nn.Module):
    # 用模型参数声明层。这里，我们声明两个全连接的层
    def __init__(self,max_length,dim,num_cls):
        # 调用MLP的父类Module的构造函数来执行必要的初始化。
        # 这样，在类实例化时也可以指定其他函数参数，例如模型参数params（稍后将介绍）
        super().__init__()
        self.hidden = nn.Linear(max_length, dim)  # 隐藏层
        self.out = nn.Linear(dim, num_cls)  # 输出层

    # 定义模型的前向传播，即如何根据输入X返回所需的模型输出
    def forward(self, X):
        # 注意，这里我们使用ReLU的函数版本，其在nn.functional模块中定义。
        return self.out(F.relu(self.hidden(X)))

class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            # 这里，module是Module子类的一个实例。我们把它保存在'Module'类的成员
            # 变量_modules中。_module的类型是OrderedDict
            self._modules[str(idx)] = module

    def forward(self, X):
        # OrderedDict保证了按照成员添加的顺序遍历它们
        for block in self._modules.values():
            X = block(X)
        return X

batch_size=5
max_length = 10
embeding_dim = 8
num_cls = 10

torch.manual_seed(42)


net = nn.Sequential(nn.Linear(max_length, embeding_dim), nn.ReLU(), nn.Linear(embeding_dim, num_cls))

x = torch.rand((batch_size, max_length))
print(x)
output = net(x)
print(f"output shape:{output.shape} ***values：{output}")

net2 = MLP(max_length, embeding_dim,num_cls)
output2 = net2(x)
print(f"output2 shape:{output2.shape} ***values：{output2}")

net3 = MySequential(nn.Linear(max_length, embeding_dim), nn.ReLU(), nn.Linear(embeding_dim, num_cls))
output3 =net3(x)
print(f"output3 shape:{output3.shape} ***values：{output3}")



# torch.manual_seed(10)
# # 第一次运行
# linear1 = nn.Linear(5, 1)
# print("第一次初始化的权重:", linear1.weight.detach().numpy())
# print("第一次初始化的偏置:", linear1.bias.detach().numpy())
#
# # 第二次运行
# linear2 = nn.Linear(5, 1)
# print("\n第二次初始化的权重:", linear2.weight.detach().numpy())
# print("第二次初始化的偏置:", linear2.bias.detach().numpy())


#
# for i in range(2):
#     linear_layer = nn.Linear(max_length, embeding_dim)
#     nn.init.ones_(linear_layer.weight)
#     nn.init.zeros_(linear_layer.bias)
#     print(f"****第{i}次初始化的权重:{linear_layer.weight.detach().numpy()},偏置:{linear_layer.bias.detach().numpy()},output:{linear_layer(x)}" )


net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 1))
X = torch.rand(size=(2, 4))
op = net(X)
print("output:",op)
print("****bias:",net[0].bias)
print("****weight:",net[2].weight)

# print(*[(name, param.shape) for name, param in net[0].named_parameters()])
for name, param in net[0].named_parameters():
    print(f"====name:{name}-size:{param.shape}")