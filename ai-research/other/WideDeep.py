import torch.nn as nn
import torch.nn.functional as F
import torch


class Linear(nn.Module):
    """
    Linear part
    """

    def __init__(self, input_dim):
        super(Linear, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        return self.linear(x)


class Dnn(nn.Module):
    """
    Dnn part
    """

    def __init__(self, hidden_units, dropout=0.5):
        """
        hidden_units: 列表， 每个元素表示每一层的神经单元个数， 比如[256, 128, 64], 两层网络， 第一层神经单元128， 第二层64， 第一个维度是输入维度
        dropout: 失活率
        """
        super(Dnn, self).__init__()

        self.dnn_network = nn.ModuleList(
            [nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for linear in self.dnn_network:
            x = linear(x)
            x = F.relu(x)

        x = self.dropout(x)
        return x


'''
WideDeep模型：
   主要包括Wide部分和Deep部分
'''
class WideDeep(nn.Module):

    def __init__(self, feature_info, hidden_units, embed_dim=8):
        """
               DeepCrossing：
                   feature_info: 特征信息（数值特征， 类别特征， 类别特征embedding映射)
                   hidden_units: 列表， 隐藏单元
                   dropout: Dropout层的失活比例
                   embed_dim: embedding维度
               """
        super(WideDeep, self).__init__()

        self.dense_features, self.sparse_features, self.sparse_features_map = feature_info

        # embedding层， 这里需要一个列表的形式， 因为每个类别特征都需要embedding
        self.embed_layers = nn.ModuleDict(
            {
                'embed_' + str(key): nn.Embedding(num_embeddings=val, embedding_dim=embed_dim)
                for key, val in self.sparse_features_map.items()
            }
        )


        # 统计embedding_dim的总维度
        # 一个离散型(类别型)变量 通过embedding层变为8纬
        embed_dim_sum = sum([embed_dim] * len(self.sparse_features))
        # 总维度 = 数值型特征的纬度 + 离散型变量经过embedding后的纬度
        dim_sum = len(self.dense_features) + embed_dim_sum
        hidden_units.insert(0, dim_sum)
        # dnn网络
        self.dnn_network = Dnn(hidden_units)

        # 线性层
        self.linear = Linear(input_dim=len(self.dense_features))

        # 最终的线性层
        self.final_linear = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        # 1、先把输入向量x分成两部分处理、因为数值型和类别型的处理方式不一样
        dense_input, sparse_inputs = x[:, :len(self.dense_features)], x[:, len(self.dense_features):]
        # 2、转换为long形
        sparse_inputs = sparse_inputs.long()

        # 2、不同的类别特征分别embedding
        sparse_embeds = [
            self.embed_layers['embed_' + key](sparse_inputs[:, i]) for key, i in
            zip(self.sparse_features_map.keys(), range(sparse_inputs.shape[1]))
        ]
        # 3、把类别型特征进行拼接,即emdedding后，由3行转换为1行
        sparse_embeds = torch.cat(sparse_embeds, axis=-1)
        # 4、数值型和类别型特征进行拼接
        dnn_input = torch.cat([sparse_embeds, dense_input], axis=-1)

        # Wide部分，使用的特征为数值型类型
        wide_out = self.linear(dense_input)

        # Deep部分，使用全部特征
        deep_out = self.dnn_network(dnn_input)

        deep_out = self.final_linear(deep_out)

        # out  将Wide部分的输出和Deep部分的输出进行合并
        outputs = F.sigmoid(0.5 * (wide_out + deep_out))

        return outputs



if __name__ == '__main__':
    x = torch.rand(size=(1, 5), dtype=torch.float32)
    feature_info = [
        ['I1', 'I2'],  # 连续性特征
        ['C1', 'C2', 'C3'],  # 离散型特征
        {
            'C1': 20,
            'C2': 20,
            'C3': 20
        }
    ]

    # 建立模型
    hidden_units = [256, 128, 64]

    net = WideDeep(feature_info, hidden_units)
    print(net)
    print(net(x))
