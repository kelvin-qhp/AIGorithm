from accelerate.commands.config.config_args import cache_dir
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,DataCollatorWithPadding,AutoModel,AutoModel,BertForSequenceClassification
from datasets import load_dataset
import torch.utils.data as Data
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import evaluate


MODEL_NAME ="bert-base-chinese"
MODEL_PATH="./model/bertChinese/"
hidden_size = 768
maxlen = 8
batch_size = 2
epoches = 3
n_class = 2
n_dropout = 0.5

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,cache_dir=MODEL_PATH)
# print(model.config)
# AutoModel.from_pretrained(MODEL_NAME,cache_dir=MODEL_PATH)
# data，构造一些训练数据
sentences = ["我喜欢打篮球", "这个相机很好看", "今天玩的特别开心", "我不喜欢你", "太糟糕了", "真是件令人伤心的事情"]
labels = [1, 1, 1, 0, 0, 0]


class MyDataset(Data.Dataset):
    def __init__(self, sentences, labels=None, with_labels=True,):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.with_labels = with_labels
        self.sentences = sentences
        self.labels = labels
    def __len__(self):
        return len(sentences)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent = self.sentences[index]

        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,       # Truncate to max_length
                                      max_length=maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # tensor of token ids
        attn_masks = encoded_pair['attention_mask'].squeeze(0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)  # binary tensor with "0" for the 1st sentence tokens & "1" for the 2nd sentence tokens

        if self.with_labels:  # True if the dataset has labels
            label = self.labels[index]
            return token_ids, attn_masks, token_type_ids, label
        else:
            return token_ids, attn_masks, token_type_ids

class ChineseClassifyBert(nn.Module):
    def __init__(self):
        super(ChineseClassifyBert, self).__init__()
        self.bert = AutoModel.from_pretrained(MODEL_NAME, output_hidden_states=True, return_dict=True)
        self.linear = nn.Linear(hidden_size, n_class) # 直接用cls向量接全连接层分类
        self.dropout = nn.Dropout(n_dropout)

    def forward(self, X):
        input_ids, attention_mask, token_type_ids = X[0], X[1], X[2]
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids) # 返回一个output字典
        # 用最后一层cls向量做分类
        # outputs.pooler_output: [bs, hidden_size]
        logits = self.linear(self.dropout(outputs.pooler_output))

        return logits


if __name__ == '__main__':
    print('start:')
    my_dataset=MyDataset(sentences, labels)

    train_data = Data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    print(train_data)

    chineseClassifyBert = ChineseClassifyBert()

    optimizer = optim.Adam(chineseClassifyBert.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    train_curve = []

    # train
    sum_loss = 0
    total_step = len(train_data)
    for epoch in range(epoches):
        for i, batch in enumerate(train_data):
            optimizer.zero_grad()
            pred = chineseClassifyBert([batch[0], batch[1], batch[2]])
            loss = loss_fn(pred, batch[3])
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()
        if epoch % 1 == 0:
            print('[{}|{}] step:{}/{} loss:{:.4f}'.format(epoch+1, epoches, i+1, total_step, loss.item()))
        train_curve.append(sum_loss)
        sum_loss = 0
        epoch +=1;

    # test
    chineseClassifyBert.eval()
    with torch.no_grad():
        test_text = ['我不喜欢打篮球','小明打球像姚明一样不错']
        test = MyDataset(test_text, labels=None, with_labels=False)
        x = test.__getitem__(1)
        x = tuple(p.unsqueeze(0) for p in x)
        pred = chineseClassifyBert([x[0], x[1], x[2]])
        pred = pred.data.max(dim=1, keepdim=True)[1]
        if pred[0][0] == 0:
            print('消极')
        else:
            print('积极')

    pd.DataFrame(train_curve).plot() # loss曲线