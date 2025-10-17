import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

class BertNERModel(nn.Module):
    def __init__(self, bert_model_name, num_labels, dropout_prob=0.1):
        """
        基于BERT的NER模型

        参数:
            bert_model_name: 预训练BERT模型名称 (如 'bert-base-cased')
            num_labels: 标签数量(包括'O'标签)
            dropout_prob: dropout率
        """
        super(BertNERModel, self).__init__()

        # 加载预训练BERT模型
        self.bert = BertModel.from_pretrained(bert_model_name,cache_dir='../model_base')
        self.dropout = nn.Dropout(dropout_prob)

        # 分类层
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

        # 损失函数 (忽略索引-100，这是transformers库默认的ignore_index)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        前向传播

        参数:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码 (1表示真实token, 0表示padding)
            labels: 实体标签 (如果有)

        返回:
            logits: 每个token的分类logits
            loss: 计算得到的损失 (如果有labels)
        """
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 获取序列输出
        sequence_output = outputs.last_hidden_state

        # 应用dropout
        sequence_output = self.dropout(sequence_output)

        # 分类
        logits = self.classifier(sequence_output)

        # 计算损失
        loss = None
        if labels is not None:
            # 调整形状以计算损失
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            loss = self.criterion(logits_flat, labels_flat)

        return logits, loss

    def predict(self, input_ids, attention_mask):
        """预测方法"""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
        return preds


# 数据集类
class BertNERDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, tag2idx, max_len=128):
        """
        初始化数据集

        参数:
            sentences: 句子列表 (每个句子是token列表)
            labels: 对应的标签列表
            tokenizer: BERT tokenizer
            tag2idx: 标签到索引的映射
            max_len: 最大序列长度
        """
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.tag2idx = tag2idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Tokenize句子并转换为IDs
        encoding = self.tokenizer(
            sentence,
            is_split_into_words=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )

        # 处理WordPiece tokenization带来的标签对齐问题
        word_ids = encoding.word_ids()
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # 特殊token([CLS], [SEP], [PAD])设置为-100
            if word_idx is None:
                label_ids.append(-100)
            # 同一个word的不同piece使用相同的标签
            elif word_idx != previous_word_idx:
                label_ids.append(self.tag2idx[label[word_idx]])
            else:
                label_ids.append(-100)  # 或者使用self.tag2idx[label[word_idx]]

            previous_word_idx = word_idx

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_ids, dtype=torch.long)
        }


# 训练函数
def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs=5):
    """
    训练模型

    参数:
        model: 要训练的模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        optimizer: 优化器
        scheduler: 学习率调度器
        device: 训练设备
        epochs: 训练轮数
    """
    best_val_loss = float('inf')

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        # 训练阶段
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc='Training'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            _, loss = model(input_ids, attention_mask, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f'Train loss: {avg_train_loss:.4f}')

        # 验证阶段
        val_loss, val_report = evaluate(model, val_loader, device)
        print(f'Validation loss: {val_loss:.4f}')
        print('Validation Report:')
        print(val_report)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_bert_ner_model.pt')
            print('Saved best model!')


# 评估函数
def evaluate(model, data_loader, device):
    """
    评估模型

    参数:
        model: 要评估的模型
        data_loader: 数据加载器
        device: 评估设备

    返回:
        avg_loss: 平均损失
        report: 分类报告
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits, loss = model(input_ids, attention_mask, labels)

            total_loss += loss.item()

            # 获取预测结果
            preds = torch.argmax(logits, dim=-1)

            # 收集非特殊token的预测和标签
            mask = labels != -100
            all_preds.extend(preds[mask].cpu().numpy())
            all_labels.extend(labels[mask].cpu().numpy())

    avg_loss = total_loss / len(data_loader)

    # 生成分类报告
    report = classification_report(
        all_labels,
        all_preds,
        zero_division=0
    )

    return avg_loss, report


# 示例使用
if __name__ == "__main__":
    # 示例数据 (实际应用中应该从文件加载)
    train_sentences = [
        ["John", "lives", "in", "New", "York", "."],
        ["Apple", "is", "based", "in", "Cupertino", "."],
        ["I", "like", "Python", "programming", "."]
    ]
    train_labels = [
        ["B-PER", "O", "O", "B-LOC", "I-LOC", "O"],
        ["B-ORG", "O", "O", "O", "B-LOC", "O"],
        ["O", "O", "O", "O", "O"]
    ]

    # 标签映射
    tag2idx = {
        'O': 0,
        'B-PER': 1,
        'I-PER': 2,
        'B-ORG': 3,
        'I-ORG': 4,
        'B-LOC': 5,
        'I-LOC': 6
    }

    # 初始化BERT tokenizer
    bert_model_name = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)

    encoding = tokenizer(
        train_sentences[0],
        is_split_into_words=True,
        padding='max_length',
        truncation=True,
        max_length=24,
        return_tensors='pt'
    )

    # 处理WordPiece tokenization带来的标签对齐问题
    word_ids = encoding.word_ids()

    # 创建数据集
    train_dataset = BertNERDataset(train_sentences, train_labels, tokenizer, tag2idx, max_len=20)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BertNERModel(bert_model_name, num_labels=len(tag2idx)).to(device)

    # 设置优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * 5  # epochs=5
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # 训练模型 (这里用相同的loader作为验证集仅作示例)
    train(model, train_loader, train_loader, optimizer, scheduler, device, epochs=3)

    # 测试预测
    test_sentence = ["Microsoft", "is", "located", "in", "Redmond"]
    test_encoding = tokenizer(
        test_sentence,
        is_split_into_words=True,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=20
    )

    input_ids = test_encoding['input_ids'].to(device)
    attention_mask = test_encoding['attention_mask'].to(device)

    preds = model.predict(input_ids, attention_mask)

    # 将预测转换为标签
    idx2tag = {v: k for k, v in tag2idx.items()}
    pred_labels = [idx2tag.get(idx, 'O') for idx in preds[0].cpu().numpy() if idx != -100]

    print("\nTest Sentence:", test_sentence)
    print("Predictions:", pred_labels)