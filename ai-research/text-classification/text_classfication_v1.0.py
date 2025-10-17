import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from torch.utils.data import Dataset,DataLoader,random_split
import pandas as pd
from torch.optim import Adam


BATCH_SIZE = 32
MAX_LENGTH = 128
EPOCH = 3
LOG_STEP = 100

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
optimizer = Adam(model.parameters(),lr=2e-5)

class MyDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = pd.read_csv("../data/input/text_classfication/ChnSentiCorp_htl_all.csv")
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"],self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)

def coll_fn( batch ):
    texts,labels = [],[]
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    input = tokenizer(texts,max_length=MAX_LENGTH,padding="max_length",truncation=True,return_tensors="pt")
    input['labels'] = torch.tensor(labels)
    return  input

def eval():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validDataLoader:
            output = model(**batch)
            pred = torch.argmax(output.logits,dim=-1)
            acc_num += (pred.long() == batch['labels'].long()).float().sum()

    return acc_num / len(validDataset)

def train():
    global_step = 0
    for epoch in range(EPOCH):
        model.train()
        for batch in trainDataLoader:
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % LOG_STEP == 0:
                print(f"ep:{epoch},global step:{global_step},loss:{output.loss.item()}")
            global_step +=1
        acc = eval()
        print(f"*****ep:{epoch},acc:{acc}")

if __name__ == '__main__':
    dataset = MyDataset()
    trainDataset, validDataset = random_split(dataset,lengths=[0.9,0.1])
    # for i in range(5):
    #     print(trainDataset[i])

    trainDataLoader = DataLoader(trainDataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=coll_fn)
    validDataLoader = DataLoader(validDataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=coll_fn)
    # print(next(enumerate(trainDataLoader))[1])

    train()