import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding
from torch.utils.data import Dataset,DataLoader,random_split
import pandas as pd
from torch.optim import Adam
from datasets import load_dataset
import evaluate

BATCH_SIZE = 32
MAX_LENGTH = 128
EPOCH = 3
LOG_STEP = 100

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
optimizer = Adam(model.parameters(),lr=2e-5)
clf_metrics = evaluate.combine(["f1","accuracy"])

def eval():
    model.eval()
    with torch.inference_mode():
        for batch in validDataLoader:
            output = model(**batch)
            pred = torch.argmax(output.logits,dim=-1)
            clf_metrics.add_batch(predictions=pred.long(),references=batch['labels'].long())
    return clf_metrics.compute()

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
        print(f"*****ep:{epoch},{acc}")

def data_process(examples):
    tokenized_examples = tokenizer(examples['review'],max_length=MAX_LENGTH,padding="max_length",truncation=True)
    tokenized_examples['labels'] = examples['label']
    return tokenized_examples

if __name__ == '__main__':
    # clf_metrics = clf_metrics.compute(predictions=[0,1,0],references=[1,1,0])
    # print(clf_metrics)

    dataset = load_dataset("csv",data_files="./data/input/text_classfication/ChnSentiCorp_htl_all.csv",split="train")
    dataset = dataset.filter(lambda x:x['review'] is not None)
    datasets = dataset.train_test_split(test_size=0.1)
    # print(datasets)

    tokenized_dataset = datasets.map(data_process,batched=True,remove_columns=dataset.column_names)
    # print(tokenized_dataset)

    trainDataset,validDataset = tokenized_dataset['train'],tokenized_dataset['test']
    trainDataLoader = DataLoader(trainDataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=DataCollatorWithPadding(tokenizer))
    validDataLoader = DataLoader(validDataset,batch_size=BATCH_SIZE,shuffle=True,collate_fn=DataCollatorWithPadding(tokenizer))
    # print(next(enumerate(validDataLoader))[1])

    train()