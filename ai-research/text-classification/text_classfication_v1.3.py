import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,TrainingArguments,Trainer
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
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3",)
optimizer = Adam(model.parameters(),lr=2e-5)
acc_metrics = evaluate.combine(["accuracy"])
f1_metrics = evaluate.combine(["f1"])

def eval_metric(eval_predict):
    predictions,labels = eval_predict
    predictions = predictions.argmax(axis = -1)
    acc = acc_metrics.compute(predictions= predictions,references=labels)
    f1 = f1_metrics.compute(predictions=predictions,references=labels)
    acc.update(f1)
    return acc



def data_process(examples):
    tokenized_examples = tokenizer(examples['review'],max_length=MAX_LENGTH,padding="max_length",truncation=True)
    tokenized_examples['labels'] = examples['label']
    return tokenized_examples

if __name__ == '__main__':

    dataset = load_dataset("csv",data_files="./data/input/text_classfication/ChnSentiCorp_htl_all.csv",split="train")
    dataset = dataset.filter(lambda x:x['review'] is not None)
    datasets = dataset.train_test_split(test_size=0.1)
    # print(datasets)

    tokenized_dataset = datasets.map(data_process,batched=True,remove_columns=dataset.column_names)
    # # print(tokenized_dataset)

    train_args = TrainingArguments(output_dir ="../base-model/text_classify/checkpoint/",
                                   per_device_train_batch_size=32,
                                   per_device_eval_batch_size=64,
                                   logging_steps=10,
                                   eval_strategy="epoch",
                                   save_strategy="epoch",
                                   save_total_limit=3,
                                   # eval_steps=5,
                                   learning_rate=2e-5,
                                   weight_decay=0.01,
                                   metric_for_best_model="f1",
                                   load_best_model_at_end=True
                                   )
    print(train_args)

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset['test'],
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=eval_metric)


    trainer.train()