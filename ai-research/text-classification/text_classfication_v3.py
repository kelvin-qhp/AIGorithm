import torch
from numpy.ma.extras import average
from transformers import AutoModelForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,TrainingArguments,Trainer,pipeline
from torch.utils.data import DataLoader,random_split
import pandas as pd
import numpy as np
from torch.optim import Adam
from datasets import load_dataset
from datasets import *
import evaluate
import json
import matplotlib.pyplot as plt
from test2 import num_cls
from torch.optim import Adam
import evaluate

"""
    reference below url to fine-ture:
    https://cloud.tencent.com.cn/developer/article/2317754
"""


BATCH_SIZE = 32
MAX_LENGTH = 128
EPOCH = 3
LOG_STEP = 100
# MODEL_NAME = "distilbert-base-uncased-sst2en"
MODEL_BASE_PATH = "../base-model/distilbert-base-uncased"
MODEL_SAVE_PATH = "../model/distilbert-base-uncased/"
DATA_OUTPUT_PATH = "../data/output/product/product.csv"

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_PATH)
# model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
acc_metrics = evaluate.combine(["accuracy"])
f1_metrics = evaluate.combine(["f1"])


def load_mydataset():
    file = open('../data/input/product/products.json', encoding='utf-8')
    # load_dataset("json",data_files="./data/input/product/products.json",split="train")
    my_dataset = json.load(file)
    # my_dataset = load_dataset("json",data_files="./data/input/product/products.json",split="train")
    return my_dataset

def data_process(examples):
    tokenized_examples = tokenizer(examples['name'],max_length=MAX_LENGTH,padding="max_length",truncation=True)

    cat_ids = [label2id.get(catName) for catName in examples['category']]
    tokenized_examples['labels'] = cat_ids
    return tokenized_examples

def category_process(row):
    cats = row['category']
    # productCategories = []
    for cat in cats:
        # productCategories.append(cat['name'])
        row['category'] = cat['name']
        break
    # row['category'] = productCategories
    # print(row)
    return row

def data_precoess():
    my_dataset = load_mydataset()
    # print(type(my_dataset))
    dataset_df = pd.DataFrame(my_dataset)

    dataset_df.loc[dataset_df['shipping'] == '','shipping'] = 0
    print("before clean data size:",dataset_df.shape)
    dataset_df.dropna()
    dataset_df = dataset_df.dropna(subset=['name'])
    print("after clean data size:",dataset_df.shape)

    dataset_df = dataset_df.apply(category_process, axis=1)

    dataset_df.to_csv(DATA_OUTPUT_PATH,index=False)

def drawCategory(dataset_df):
    dataset_df["category"].value_counts(ascending=True).plot.barh()
    plt.title("Frequency of Classes")
    plt.show()

def drawCategoryLen(df):
    df["name_len"] = df["name"].str.split().apply(len) # 按空格切分，获取雷彪长度
    df.boxplot("name_len", by="category", grid=False, showfliers=False,color="black")
    plt.suptitle("Product length per Category")
    plt.xlabel("Category",)
    plt.show()


def eval_metric(eval_predict):
    predictions,labels = eval_predict
    predictions = predictions.argmax(axis = -1)
    acc = acc_metrics.compute(predictions= predictions,references=labels)
    f1 = f1_metrics.compute(predictions=predictions,references=labels, average="weighted")
    acc.update(f1)
    return acc

def saveModel(model):
    torch.save(model.state_dict(),MODEL_SAVE_PATH+"state_dict.pth")
    torch.save(model,MODEL_SAVE_PATH+"m.pt")

if __name__ == '__main__':

    # #显示所有列
    # pd.set_option('display.max_columns', None)
    # #显示所有行
    # pd.set_option('display.max_rows', None)
    # #设置value的显示长度为100，默认为50
    # pd.set_option('max_colwidth',400)

    # mm_ds = pd.read_json("./data/input/product/products.json")
    # my_dataset = load_dataset("json", data_files="./data/input/product/products.json")
    # my_dataset2 = Dataset.from_list(my_dataset)

    # data_precoess()

    # drawCategory(dataset_df)
    # drawCategoryLen(dataset_df)

    dataset_df = pd.read_csv(DATA_OUTPUT_PATH)
    dataset_df = dataset_df[:1000]
    print(f'df size:{dataset_df.shape}')
    category_names = dataset_df['category'].unique()
    id2label = {idx:name for idx,name in enumerate(category_names)}
    label2id = {label: idx for idx,label in id2label.items()}

    print(f"id2label size:{len(id2label)} for:{id2label}")
    dataset = Dataset.from_pandas(dataset_df)

    dataset = dataset.train_test_split(test_size=0.1)
    print(dataset)
    # train_dataset,test_dataset = dataset['train'],dataset['test']


    # print(dataset.columns)
    tokenized_dataset = dataset.map(data_process,batched=True,remove_columns=dataset['train'].column_names)
    # tokenized_test_dataset = tokenized_dataset.map(data_process,batched=True,remove_columns=train_dataset.column_names)
    print(tokenized_dataset)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_BASE_PATH,num_labels=len(id2label),id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True)
    print(model.config)

    train_args = TrainingArguments(output_dir =MODEL_BASE_PATH+"/checkpoint/",
                                   num_train_epochs=4,
                                   per_device_train_batch_size=32,
                                   per_device_eval_batch_size=64,
                                   logging_steps=10,
                                   eval_strategy="epoch",
                                   save_strategy="epoch",
                                   save_total_limit=3,
                                   disable_tqdm=False,
                                   # eval_steps=5,
                                   learning_rate=2e-5,
                                   weight_decay=0.01,
                                   metric_for_best_model="f1",
                                   load_best_model_at_end=True
                                   )
    print(train_args)

    # load trained model state
    model.load_state_dict(torch.load(MODEL_SAVE_PATH+"state_dict.pth"))
    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=tokenized_dataset['train'],
                      eval_dataset=tokenized_dataset['test'],
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=eval_metric)


    # trainer.train()

    # saveModel(model)

    preds_output  = trainer.predict(tokenized_dataset['test'])
    print(f"*****:{preds_output.metrics}")
    y_preds = np.argmax(preds_output.predictions, axis=1)

    # model_best = torch.load(MODEL_SAVE+"m.pt")
    # # model_best = AutoModelForSequenceClassification.from_pretrained("./model/"+MODEL_NAME,num_labels=len(id2label),id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True)
    # # model_best.load_state_dict(torch.load(MODEL_SAVE+"state_dict.pth"))
    # cls_pipeline = pipeline("text-classification",model=model_best,tokenizer=tokenizer)
    # res = cls_pipeline(["toy bear","apple earphone"])
    # print("Model predict result:",res)

