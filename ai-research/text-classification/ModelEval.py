from transformers import AutoModelForSequenceClassification,BertForSequenceClassification,DistilBertForSequenceClassification,AutoTokenizer,DataCollatorWithPadding,TrainingArguments,Trainer,pipeline,DistilBertModel,DistilBertConfig
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F

MAX_LENGTH = 128
MODEL_BASE_PATH = "../model-base/bert-base-uncased"
MODEL_SAVE_PATH = "../model-fine-tuning/bert-base-uncased/"
DATA_OUTPUT_PATH = "../data/output/product/product2.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_PATH)


df = pd.read_csv(DATA_OUTPUT_PATH)

category_names = df['category'].unique()
id2label = {idx:name for idx,name in enumerate(category_names)}
label2id = {label: idx for idx,label in id2label.items()}

model_best = AutoModelForSequenceClassification.from_pretrained(MODEL_BASE_PATH,num_labels=len(id2label),id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True)
model_best.load_state_dict(torch.load(MODEL_SAVE_PATH+"state_dict.pth",map_location=device))
cls_pipeline = pipeline("text-classification",model=model_best,tokenizer=tokenizer)


val_df = df[1000:1020]
print(f"df size:{df.shape}")
res = cls_pipeline(list(val_df['category']))
print("****Model predict result 1:",res)


inputs = tokenizer(list(val_df['category']) ,max_length=MAX_LENGTH,padding="max_length",truncation=True,return_tensors="pt")
with torch.no_grad():
    outputs = model_best(**inputs)
    scores = F.softmax(outputs.logits,dim=-1)
    probs = F.softmax(scores,dim=-1)
    probs_cls = torch.argmax(probs,dim=-1)
    print(f'***predict result cls is:{[id2label.get(int(c)) for c in probs_cls]}')
    # print(f'*****output:{outputs}')
    # print(f'=====last_hidden_state.size:{outputs.hidden_state.size()},**[:,0]:{outputs.hidden_state[:,0]}')
# print(outputs)
