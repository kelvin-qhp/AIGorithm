from pydantic.v1.utils import get_model
from transformers import AutoModelForSequenceClassification,BertForSequenceClassification,DistilBertForSequenceClassification,AutoTokenizer,AutoModel,DataCollatorWithPadding,TrainingArguments,Trainer,pipeline,DistilBertModel,DistilBertConfig
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from datasets import Dataset

MAX_LENGTH = 128
MODEL_BASE_PATH = "../model-base/bert-base-uncased"
MODEL_SAVE_PATH = "../model-fine-tuning/bert-base-uncased/"
DATA_OUTPUT_PATH = "../data/output/product/product2.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_BASE_PATH)

# 0. Prepare for dataset
df = pd.read_csv(DATA_OUTPUT_PATH)

category_names = df['category'].unique()
id2label = {idx:name for idx,name in enumerate(category_names)}
label2id = {label: idx for idx,label in id2label.items()}
# print(id2label)
def getModel():
    model_best = AutoModelForSequenceClassification.from_pretrained(MODEL_BASE_PATH,num_labels=len(id2label),id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True).to(device)
    model_best.load_state_dict(torch.load(MODEL_SAVE_PATH+"state_dict.pth",map_location=device))
    return model_best

def getModel2():
    model_best = AutoModel.from_pretrained(MODEL_BASE_PATH,num_labels=len(id2label),id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True)
    model_best.load_state_dict(torch.load(MODEL_SAVE_PATH+"state_dict.pth",map_location=device))
    return model_best

def getModel3():
    model_best = torch.load(MODEL_SAVE_PATH+"bert-base-uncased.pt",map_location=device,weights_only=False)
    return model_best

def getModel4():
    model_best = AutoModel.from_pretrained(MODEL_BASE_PATH,num_labels=len(id2label),id2label=id2label,label2id=label2id,ignore_mismatched_sizes=True)
    return model_best

def getPipeline():

    cls_pipeline = pipeline("text-classification",model=best_model,tokenizer=tokenizer)
    return cls_pipeline

def data_process(examples):
    tokenized_examples = tokenizer(examples['name'],max_length=MAX_LENGTH,padding="max_length",truncation=True)

    cat_ids = [label2id.get(catName) for catName in examples['category']]
    tokenized_examples['labels'] = cat_ids
    return tokenized_examples
def data_process2(examples):
    tokenized_examples = tokenizer(examples['name'],max_length=MAX_LENGTH,padding="max_length",truncation=True,return_tensors="pt")

    cat_ids = [label2id.get(catName) for catName in examples['category']]
    tokenized_examples['labels'] = cat_ids
    return tokenized_examples
def forward_pass_with_label(batch):
    # Place all input tensors on the same device as the model
    inputs = {k:v.to(device) for k,v in batch.items()
              if k in tokenizer.model_input_names}

    with torch.no_grad():
        output = best_model(**inputs)
        pred_label = torch.argmax(output.logits, axis=-1)
        loss = cross_entropy(output.logits, batch["labels"].to(device), reduction="none")
    # Place outputs on CPU for compatibility with other dataset columns
    return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}

def label_int2str(row):
    return id2label.get(row)


best_model = getModel()

if __name__ == '__main__':

    #显示所有列
    pd.set_option('display.max_columns', None)
    #显示所有行
    pd.set_option('display.max_rows', None)
    #设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth',1000)

    val_df = df[1000:1010]
    print(f"df size:{df.shape}")

    # 1. Base for pipeline
    my_pipeline = getPipeline()
    res  =my_pipeline(list(val_df['category']))
    print("****Model predict result 1:",res)

    # 2. Base for base model
    inputs = tokenizer(list(val_df['name']) ,max_length=MAX_LENGTH,padding="max_length",truncation=True,return_tensors="pt")
    val_df["labels"] = val_df["category"].apply(lambda x:label2id.get(x))
    with torch.no_grad():
        # my_model = getModel()
        outputs = best_model(**inputs)
        scores = F.softmax(outputs.logits,dim=-1)
        probs = F.softmax(scores,dim=-1)
        probs_cls = torch.argmax(probs,dim=-1)
        loss = cross_entropy(outputs.logits, torch.tensor(list(val_df["labels"])), reduction="none")
        print(f'***Model predict result 2:{[id2label.get(int(c)) for c in probs_cls]}')
        print(f'***loss is:{loss}')
        # print(f'*****output:{outputs}')
        # print(f'=====last_hidden_state.size:{outputs.hidden_state.size()},**[:,0]:{outputs.hidden_state[:,0]}')
    # print(outputs)

    # 3. review loss
    dataset = Dataset.from_pandas(val_df)
    dataset = dataset.train_test_split(test_size=0.1)
    columns = ['sku', 'type', 'price', 'upc', 'shipping', 'description', 'manufacturer', 'model', 'url', 'image']
    tokenized_dataset = dataset.map(data_process,batched=True,remove_columns=columns)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask","token_type_ids", "labels"])
    tokenized_dataset['test'] = tokenized_dataset['test'].map(forward_pass_with_label,batched=True, batch_size=16)

    # Convert dataset to pandas dataframe
    tokenized_dataset.set_format(type="pandas")
    validate_df =  tokenized_dataset['test'].to_pandas()
    # Delete useless columns in dataframe
    validate_df.drop(['input_ids', 'attention_mask','token_type_ids'], axis=1, inplace=True)
    validate_df["predicted_label2"] = validate_df["predicted_label"].apply(label_int2str)
    # print(validate_df[validate_df['labels'] == 11].sort_values("loss", ascending=False).head(10))
    print(validate_df.sort_values("loss", ascending=True).head(10))
    print("*"*100)
    print(validate_df.sort_values("loss", ascending=False).head(10))
