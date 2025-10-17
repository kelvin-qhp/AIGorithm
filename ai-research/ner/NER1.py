import evaluate
from accelerate.utils import MODEL_NAME
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification,pipeline
from datasets import load_dataset
import numpy as np
import torch


MODEL_NAME = "hfl/chinese-macbert-base"
MODEL_SAVE = "./model/ner1/"
def loadData():
    ner_datasets = load_dataset("peoples_daily_ner", cache_dir="./data/input/ner",trust_remote_code=True)
    # ner_datasets = load_dataset("peoples_daily_ner", data_dir="./data/input/ner",trust_remote_code=True)
    # ner_datasets = DatasetDict.load_from_disk("ner_data")
    # print(ner_datasets)
    return ner_datasets

def process_function(examples):
    tokenized_exmaples = tokenizer(examples["tokens"], max_length=128, truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_exmaples.word_ids(batch_index=i)
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_id])
        labels.append(label_ids)
    tokenized_exmaples["labels"] = labels
    return tokenized_exmaples

def eval_metric(pred):
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=-1)

    # 将id转换为原始的字符串类型的标签
    true_predictions = [
        [label_list[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    true_labels = [
        [label_list[l] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    result = seqeval.compute(predictions=true_predictions, references=true_labels, mode="strict", scheme="IOB2")

    return {
        "f1": result["overall_f1"]
    }

def saveModel(model):
    torch.save(model.state_dict(),MODEL_SAVE+"state_dict.pth")
    torch.save(model,MODEL_SAVE+"m.pt")

def doTrain():

    args = TrainingArguments(
        output_dir="./model/ner1/ner_v1",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=128,
        eval_strategy="epoch",
        save_strategy="epoch",
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        logging_steps=96,
        num_train_epochs=2,
        save_total_limit=2
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=eval_metric,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer)
    )

    trainer.train()

if __name__ == '__main__':
    ner_datasets = loadData()
    label_list = ner_datasets["train"].features["ner_tags"].feature.names
    print(label_list)
    id2label = {i:j for i, j in enumerate(label_list)}
    label2id = {label: idx for idx,label in id2label.items()}


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(tokenizer(ner_datasets["train"][0]["tokens"], is_split_into_words=True))

    tokenized_datasets = ner_datasets.map(process_function, batched=True)
    print(tokenized_datasets)

    # model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list),id2label=id2label,label2id=label2id)
    # model.load_state_dict(torch.load(MODEL_SAVE+"state_dict.pth"))
    model = torch.load(MODEL_SAVE+"m.pt")

    seqeval = evaluate.load("../util/seqeval_metric.py")
    print(seqeval)

    # doTrain()

    # saveModel(model)

    # trainer.evaluate(eval_dataset=tokenized_datasets["validation"])
    # trainer.predict(test_dataset=tokenized_datasets["test"])

    ner_pipeline = pipeline("token-classification",model=model,tokenizer=tokenizer,aggregation_strategy="simple")
    res = ner_pipeline("小明坐火车到哈尔宾去看世界之窗")
    print(res)

#
