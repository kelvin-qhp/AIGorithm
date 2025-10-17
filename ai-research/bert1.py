
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,DataCollatorWithPadding
from datasets import load_dataset
import torch
import evaluate


MODEL_NAME ="hfl/rbt3"
MODEL_PATH="./model/bert1/"
MODEL_CHECKPOINT = "./model/checkpoints"
MODEL_SAVE= "./model/save/"
MODEL_SAVE_DICT= "./model/save_dict/"
def loadDataset():
    dataset = load_dataset("csv", data_files="./data/input/ChnSentiCorp_htl_all.csv", split="train")
    dataset = dataset.filter(lambda x: x["review"] is not None)
    print(dataset)
    return dataset

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

def eval_metric(eval_predict):

    predictions, labels = eval_predict
    predictions = predictions.argmax(axis=-1)
    acc = acc_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    acc.update(f1)
    return acc

def getTrainingArguments():
    train_args = TrainingArguments(output_dir=MODEL_CHECKPOINT,      # 输出文件夹
                                   per_device_train_batch_size=64,  # 训练时的batch_size
                                   per_device_eval_batch_size=128,  # 验证时的batch_size
                                   logging_steps=10,                # log 打印的频率
                                   eval_strategy="epoch",     # 评估策略
                                   save_strategy="epoch",           # 保存策略
                                   save_total_limit=3,              # 最大保存数
                                   learning_rate=2e-5,              # 学习率
                                   weight_decay=0.01,               # weight_decay
                                   metric_for_best_model="f1",      # 设定评估指标
                                   load_best_model_at_end=True)     # 训练完成后加载最优模型

    return train_args

def doTrain():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,cache_dir=MODEL_PATH)
    print(model.config)

    # acc_metric = evaluate.load("accuracy")
    # f1_metric = evaluate.load("f1")

    train_args = getTrainingArguments()

    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=tokenized_datasets["train"],
                      eval_dataset=tokenized_datasets["test"],
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=eval_metric)
    trainer.train()

    saveModel(model)

    trainer.evaluate(tokenized_datasets["test"])
    trainer.predict(tokenized_datasets["test"])

def reDoTrain():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,cache_dir=MODEL_PATH)
    train_args = getTrainingArguments()
    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=tokenized_datasets["train"],
                      eval_dataset=tokenized_datasets["test"],
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=eval_metric)

    trainer.train(resume_from_checkpoint=True)

    trainer.evaluate(tokenized_datasets["test"])
    trainer.predict(tokenized_datasets["test"])

    saveModel(model)



def doTest():
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL_CHECKPOINT+"/checkpoint-220")
    # model = torch.load(MODEL_SAVE+"m.pth")

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,cache_dir=MODEL_PATH)
    state_dict = torch.load(MODEL_SAVE+"dict.pth")
    model.load_state_dict(state_dict)
    print(model)
    sen = "这家酒店有蚊子！"
    id2_label = {0: "差评！", 1: "好评！"}
    model.eval()
    with torch.inference_mode():
        inputs = tokenizer(sen, return_tensors="pt")
        logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=-1)
        print(f"输入：{sen}\n模型预测结果:{id2_label.get(pred.item())}")

def saveModel(model):
    torch.save(model.state_dict(),MODEL_SAVE+"dict.pth")
    torch.save(model,MODEL_SAVE+"m.pth")

if __name__ == '__main__':
    myDataset = loadDataset()
    datasets = myDataset.train_test_split(test_size=0.1)
    print(datasets)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
    # print(tokenized_datasets)

    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")


    # doTrain()
    # model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,cache_dir=MODEL_CHECKPOINT+"/checkpoint-220")
    # checkpoint = torch.load(MODEL_CHECKPOINT+"/checkpoint-220/rng_state.pth")

    reDoTrain()

    doTest()