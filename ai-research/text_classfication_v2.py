from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments,DataCollatorWithPadding
from datasets import load_dataset
import torch
import evaluate


def load_mydataset():
    my_dataset = load_dataset("csv",data_files="./data/input/text_classfication/ChnSentiCorp_htl_all.csv",split="train")
    my_dataset = my_dataset.filter(lambda x: x["review"] is not None)
    return my_dataset

def process_function(examples):
    tokenized_examples = tokenizer(examples["review"], max_length=128, truncation=True)
    tokenized_examples["labels"] = examples["label"]
    return tokenized_examples

def eval_metric(eval_predict):
    predictions,labels = eval_predict
    predictions = predictions.argmax(axis = -1)
    acc = acc_meric.compute(predictions=predictions,references=labels)
    f1 = f1_metric.compute(predictions=predictions,references=labels)

    acc.update(f1)
    return acc


if __name__ == '__main__':
    my_dataset = load_mydataset()
    print(my_dataset)

    tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
    model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3")
    datasets = my_dataset.train_test_split(test_size=0.1)
    tokenized_datasets = datasets.map(process_function, batched=True, remove_columns=datasets["train"].column_names)
    tokenized_datasets

    acc_meric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    train_args = TrainingArguments(output_dir="./model/checkpoints2",      # 输出文件夹
                                   per_device_train_batch_size=64,    # 训练时的batch_size
                                   per_device_eval_batch_size=128,    # 验证时的batch_size
                                   logging_steps=10,           # log 打印的频率
                                   eval_strategy="epoch",     # 评估策略
                                   save_strategy="epoch",        # 保存策略
                                   save_total_limit=3,          # 最大保存数
                                   learning_rate=2e-5,          # 学习率
                                   weight_decay=0.01,           # weight_decay
                                   metric_for_best_model="f1",      # 设定评估指标
                                   load_best_model_at_end=True)


    trainer = Trainer(model=model,
                      args=train_args,
                      train_dataset=tokenized_datasets["train"],
                      eval_dataset=tokenized_datasets["test"],
                      data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
                      compute_metrics=eval_metric)

    trainer.train()
    trainer.evaluate(tokenized_datasets["test"])
    trainer.predict(tokenized_datasets["test"])