from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, HfArgumentParser, EvalPrediction
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import os
import torch
import numpy as np
import sys
from dataclasses import dataclass, field
from typing import Optional
import datasets
import json


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "The model checkpoint for weights initialization."}
    )

@dataclass
class DataArguments:
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the model during training (a text file)."}
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the model after training (a text file)."}
    )
    all_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input data file for all data."}
    )


def compute_metrics(p: EvalPrediction):
    preds = p.predictions
    y_pred = np.argmax(preds, axis=1)
    labels = p.label_ids
    accuracy = accuracy_score(y_true=labels, y_pred=y_pred)
    f1 = f1_score(y_true=labels, y_pred=y_pred, average='micro')
    return {"accuracy": accuracy, "f1": f1}

def preprocess_data(examples, tokenizer,label2index):
    text = []
    for i in range(len(examples['标题'])):
        if examples['标题'][i] is None:
            examples['标题'][i] = ""
        if examples['内容'][i] is None:
            examples['内容'][i] = ""
        text.append(examples['标题'][i] + ' ' + examples['内容'][i])
    tokenized_text = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    label = examples['emotion']
    label = [label2index[l] for l in label]
    tokenized_text['label'] = label
    return tokenized_text

def main():
    parser = HfArgumentParser((ModelArguments,DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 获取class文件
    known = [
        "不知情", "不公平", "不认可", "有贪腐", "有区别", "有包庇", "有偏袒", 
        "有限制", "有求助", "有困难", "有不满", "有不足", "有要求", "有担忧", 
        "有疏漏", "有抱怨", "有不便", "有不适", "有浪费", "有损害", "有意见", 
        "有批评", "有拖延", "有建议", "有争议", "有质疑"
    ]

    label2index = {}
    for i, k in enumerate(known):
        label2index[k] = i
    num_labels = len(label2index)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                            num_labels=num_labels,
                                                            )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if data_args.all_file is None:
        data_files = {'train': data_args.train_file, 'validation': data_args.validation_file, 'test': data_args.test_file}
        raw_datasets = datasets.load_dataset('json', data_files=data_files)
    else:
        raw_datasets = datasets.load_dataset('json', data_files=data_args.all_file)
        raw_datasets = raw_datasets.shuffle(seed=42)
        raw_datasets = raw_datasets['train'].train_test_split(test_size=0.1)
        test_valid_split = raw_datasets['test'].train_test_split(test_size=0.5)
        raw_datasets['validation'] = test_valid_split['test']
        raw_datasets['test'] = test_valid_split['train']


    encoded_datasets = raw_datasets.map(lambda examples: preprocess_data(examples, tokenizer, label2index), batched=True)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=encoded_datasets['train'],
        eval_dataset=encoded_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()
    
if __name__ == '__main__':
    main()