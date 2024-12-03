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
    class_file: str = field(
        metadata={"help": "The class file for the dataset."}
    )
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
    preds = torch.sigmoid(torch.tensor(preds)).numpy()  # 将 logits 转换为概率
    y_true = p.label_ids

    # 使用阈值0.5将概率转换为二进制预测
    y_pred = (preds >= 0.5).astype(int)

    # 计算多个指标
    roc_auc = roc_auc_score(y_true, preds, average="micro")  # AUC for probabilities
    f1 = f1_score(y_true, y_pred, average="micro")  # F1 for binary predictions
    return {"roc_auc": roc_auc, "f1": f1}

def preprocess_data(examples, tokenizer, id2label):
    if examples['title'] is None:
        examples['title'] = ""
    if examples['abstract'] is None:
        examples['abstract'] = ""
    text = []
    for i in range(len(examples['title'])):
        if examples['title'][i] is None:
            examples['title'][i] = ""
        if examples['abstract'][i] is None:
            examples['abstract'][i] = ""
        text.append(examples['title'][i] + ' ' + examples['abstract'][i])
    tokenized_text = tokenizer(text, padding="max_length", truncation=True, max_length=512)
    # 假设 examples['label'] 是类别索引列表
    num_labels = len(id2label)  # 获取总类别数
    labels = np.zeros((len(examples['label']), num_labels))
    for i, sample_labels in enumerate(examples['label']):
        for label in sample_labels:
            labels[i, label] = 1  # 设置为多热编码

    tokenized_text['label'] = labels.tolist()
    # import pdb; pdb.set_trace()
    return tokenized_text

def main():
    parser = HfArgumentParser((ModelArguments,DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 读取class文件
    with open(data_args.class_file, 'r', encoding='utf-8') as f:
        class_data = json.load(f)

    id2label = class_data['id2label']
    num_labels = len(id2label)
    model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path,
                                                            num_labels=num_labels,
                                                            problem_type="multi_label_classification"
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

    # import pdb; pdb.set_trace()

    encoded_datasets = raw_datasets.map(
        lambda examples: preprocess_data(examples, tokenizer, id2label), 
        batched=True,
        remove_columns=raw_datasets["train"].column_names  # 删除原始列，只保留处理后的特征
    )
    # import pdb; pdb.set_trace()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_datasets['train'],
        eval_dataset=encoded_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    trainer.train()

    trainer.evaluate()
    
if __name__ == '__main__':
    main()