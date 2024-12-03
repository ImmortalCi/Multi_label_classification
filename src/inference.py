import torch
from transformers import BertTokenizer, BertForSequenceClassification
# from preprocess_test import preprocess_data
import pandas as pd

# 加载预训练的 BERT tokenizer 和模型
model_name_or_path = '/home/jzm/gov_KeyExtraction/output/multi_label_v1/checkpoint-1080'  # 替换为你的模型路径
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)
model.eval()  # 设置模型为评估模式
model.to("cuda")  # 将模型移动到GPU

# 读取测试的json文件
import json
with open("/home/jzm/gov_KeyExtraction/data/呼市供暖验证数据.json", "r") as f:
    dataset = json.load(f)

texts = []
for data in dataset:
    texts.append(data['标题'] + data['内容'])

# 读取分类标签
with open("/home/jzm/gov_KeyExtraction/data/class_file.json", "r") as f:
    class_file = json.load(f)
id2label = class_file['id2label']


# 准备输入数据
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
inputs.to("cuda")

# 进行推理
with torch.no_grad():
    outputs = model(**inputs)

# 获取 logits
logits = outputs.logits
# 使用 Sigmoid 函数将 logits 转换为概率
probabilities = torch.sigmoid(logits).cpu().numpy()

# 设定阈值（如0.5），将概率转换为多标签预测
threshold = 0.5
predictions = (probabilities >= threshold).astype(int)

# 将预测的标签索引转换为标签名称
predicted_labels = []
for pred in predictions:
    predicted_labels.append([id2label[str(i)] for i in range(len(pred)) if pred[i] == 1])

# 将预测结果保存为json文件
output = []
for i, data in enumerate(dataset):
    output.append({"标题": data['标题'], "内容": data['内容'], "预测标签": predicted_labels[i]})
with open("/home/jzm/gov_KeyExtraction/data/呼市供暖验证数据_predict.json", "w") as f:
    json.dump(output, f, indent=4, ensure_ascii=False)
# import pdb; pdb.set_trace()
