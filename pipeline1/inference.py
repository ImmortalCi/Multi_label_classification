import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import json
from tqdm import tqdm

# 加载预训练的 BERT tokenizer 和模型
model_name_or_path = 'output/cn_v2/checkpoint-21975'  # 替换为你的模型路径
tokenizer = BertTokenizer.from_pretrained(model_name_or_path)
model = BertForSequenceClassification.from_pretrained(model_name_or_path)
model.eval()  # 设置模型为评估模式
model.to("cuda")  # 将模型移动到GPU

# 读取json文件
class_file = "data/filtered_class.json"
with open(class_file, "r", encoding="utf-8") as f:
    class_dict = json.load(f)

id2index = class_dict["id2index"]
index2id = {index: int(i) for i, index in id2index.items()}
id2label = class_dict["id2label"]

data_file = "data/qwen-plus-0919/train.json"
with open(data_file, 'r', encoding="utf-8") as f:
    raw_data = json.load(f)

query = [i["工单标题"] + " " + i["工单内容"] for i in raw_data]
# import pdb; pdb.set_trace()

# 准备输入数据
inputs = tokenizer(query, padding=True, truncation=True, return_tensors="pt")

# 假设 inputs 是一个字典，我们需要将它转换成可以分批处理的 TensorDataset
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# 创建 TensorDataset
dataset = TensorDataset(input_ids, attention_mask)

# 创建 DataLoader，指定 batch_size
batch_size = 2560  # 根据 GPU 内存大小选择适当的批次大小
dataloader = DataLoader(dataset, batch_size=batch_size)

# 分批进行推理
all_outputs = []
with torch.no_grad():
    for batch in tqdm(dataloader):
        # 将 batch 数据转移到 CUDA
        batch = {k: v.to("cuda") for k, v in zip(['input_ids', 'attention_mask'], batch)}
        
        # 进行推理
        outputs = model(**batch)
        
        # 将输出保存下来
        all_outputs.append(outputs.logits)

# 合并所有批次的输出
res = torch.cat(all_outputs, dim=0)

# 获取预测的概率分布
probs = torch.softmax(res, dim=-1)

import pdb;pdb.set_trace()

# 找出每个样本的TOP3预测类别
top_k = 3
top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)


# 创建一个DataFrame来存储预测结果
results = []
for i, (text, top_prob, top_index,) in enumerate(zip(query, top_probs, top_indices, )):
    result = {
        "Text": text,
        "TOP1_id": index2id[top_index[0].item()],
        "TOP1_prob": top_prob[0].item(),
        "TOP1_name" : id2label[str(index2id[top_index[0].item()])],
        "TOP2_id": index2id[top_index[1].item()],
        "TOP2_prob": top_prob[1].item(),
        "TOP2_name" : id2label[str(index2id[top_index[1].item()])],
        "TOP3_id": index2id[top_index[2].item()],
        "TOP3_prob": top_prob[2].item(),
        "TOP3_name" : id2label[str(index2id[top_index[2].item()])],
    }
    results.append(result)

df = pd.DataFrame(results)

# 将DataFrame保存到Excel文件
output_path = 'data/qwen-plus-0919/prediction_results.xlsx'
df.to_excel(output_path, index=False)
print(f"Prediction results saved to {output_path}")