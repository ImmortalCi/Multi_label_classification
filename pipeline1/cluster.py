import os
import time
import json
import pandas as pd
import pandas as pd
from openai import OpenAI
import multiprocessing
from tqdm import tqdm
api_key = "sk-be07869188724fbe987b247ccbbcd79c"
import re
client = OpenAI(
    api_key="sk-be07869188724fbe987b247ccbbcd79c", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def api_request(raw_data):
    completion = client.chat.completions.create(
        model="qwen-plus-0919", # 更多模型请参见模型列表文档。
        messages=[
            {"role": "system", "content": "你将收到一个包含多个核心属性的列表。请根据以下规则对这个列表进行处理，并返回一个新的核心属性列表："},
            {"role": "system", "content": "- 如果两个或多个属性的意思非常接近或者完全相同，请将它们合并为一个属性。\n- 如果一个属性是另一个属性的子集（即被另一个属性所包含），请将这两个属性合并为一个更广泛的属性。\n- 如果两个属性之间存在部分重叠但又不完全相同，请考虑是否可以合理地将它们合并成一个更概括性的属性。\n- 合并后的属性应该能够准确反映原始属性的主要含义。"},
            {"role": "system", "content": "现在，请根据上述指示处理以下核心属性列表："},
            {"role": "user", "content": raw_data}
            ]
)
    resp = completion.choices[0].message.content
    return resp


# 读取Excel文件
file_path = "data/pipeline1/xiaofeijiufen_result_step1.json"
with open(file_path, mode='r') as f:
    data = json.load(f)

core_lst = []
for key, value in tqdm(data.items()):
    for key, value in value.items():
        if key not in core_lst:
            core_lst.append(key.strip())

# list转换为字符串
core_lst = "\n".join(core_lst)
# print(len(core_lst))
# print(core_lst)

res = api_request(core_lst)
import pdb; pdb.set_trace()
print(res)