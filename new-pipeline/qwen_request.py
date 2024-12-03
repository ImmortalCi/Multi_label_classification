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



sta = "你是一个出色的政府工作人员，接下来我将给你一个工单，请你提取和概括出其中的关键词"

prompt = """
请从工单内容中提取能够准确概括事件核心的词或短语，要求如下：

1. 完整反映原文中的情境和行为意图，突出工单的主要诉求或问题，特别是行为、状态或事件的具体性质。

2. 不包含地址、时间、涉事主体、姓名、手机号或任何个人敏感信息。

3. 优先提取高频出现的词汇：
   - 如果某个词或短语在不同句子中多次出现，应优先提取，确保其能够完整呈现原文的核心诉求。

4. 输出格式：
   - 每行只输出一个词或短语，不要包含标点符号或任何额外信息。

示例：

- 正面示例：
  - 工单内容：“暖气温度只有17度，家里很冷。” 提取结果：
    暖气温度低

  - 工单内容：“暖气的价格涨了很多，让人难以接受。” 提取结果：
    暖气涨价


- 反面示例：
  - 不要提取：“家里”、“难以接受”、“17度”等非核心或包含个人信息的词汇。
"""

def api_request(raw_data):
    content = raw_data["title"] + " " + raw_data["content"]
    completion = client.chat.completions.create(
        model="qwen-plus", # 更多模型请参见模型列表文档。
        messages=[
            {"role": "system", "content": sta},
            {"role": "user", "content": f"{content}"},
            {"role": "system", "content": prompt}
            ]
)
    resp = completion.choices[0].message.content
    return resp


# 读取json文件
with open("/home/jzm/gov_KeyExtraction/data/pipeline1/gongnuan_result_step1.json", mode='r') as  f:
    all_data = json.load(f)

for data in tqdm(all_data):
    res = api_request(data)
    # 按照换行符分割
    res = res.split("\n")
    data["keywords"] = res

# 保存结果
with open("/home/jzm/gov_KeyExtraction/data/pipeline1/gongnuan_result_keyword.json", mode='w') as f:
    json.dump(all_data, f, ensure_ascii=False, indent=4)



           
