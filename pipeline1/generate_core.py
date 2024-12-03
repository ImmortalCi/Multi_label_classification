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
    content = raw_data
    completion = client.chat.completions.create(
        model="qwen-plus", # 更多模型请参见模型列表文档。
        messages=[
            {"role": "system", "content": "你将收到一段关于市民反馈的工单内容。请仔细阅读这段文字，并从中提取所有关键事件属性信息。这些信息应该能够全面描述工单中提到的问题或请求。请按照以下格式组织你的回答："},
            {"role": "system", "content": "每个关键信息点用一行表示。样例为“属性名：属性信息”\n如果有多个相似的信息点，请合并成一条简洁的说明。\n现在，请根据上述指示分析以下工单内容，并提取其中的关键事件属性信息："},
            {"role": "user", "content": content}
            ]
)
    resp = completion.choices[0].message.content
    return resp

# # 读取Excel文件
# file_path = "data/pipeline1/xiaofeijiufen.xlsx"
# df = pd.read_excel(file_path, engine='openpyxl')

# # 取df的前10行数据
# df = df.head(1000)

# result = {}
# for _, row in tqdm(df.iterrows()):
#     text = row["Text"]
#     try:
#         res = api_request(row)
#         # 将字符串使用换行符分割
#         res_list = res.split("\n")
#         # 将字符串转换为字典
#         res_dict = {}
#         for item in res_list:
#             key, value = item.split("：")
#             res_dict[key] = value
#         result[text] = res_dict
#         # import pdb; pdb.set_trace()
#     except:
#         print(f"处理{key}时出现错误")
#         # import pdb; pdb.set_trace()
#         continue

# 读取json文件
with open('/home/jzm/gov_llm/vllm/data/gongnuan_id_llm.json', mode='r') as f:
    data = json.load(f)


result = []

for i, item in tqdm(enumerate(data)):
    tmp = item
    if item['llm_classify'] == "是":
        # import pdb; pdb.set_trace()
        text = item["title"] + " " + item["content"]
        try:
            res = api_request(text)
            # 将字符串使用换行符分割
            res_list = res.split("\n")
            # 将字符串转换为字典
            property_dict = {}
            for item in res_list:
                key, value = item.split("：")
                property_dict[key] = value
            tmp['property_dict'] = property_dict
            result.append(tmp)
            # import pdb; pdb.set_trace()
        except Exception as e:
            # import pdb; pdb.set_trace()
            print(e)
            continue
# 将结果保存到json文件
output_path = 'data/pipeline1/gongnuan_result_step1.json'
with open(output_path, mode='w') as f:
    json.dump(result, f, ensure_ascii=False, indent=4)