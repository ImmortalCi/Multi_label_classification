import os
import time
import pandas as pd
import json
from openai import OpenAI
import multiprocessing
from tqdm import tqdm
api_key = "sk-be07869188724fbe987b247ccbbcd79c"
import re
client = OpenAI(
    api_key="sk-be07869188724fbe987b247ccbbcd79c", 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)



def api_request(raw_data, prompt):
    content = raw_data["标题"] + " " + raw_data["内容"]

    completion = client.chat.completions.create(
        model="qwen-plus-0919", # 更多模型请参见模型列表文档。
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"工单标题和内容为{content}"},
            ]
)
    resp = completion.choices[0].message.content
    return resp
prompt = """
您将接收一个工单内容，请完成以下任务：

1.	**判断是否属于供暖相关问题**

如果工单内容不属于供暖问题，请直接返回以下格式：

{

"status": "非供暖问题"

}

如果工单内容属于供暖问题，请继续完成以下任务。

2.	**从可选问题标签列表中选择一个或者多个相关标签**

[未及时供暖,
供暖温度不足,
暖气频繁停暖,
夜间无法供暖,
暖气管道损坏,
暖气设备故障,
暖气管道被恶意破坏,
供暖管路爆炸,
供暖管道施工问题,
供暖设施未完善,
物业服务问题,
热力公司服务问题,
收费标准问题,
退费问题,
供暖合同问题]

3.	**从可选情绪标签选择一个最贴切的情绪标签**

[
    "不知情", "不公平", "不认可", "有贪腐", "有区别", "有包庇", "有偏袒", 
    "有限制", "有求助", "有困难", "有不满", "有不足", "有要求", "有担忧", 
    "有疏漏", "有抱怨", "有不便", "有不适", "有浪费", "有损害", "有意见", 
    "有批评", "有拖延", "有建议", "有争议", "有质疑"
]

**输出格式：**

返回结果严格按照以下 JSON 格式返回：

**情况1：不属于供暖问题**

{

"status": "非供暖问题"

}

**情况2：属于供暖问题**

{

"status": "供暖问题",

"tags": ["供暖温度不足", "供暖系统故障"],

"emotion": "有意见",

}

**注意事项：**

返回的问题标签tags仅限于可选标签列表中的选项,如果工单涉及到多个问题标签，请返回多个问题标签。

情绪标签emotion只返回一个最贴切的

JSON 字符串必须符合标准，确保可以直接解析。
"""

res = []

# 读取Excel文件
file_path = "/home/jzm/gov_KeyExtraction/data/20241101-呼市.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
# 读取前500行
# df = df.head(100)

output_path = f"/home/jzm/gov_KeyExtraction/data/iter_all.json"

for index, row in tqdm(df.iterrows(), total=max(500, df.shape[0])):
    now_dict = {"标题": row["标题"], "内容": row["内容"]}
    try:
        result = api_request(row, prompt)
    except Exception as e:
        print(e)
        result = {"status": "LLM生成结果失败，error"}
    # 使用正则表达式匹配第一个{开始，第一个}结束，确保是字典形式
    match = re.search(r'\{.*?\}', result, re.DOTALL)
    if match:
        result = match.group(0)
    else:
        result = {"status": "未生成json格式"}
    # 使用json.loads将字符串转换为字典
    try:
        result_dict = json.loads(result)
        now_dict.update(result_dict)
        res.append(now_dict)
        with open(output_path, mode='w') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(e)
        print(result)
        now_dict.update({"status": "error"})
        res.append(now_dict)
        with open(output_path, mode='w+') as f:
            json.dump(res, f, ensure_ascii=False, indent=4)
        continue



# 将结果保存到json文件

with open("/home/jzm/gov_KeyExtraction/data/all.json", mode='w') as f:
    json.dump(res, f, ensure_ascii=False, indent=4)


