import pandas as pd
import json

# 读取xlsx文件
df = pd.read_excel("/home/jzm/gov_KeyExtraction/data/呼市供暖验证数据.xlsx")
datas = []

# 遍历数据
for index, row in df.iterrows():
    data = {
        "标题": row["标题"],
        "内容": row["内容"]
    }
    datas.append(data)

# 保存为json文件
with open("/home/jzm/gov_KeyExtraction/data/呼市供暖验证数据.json", "w") as f:
    json.dump(datas, f, ensure_ascii=False, indent=4)