import json
import pandas as pd
# 读取json文件
with open('data/pipeline1/gongnuan_result_keyword.json', 'r') as f:
    datas = json.load(f)

keywords = {}
for data in datas:
    for keyword in data['keywords']:
        if keyword in keywords:
            keywords[keyword] += 1
        else:
            keywords[keyword] = 1

# 按照出现次数排序
keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)

# 保存为Excel文件
df = pd.DataFrame(keywords, columns=['关键词', '出现次数'])
df.to_excel('data/pipeline1/gongnuan_keywords.xlsx', index=False)

