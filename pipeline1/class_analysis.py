# 分析数据集中，类别的数量分布和占比

import pandas as pd

# 读取Excel文件
file_path = "/home/jzm/gov_classification/data/qwen-plus-0919/prediction_results.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')

class_analysis = {}

for _, row in df.iterrows():
    if row["TOP1_name"] not in class_analysis:
        class_analysis[row["TOP1_name"]] = 1
    else:
        class_analysis[row["TOP1_name"]] += 1

# 按照类别数量排序
class_analysis = dict(sorted(class_analysis.items(), key=lambda x: x[1], reverse=True))
ls = []
for k, v in class_analysis.items():
    ls.append([k, v])

for i in ls[:10]:
    print(i)
    print('\n')
