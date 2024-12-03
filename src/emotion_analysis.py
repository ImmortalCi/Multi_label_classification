import json

# 读取数据集
with open('/home/jzm/gov_KeyExtraction/data/all.json', "r") as f:
    dataset = json.load(f)

# 预定义的情绪
known = [
    "不知情", "不公平", "不认可", "有贪腐", "有区别", "有包庇", "有偏袒", 
    "有限制", "有求助", "有困难", "有不满", "有不足", "有要求", "有担忧", 
    "有疏漏", "有抱怨", "有不便", "有不适", "有浪费", "有损害", "有意见", 
    "有批评", "有拖延", "有建议", "有争议", "有质疑"
]
res = {}
res_filtered = []
cnt = 0
for data in dataset:
    # import pdb;pdb.set_trace()
    if "emotion" in data.keys() and data['emotion'] in known:
        res_filtered.append(data)
    else:
        if "emotion" not in data.keys():
            print("数据没有情感")
            print(data["标题"])
        cnt += 1
print(f"共有{cnt}个样本没有")

with open("/home/jzm/gov_KeyExtraction/data/all_filtered.json", 'w') as f:
    json.dump(res_filtered, f, ensure_ascii=False, indent=4)
