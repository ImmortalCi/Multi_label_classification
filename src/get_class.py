import json

all_class = ["未及时供暖",
"供暖温度不足",
"暖气频繁停暖",
"夜间无法供暖",
"暖气管道损坏",
"暖气设备故障",
"暖气管道被恶意破坏",
"供暖管路爆炸",
"供暖管道施工问题",
"供暖设施未完善",
"物业服务问题",
"热力公司服务问题",
"收费标准问题",
"退费问题",
"供暖合同问题"]

id2label = {str(i):label for i, label in enumerate(all_class)}
label2id = {label:str(i) for i, label in enumerate(all_class)}

res_dict = {}
res_dict['id2label'] = id2label
res_dict['label2id'] = label2id

with open('/home/jzm/gov_KeyExtraction/data/class_file.json', 'w') as f:
    json.dump(res_dict, f, ensure_ascii=False, indent=4)