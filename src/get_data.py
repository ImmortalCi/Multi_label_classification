import json

with open('/home/jzm/gov_KeyExtraction/data/all.json', 'r', encoding='utf-8') as f:
    all_data = json.load(f)

with open('/home/jzm/gov_KeyExtraction/data/class_file.json', 'r', encoding='utf-8') as f:
    class_data = json.load(f)   
    id2label = class_data['id2label']
    label2id = class_data['label2id']

dataset = []
for data in all_data:
    now_dict = {}
    if data["status"] == "非供暖问题" or data["status"] == "error":
        continue
    else:
        now_dict['title'] = data['标题']
        now_dict['abstract'] = data['内容']
        now_dict['label'] = []
        try:
            for label in data['tags']:
                if label in label2id:
                    now_dict['label'].append(int(label2id[label]))
        except KeyError:
            import pdb; pdb.set_trace()
    dataset.append(now_dict)

with open("/home/jzm/gov_KeyExtraction/data/dataset.json", "w", encoding='utf-8') as f:
    json.dump(dataset, f, ensure_ascii=False, indent=4)
