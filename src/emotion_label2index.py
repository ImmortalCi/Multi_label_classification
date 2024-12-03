import json

known = [
    "不知情", "不公平", "不认可", "有贪腐", "有区别", "有包庇", "有偏袒", 
    "有限制", "有求助", "有困难", "有不满", "有不足", "有要求", "有担忧", 
    "有疏漏", "有抱怨", "有不便", "有不适", "有浪费", "有损害", "有意见", 
    "有批评", "有拖延", "有建议", "有争议", "有质疑"
]

res = {}
for i, k in enumerate(known):
    res[k] = i

