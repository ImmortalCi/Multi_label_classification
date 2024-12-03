import pandas as pd

# 读取Excel文件
file_path = "data/qwen-plus-0919/prediction_results.xlsx"
df = pd.read_excel(file_path, engine='openpyxl')
# import pdb; pdb.set_trace()
filtered_df = df[df["TOP1_name"].str.contains('市场监管|消费维权|消费纠纷', na=False)]
# import pdb; pdb.set_trace()

# 将DataFrame保存到Excel文件
output_path = 'data/pipeline1/xiaofeijiufen.xlsx'
filtered_df.to_excel(output_path, index=False)
