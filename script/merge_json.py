import os
import pandas as pd

# 指定文件夹路径
folder_path = 'C:/Users/18054/Desktop/新建文件夹'

# 创建一个空的 DataFrame 列表
dataframes = []

CountExist=0
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.json'):
        file_path = os.path.join(folder_path, filename)
        # 读取 JSON 文件，指定 encoding 为 'utf-8'
        df = pd.read_json(file_path, encoding='utf-8')
        dataframes.append(df)
        if df.shape[0]>=10:
            CountExist+=1
            print(filename,df.shape[0])

# 合并所有 DataFrame
combined_df = pd.concat(dataframes, ignore_index=True)

# 保存为 JSON 文件
combined_df.to_json('merged_file.json', orient='records', force_ascii=False)
print(CountExist)