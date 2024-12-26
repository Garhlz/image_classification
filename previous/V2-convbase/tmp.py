import pandas as pd

# 读取 CSV 文件
df = pd.read_csv('./submission.csv')

# 去掉 'tensor()' 部分
df['id'] = df['id'].apply(lambda x: str(x).replace('tensor(', '').replace(')', ''))

# 保存为新的 CSV 文件
df.to_csv('sub2-convbase.csv', index=False)

print(df.head())  # 打印前几行，查看处理效果
