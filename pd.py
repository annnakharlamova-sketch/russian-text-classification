import pandas as pd
df = pd.read_csv('data/rusentiment/train.csv')
print(f'✅ Train data: {len(df)} строк, {df.columns.tolist()} колонки')
print('Первые 3 строки:')
print(df.head(3))