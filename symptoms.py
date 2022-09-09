import pandas as pd
import numpy as np

org_df = pd.read_csv('raw_highest_symptom.csv', header=0)
df = org_df.copy()
df['highest_symptom_next_day'] = df['highest_symptom'].shift(-1)
df['greater_than_1'] = np.where(df['highest_symptom'] > 1, 1, 0)
df['more_than_1_next_day'] = df.greater_than_1.shift(-1)
df['greater_than_2'] = np.where(df['highest_symptom'] > 2, 1, 0)
df['more_than_2_next_day'] = df.greater_than_2.shift(-1)
df['greater_than_3'] = np.where(df['highest_symptom'] > 3, 1, 0)
df['more_than_3_next_day'] = df.greater_than_3.shift(-1)

df['greater_than_4'] = np.where(df['highest_symptom'] > 4, 1, 0)
df['more_than_4_next_day'] = df.greater_than_4.shift(-1)
print(df.head(10))
df.to_csv('highest_symptom.csv', index=False)
org_df = df[['date', 'highest_symptom']]
org_df.to_csv('raw_highest_symptom.csv', index=False)
