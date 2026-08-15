'''Create a DataFrame with missing values and fill them with column mean.'''
import pandas as pd
import numpy as np

data2 = {
    'Name': ['Amit', 'Sagar', 'Pooja'],
    'Math': [np.nan, 76, 88],
    'Science': [91, np.nan, 85]
}

df = pd.DataFrame(data2)

print("DataFrame with Missing Values:")
print(df)

df['Math'] = df['Math'].fillna(df['Math'].mean())
df['Science'] = df['Science'].fillna(df['Science'].mean())

print("\nDataFrame after filling missing values:")
print(df)