import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/imdb.csv')
print(df.head())

print(df.describe())

print(df.isna().sum())
print(df.isnull().sum())

print(df.info())
print(df)

df["label"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)
df.drop("sentiment", axis=1, inplace=True)

train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv('data/train.csv')
test.to_csv('data/test.csv')