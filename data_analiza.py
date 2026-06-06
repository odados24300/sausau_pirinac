import pandas as pd

X = pd.read_csv('archive/X1.csv')
y = pd.read_csv('archive/y1.csv')

df = pd.concat([X, y], axis=1)

print(df.shape)
print("---")
print(df.describe())

# 1. Proveri koliko ima negativnih vrednosti
print((df['ANNUAL'] < 0).sum())
print((df['avg_rain'] < 0).sum())

# 2. Proveri null vrednosti
print(df.isnull().sum())
