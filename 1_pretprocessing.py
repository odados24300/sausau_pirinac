import pandas as pd

X = pd.read_csv('archive/X1.csv')
y = pd.read_csv('archive/y1.csv')

df = pd.concat([X, y], axis=1)

# 1. Ukloni negativne vrednosti
df = df[df['ANNUAL'] >= 0]
df = df[df['avg_rain'] >= 0]

print(f"Redova nakon čišćenja: {len(df)}")

# 2. Odluči target i ukloni drugu kolonu
# Koristimo RICE_YIELD, uklanjamo RICE_PRODUCTION
df = df.drop(columns=['RICE_PRODUCTION'])
df = df.drop(columns=['ANNUAL'])

# 3. Proveri ko su X i y
X = df.drop(columns=['RICE_YIELD'])
y = df['RICE_YIELD']

# Ukloni kolone koje su skoro sve nule
# (DYSTROPEPTS, FLUVENTS, ORTHENTS, UDALFS, USTALFS)
nule_kolone = ['DYSTROPEPTS', 'FLUVENTS', 'ORTHENTS', 'UDALFS', 'USTALFS']
df = df.drop(columns=nule_kolone)

for col in ['Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5*IQR, 
                           upper=Q3 + 1.5*IQR)

print(X.shape)
print(y.describe())

df.to_csv('cleaned_data.csv', index=False)
# print(df.columns.tolist())
