import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

X = pd.read_csv(os.path.join(ROOT, 'data/archive/X1.csv'))
y = pd.read_csv(os.path.join(ROOT, 'data/archive/y1.csv'))

df = pd.concat([X, y], axis=1)

# 1. Brisanje negativnih (dok ANNUAL jos postoji!)
df = df[df['ANNUAL'] >= 0]
df = df[df['avg_rain'] >= 0]

# 2. Ukloni nepotrebne kolone
df = df.drop(columns=['RICE_PRODUCTION', 'ANNUAL'])
nule_kolone = ['DYSTROPEPTS', 'FLUVENTS', 'ORTHENTS', 'UDALFS', 'USTALFS']
df = df.drop(columns=nule_kolone)

# 3. IQR capping
for col in ['Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = df[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

# Sacuvaj kopiju PRE skaliranja za poredjenje
df_pre = df.copy()

# 4. Skaliranje (samo feature kolone, ne target)
feature_kolone = df.drop(columns=['RICE_YIELD']).columns
scaler = StandardScaler()
df[feature_kolone] = scaler.fit_transform(df[feature_kolone])

joblib.dump(scaler, os.path.join(ROOT, 'models/scaler.pkl'))
df.to_csv(os.path.join(ROOT, 'data/cleaned_data_scaled.csv'), index=False)

print(f"Skalirano: {df.shape} -> cleaned_data_scaled.csv")

# ========== GRAFICKO POREDJENJE PRE/POSLE ==========
# Biramo par reprezentativnih kolona
kolone_za_prikaz = ['Nitrogen', 'POTASH', 'PHOSPHATE', 'avg_rain']

fig, axes = plt.subplots(2, len(kolone_za_prikaz), figsize=(16, 8))

for i, col in enumerate(kolone_za_prikaz):
    # Gornji red - PRE skaliranja
    axes[0, i].hist(df_pre[col], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, i].set_title(f'{col}\n(pre skaliranja)')
    axes[0, i].set_ylabel('Frekvencija')
    
    # Donji red - POSLE skaliranja
    axes[1, i].hist(df[col], bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[1, i].set_title(f'{col}\n(posle skaliranja)')
    axes[1, i].set_ylabel('Frekvencija')

plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'plots/poredjenje_skaliranje.png'), dpi=150)
plt.show()