import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib
import algoritmi.linearna_reg as lr      # treba za unpickle custom klasa
import algoritmi.random_forest as rf
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

df = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data.csv'))
df_scaled = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data_scaled.csv'))

def get_test(dframe):
    X = dframe.drop(columns=['RICE_YIELD']).values
    y = dframe['RICE_YIELD'].values
    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    return Xte, yte

Xte, yte = get_test(df)              # neskalirano (stabla)
Xte_s, yte_s = get_test(df_scaled)   # skalirano (linear)

# Učitaj sačuvane modele
modeli = {
    'Linear Regression':       (joblib.load(os.path.join(ROOT, 'models/linear_model.pkl')),   Xte_s, yte_s),
    'Random Forest (sklearn)': (joblib.load(os.path.join(ROOT, 'models/best_model.pkl')),     Xte,   yte),
    'XGBoost':                 (joblib.load(os.path.join(ROOT, 'models/xgboost_model.pkl')),  Xte,   yte),
    'Moj RF':        (joblib.load(os.path.join(ROOT, 'models/moj_random_forest_100_stabala_20_dubina.pkl')), Xte, yte),
}

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()
PRAG = 500  # prag iznad kog je greška "velika" - npr. 2x MAE

for i, (naziv, (model, X_test, y_true)) in enumerate(modeli.items()):
    y_pred = model.predict(X_test)
    
    greske = np.abs(y_true - y_pred)
    
    boje = np.where(greske > PRAG, 'crimson', 'steelblue')
    n_crvenih = (greske > PRAG).sum()
    
    ax = axes[i]
    ax.scatter(y_true, y_pred, c=boje, alpha=0.6, edgecolors='black', linewidths=0.3)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', linewidth=2)
    ax.set_title(f'{naziv}  (R² = {r2_score(y_true, y_pred):.3f}, greške > {PRAG}: {n_crvenih})')
    ax.set_xlabel('Stvarne vrednosti')
    ax.set_ylabel('Predviđene vrednosti')

plt.suptitle(f'Poređenje modela — crveno = greška veća od {PRAG} kg/ha', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'plots/poredjenje_predikcija.png'), dpi=150)
plt.show()