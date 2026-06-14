import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import algoritmi.linearna_reg as lr
import matplotlib.pyplot as plt
import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

# ========== GLAVNI MODEL (skalirani podaci) ==========
df = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data_scaled.csv'))

X = df.drop(columns=['RICE_YIELD']).values
y = df['RICE_YIELD'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lr.MyLinearRegression()
model.fit(X_train, y_train)
joblib.dump(model, os.path.join(ROOT, 'models/linear_model.pkl'))

y_pred = model.predict(X_test)

print("=== My Linear Regression ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")

# ========== CROSS-VALIDACIJA (5-fold, ručna) ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_r2 = []

for train_idx, test_idx in kf.split(X):
    m = lr.MyLinearRegression()
    m.fit(X[train_idx], y[train_idx])
    pred = m.predict(X[test_idx])
    cv_r2.append(r2_score(y[test_idx], pred))

cv_r2 = np.array(cv_r2)
print("\n=== Cross-validacija (5-fold) ===")
print(f"R² po foldovima: {np.round(cv_r2, 4)}")
print(f"Prosečan R²: {cv_r2.mean():.4f} (+/- {cv_r2.std():.4f})")

# ========== POREĐENJE TEŽINA PRE/POSLE SKALIRANJA ==========
df_unscaled = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data.csv'))
feature_names = df.drop(columns=['RICE_YIELD']).columns.tolist()

def istreniraj_tezine(dframe):
    Xf = dframe.drop(columns=['RICE_YIELD']).values
    yf = dframe['RICE_YIELD'].values
    Xf_train, _, yf_train, _ = train_test_split(Xf, yf, test_size=0.2, random_state=42)
    m = lr.MyLinearRegression()
    m.fit(Xf_train, yf_train)
    return m.weights

w_unscaled = istreniraj_tezine(df_unscaled)
w_scaled = model.weights

print("\n=== Poređenje težina ===")
print(f"{'Atribut':<18} {'Pre skaliranja':>18} {'Posle skaliranja':>18}")
print("-" * 56)
print(f"{'(bias)':<18} {w_unscaled[0]:>18.4f} {w_scaled[0]:>18.4f}")
for i, name in enumerate(feature_names):
    print(f"{name:<18} {w_unscaled[i+1]:>18.4f} {w_scaled[i+1]:>18.4f}")

# ========== VIZUALIZACIJA ==========
greske = np.abs(y_test - y_pred)
prag = 800
boje = np.where(greske > prag, 'crimson', 'steelblue')

plt.figure(figsize=(8, 7))
plt.scatter(y_test, y_pred, c=boje, alpha=0.6, edgecolors='black', linewidths=0.3)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'k--', linewidth=2, label='Idealna predikcija')
plt.xlabel('Stvarne vrednosti')
plt.ylabel('Predviđene vrednosti')
plt.title(f'Stvarne vs Predviđene (crveno: greška > {prag} kg/ha)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'plots/scatter_greske.png'), dpi=150)
plt.show()