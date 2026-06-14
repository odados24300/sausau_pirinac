import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time
import algoritmi.random_forest as rf
import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

df = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data.csv'))

X = df.drop(columns=['RICE_YIELD']).values
y = df['RICE_YIELD'].values

# CEO dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

start = time.time()

forest = rf.RandomForest(n_trees=100, max_depth=20, min_samples_split=5)
forest.fit(X_train, y_train)

joblib.dump(forest, os.path.join(ROOT, f'models/moj_random_forest_{forest.n_trees}_stabala_{forest.max_depth}_dubina.pkl'))

y_pred = forest.predict(X_test)

trajanje = time.time() - start

print("=== Moj Random Forest (ceo dataset, 100 stabala) ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")
print(f"Vreme: {trajanje:.1f} sekundi")