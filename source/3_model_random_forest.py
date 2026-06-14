import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

df = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data.csv'))

X = df.drop(columns=['RICE_YIELD']).values
y = df['RICE_YIELD'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("=== Random Forest ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_rf):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred_rf):.4f}")

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    n_iter=20,
    cv=5,
    scoring='r2',
    random_state=42
)

search.fit(X_train, y_train)

joblib.dump(search.best_estimator_, os.path.join(ROOT, 'models/best_model.pkl')) #OVO CUVA MODEL

print(f"Najbolji parametri: {search.best_params_}")

y_pred_best = search.best_estimator_.predict(X_test)
print(f"MAE:  {mean_absolute_error(y_test, y_pred_best):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_best)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred_best):.4f}")