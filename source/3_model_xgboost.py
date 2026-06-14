import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

df = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data.csv'))

X = df.drop(columns=['RICE_YIELD']).values
y = df['RICE_YIELD'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, os.path.join(ROOT, 'models/xgboost_model.pkl'))

y_pred_xgb = xgb_model.predict(X_test)

print("=== XGBoost ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred_xgb):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred_xgb):.4f}")

# ========== CROSS-VALIDACIJA (5-fold) ==========
kf = KFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    XGBRegressor(n_estimators=100, random_state=42),
    X, y,
    cv=kf,
    scoring='r2'
)

print("\n=== Cross-validacija (5-fold) ===")
print(f"R² po foldovima: {np.round(cv_scores, 4)}")
print(f"Prosečan R²: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")