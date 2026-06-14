import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

df = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data.csv'))

X = df.drop(columns=['RICE_YIELD']).values
y = df['RICE_YIELD'].values


top_features = ['Nitrogen', 'SANDY_ALFISOL', 'POTASH', 'PHOSPHATE', 'avg_rain']

X_top = df[top_features].values
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(
    X_top, y, test_size=0.2, random_state=42)

rf_top = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
rf_top.fit(X_train_top, y_train_top)

y_pred_top = rf_top.predict(X_test_top)

print("=== Random Forest (top 5 atributa) ===")
print(f"MAE:  {mean_absolute_error(y_test_top, y_pred_top):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_top, y_pred_top)):.2f}")
print(f"R²:   {r2_score(y_test_top, y_pred_top):.4f}")