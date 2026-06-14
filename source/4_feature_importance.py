import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, '..')

df = pd.read_csv(os.path.join(ROOT, 'data/cleaned_data.csv'))
rf_best = joblib.load(os.path.join(ROOT, 'models/best_model.pkl'))

importances = rf_best.feature_importances_
feature_names = df.drop(columns=['RICE_YIELD']).columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices], color='steelblue')
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(ROOT, 'plots/feature_importance.png'), dpi=150)
plt.show()