import joblib
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/cleaned_data.csv')
rf_best = joblib.load('best_model.pkl')

importances = rf_best.feature_importances_
feature_names = df.drop(columns=['RICE_YIELD']).columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[indices], color='steelblue')
plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
plt.title('Feature Importance — Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()