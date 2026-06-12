import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import algoritmi.linearna_reg as lr

import matplotlib.pyplot as plt

df = pd.read_csv('data/cleaned_data.csv')

X = df.drop(columns=['RICE_YIELD']).values
y = df['RICE_YIELD'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = lr.MyLinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("=== My Linear Regression ===")
print(f"MAE:  {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²:   {r2_score(y_test, y_pred):.4f}")


# A: Stvarne vs predviđene vrednosti
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5, color='steelblue')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='Idealna predikcija')
plt.xlabel('Stvarne vrednosti')
plt.ylabel('Predviđene vrednosti')
plt.title('Stvarne vs Predviđene vrednosti')
plt.legend()

# B: Residual plot
residuals = y_test - y_pred

plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuals, alpha=0.5, color='coral')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predviđene vrednosti')
plt.ylabel('Greška (residual)')
plt.title('Residual plot')

plt.tight_layout()
plt.savefig('linearna_regresija_vizualizacija.png')
plt.show()