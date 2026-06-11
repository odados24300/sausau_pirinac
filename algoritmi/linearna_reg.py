import numpy as np

class MyLinearRegression:
    def fit(self, X, y):
        # Dodaj kolonu jedinica za bias
        X_b = np.c_[np.ones(X.shape[0]), X]
        # Formula najmanjih kvadrata
        self.weights = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    
    def predict(self, X):
        X_b = np.c_[np.ones(X.shape[0]), X]
        return X_b @ self.weights