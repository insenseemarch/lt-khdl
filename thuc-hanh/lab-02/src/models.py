# src/models.py
# Thuật toán Machine Learning implement bằng NumPy
# Gồm: sigmoid, loss, logistic regression (GD), predict, confusion matrix
# Viết vectorized – ổn định số học – sạch sẽ cho notebook 03

import numpy as np

# ============================================================
# 1) Hàm sigmoid ổn định số học
# ============================================================
def sigmoid(z):
    # clip để tránh overflow
    z = np.clip(z, -20, 20)
    return 1 / (1 + np.exp(-z))

# ============================================================
# 2) Loss function (binary cross-entropy)
# ============================================================
def loss_fn(X, y, w):
    z = X @ w
    p = sigmoid(z)
    eps = 1e-9  # tránh log(0)
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

# ============================================================
# 3) Gradient Descent Logistic Regression
# ============================================================
def logistic_regression(X, y, lr=0.01, epochs=300):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)

    for _ in range(epochs):
        z = X @ w
        p = sigmoid(z)
        grad = (X.T @ (p - y)) / n_samples
        w -= lr * grad
    return w

# ============================================================
# 4) Prediction
# ============================================================
def predict(X, w, threshold=0.5):
    return (sigmoid(X @ w) >= threshold).astype(int)

# ============================================================
# 5) Confusion Matrix
# ============================================================
def confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TP, FP], [FN, TN]])

# ============================================================
# File kết thúc
# ============================================================
