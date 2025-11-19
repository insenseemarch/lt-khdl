import numpy as np

# -----------------------------
# Sigmoid
# -----------------------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# -----------------------------
# Logistic loss + gradient
# -----------------------------
def logistic_loss_and_grad(X, y, w, b, l2=0.0):
    m = X.shape[0]
    logits = X.dot(w) + b
    probs = sigmoid(logits)
    eps = 1e-12
    loss = - (y * np.log(probs + eps) + (1-y) * np.log(1 - probs + eps)).mean()
    loss += 0.5 * l2 * np.sum(w*w)

    diff = (probs - y).reshape(-1,1)
    grad_w = (X * diff).mean(axis=0) + l2 * w
    grad_b = diff.mean()
    return loss, grad_w, grad_b

# -----------------------------
# Logistic Regression (SGD)
# -----------------------------
def logistic_regression_sgd(X, y, lr=0.1, epochs=100, batch_size=64, l2=0.0, verbose=False):
    n, d = X.shape
    w = np.zeros(d, float)
    b = 0.0
    loss_hist = []

    for ep in range(epochs):
        perm = np.random.permutation(n)
        Xp = X[perm]
        yp = y[perm]

        for i in range(0, n, batch_size):
            xb = Xp[i:i+batch_size]
            yb = yp[i:i+batch_size]
            loss, gw, gb = logistic_loss_and_grad(xb, yb, w, b, l2=l2)
            w -= lr * gw
            b -= lr * gb

        full_loss, _, _ = logistic_loss_and_grad(X, y, w, b, l2=l2)
        loss_hist.append(full_loss)
        
        if verbose and ep % max(1, epochs//10) == 0:
            print(f"Epoch {ep} | Loss {full_loss:.4f}")

    return w, b, np.array(loss_hist)

# -----------------------------
# Predict functions
# -----------------------------
def predict_proba(X, w, b):
    return sigmoid(X.dot(w) + b)

def predict_label(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)

# -----------------------------
# Metrics
# -----------------------------
def accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()


def confusion_matrix_np(y_true, y_pred):
    tn = np.sum((y_true==0) & (y_pred==0))
    fp = np.sum((y_true==0) & (y_pred==1))
    fn = np.sum((y_true==1) & (y_pred==0))
    tp = np.sum((y_true==1) & (y_pred==1))
    return np.array([[tn, fp],[fn, tp]])


def precision_recall_f1(y_true, y_pred):
    cm = confusion_matrix_np(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]

    prec = tp / (tp + fp + 1e-12)
    rec  = tp / (tp + fn + 1e-12)
    f1   = 2 * prec * rec / (prec + rec + 1e-12)
    
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion": cm
    }

# -----------------------------
# Simple KNN (NumPy-only)
# -----------------------------
def knn_predict(X_train, y_train, X_test, k=5):
    preds = []
    for x in X_test:
        dists = np.sqrt(((X_train - x)**2).sum(axis=1))
        idx = np.argsort(dists)[:k]
        votes = y_train[idx]
        pred = 1 if votes.mean() >= 0.5 else 0
        preds.append(pred)
    return np.array(preds)