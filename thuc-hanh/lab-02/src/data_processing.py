# src/data_processing.py
# Tập trung toàn bộ hàm xử lý dữ liệu cho notebook 01 + 02
# Viết gọn – rõ – vectorized – không dùng loop thừa

import numpy as np

# ============================================================
# 1) Đọc CSV bằng NumPy (vectorized)
# ============================================================
def load_csv_numpy(path, delimiter=","):
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    headers = np.array(lines[0].split(delimiter), dtype=object)
    data_lines = np.array(lines[1:], dtype=object)

    split_data = np.char.split(data_lines, delimiter)
    max_len = len(headers)

    padded = np.array([
        row + [""] * (max_len - len(row)) if len(row) < max_len else row[:max_len]
        for row in split_data
    ], dtype=object)

    return headers, padded

# ============================================================
# 2) Missing values
# ============================================================
def missing_mask(col):
    col_str = np.char.strip(col.astype(str))
    lower = np.char.lower(col_str)
    return np.isin(lower, np.array([""], dtype=object))

# fill bằng mode

def fill_mode(col):
    nonmiss = col[col != ""]
    uniq, counts = np.unique(nonmiss, return_counts=True)
    mode = uniq[np.argmax(counts)]
    return np.where(col == "", mode, col)

# ============================================================
# 3) Xử lý giá trị không hợp lệ (experience, last_new_job)
# ============================================================
def clean_experience(col):
    c = np.char.strip(col.astype(str))
    c = np.where(c == "<1", "0", c)
    c = np.where(c == ">20", "21", c)
    c = np.where(c == "", "nan", c)
    return c

def clean_last_new_job(col):
    c = np.char.strip(col.astype(str))
    c = np.where(c == "never", "0", c)
    c = np.where(c == ">4", "5", c)
    c = np.where(c == "", "nan", c)
    return c

# ============================================================
# 4) Encode categorical (Label Encoding)
# ============================================================
def label_encode(col):
    uniq, inv = np.unique(col.astype(str), return_inverse=True)
    return inv.astype(float)

# ============================================================
# 5) Outlier detection & clip
# ============================================================
def detect_outlier_iqr(arr):
    Q1 = np.percentile(arr, 25)
    Q3 = np.percentile(arr, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (arr < lower) | (arr > upper)
    return mask, lower, upper

# clip outlier thay vì xoá

def remove_outlier_iqr(arr):
    mask, lower, upper = detect_outlier_iqr(arr)
    return np.clip(arr, lower, upper)

# ============================================================
# 6) Scaling
# ============================================================
def minmax(arr):
    amin, amax = arr.min(), arr.max()
    return (arr - amin) / (amax - amin + 1e-9)

def standardize(arr):
    mean = arr.mean()
    std = arr.std() + 1e-9
    return (arr - mean) / std

# ============================================================
# 7) Train-test split (vectorized)
# ============================================================
def train_test_split(X, y, test_size=0.2, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    cut = int(len(X) * (1 - test_size))
    train_idx, test_idx = idx[:cut], idx[cut:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# ============================================================
# File kết thúc
# ============================================================
