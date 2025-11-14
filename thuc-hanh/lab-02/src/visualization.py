# src/visualization.py
# Các hàm trực quan hoá phục vụ cho notebook 01 và 03
# Tối ưu: clean, dễ dùng, không dùng loop thừa

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", rc={"figure.figsize": (8, 5)})

# ============================================================
# 1) Histogram cho biến numeric
# ============================================================
def plot_hist_numeric(arr, title="", xlabel=""):
    arr = arr.astype(float)
    plt.figure()
    sns.histplot(arr, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.show()
    plt.close()

# ============================================================
# 2) Biểu đồ Top K categorical
# ============================================================
def plot_top_categories(vals, counts, title="", xlabel=""):
    k = len(vals)
    plt.figure()
    sns.barplot(x=counts, y=vals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Category")
    plt.show()
    plt.close()

# ============================================================
# 3) Pie chart đơn giản
# ============================================================
def plot_pie(labels, sizes, title=""):
    plt.figure()
    plt.pie(sizes, labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.show()
    plt.close()

# ============================================================
# 4) Scatter plot
# ============================================================
def plot_scatter(x, y, xlabel="", ylabel="", title=""):
    plt.figure()
    plt.scatter(x, y, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
    plt.close()

# ============================================================
# 5) Heatmap tương quan
# ============================================================
def plot_corr_heatmap(data, col_names):
    data = data.astype(float)
    corr = np.corrcoef(data.T)
    plt.figure(figsize=(10, 7))
    sns.heatmap(corr, annot=False, cmap="coolwarm", xticklabels=col_names, yticklabels=col_names)
    plt.title("Correlation Heatmap")
    plt.show()
    plt.close()

# ============================================================
# Kết thúc file
# ============================================================
