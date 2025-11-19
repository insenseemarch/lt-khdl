import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Histogram for numeric column
# -----------------------------
def plot_hist_numeric(arr, title="Histogram", bins=30):
    arr = np.asarray(arr, float)
    arr = arr[~np.isnan(arr)]
    plt.figure(figsize=(6,4))
    plt.hist(arr, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

# -----------------------------
# Bar plot for categorical counts
# -----------------------------
def plot_bar_counts(cat_arr, title="Bar Chart"):    
    cat_arr = np.asarray(cat_arr, dtype=str)
    vals, counts = np.unique(cat_arr, return_counts=True)
    plt.figure(figsize=(6,4))
    plt.bar(vals, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.show()

# -----------------------------
# Scatter plot for two numeric columns
# -----------------------------
def plot_scatter(x, y, title="Scatter Plot", xlabel="X", ylabel="Y"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    plt.figure(figsize=(6,4))
    plt.scatter(x[mask], y[mask], s=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.show()

# -----------------------------
# Heatmap for correlation matrix
# -----------------------------
def plot_corr_heatmap(corr_matrix, labels=None, title="Correlation Heatmap"):
    plt.figure(figsize=(6,5))
    plt.imshow(corr_matrix, cmap="Blues")
    plt.colorbar()
    if labels is not None:
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Top-N categorical bars
# -----------------------------
def plot_top_categories(cat_arr, N=10, title="Top Categories"):
    cat_arr = np.asarray(cat_arr, dtype=str)
    vals, counts = np.unique(cat_arr, return_counts=True)
    idx = np.argsort(-counts)[:N]
    vals = vals[idx]
    counts = counts[idx]
    plt.figure(figsize=(6,4))
    plt.bar(vals, counts)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.show()