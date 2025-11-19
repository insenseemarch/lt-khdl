import numpy as np

# -----------------------------
# CSV loader
# -----------------------------
def load_csv_numpy(path, delimiter=",", encoding="utf-8"):
    raw = np.genfromtxt(path, dtype=str, delimiter=delimiter, encoding=encoding, autostrip=True)
    if raw.ndim == 1:
        headers = list(raw)
        data = np.empty((0, len(headers)), dtype=object)
    else:
        headers = list(raw[0])
        data = raw[1:].astype(object)
    return np.array(headers), data

# -----------------------------
# Missing value
# -----------------------------
def missing_mask(arr):
    arr = np.asarray(arr, dtype=object)
    s = np.char.lower(arr.astype(str))
    return (arr == None) | (arr == "") | (s == "nan") | (s == "none")

def mode_np(arr):
    arr = np.asarray(arr)
    vals, counts = np.unique(arr, return_counts=True)
    return vals[np.argmax(counts)]

# -----------------------------
# Numeric conversion
# -----------------------------
def to_numeric_col(col, fill_nan_with=np.nan):
    a = np.array(col, dtype=str)
    out = np.full(a.shape, np.nan, float)
    for i, v in enumerate(a):
        v = v.strip()
        if v == "" or v.lower() in ("nan", "none"):
            out[i] = np.nan
            continue
        try:
            out[i] = float(v)
        except:
            out[i] = np.nan
    if not np.isnan(fill_nan_with):
        out = np.where(np.isnan(out), fill_nan_with, out)
    return out

# -----------------------------
# Clean special categorical numeric-like values
# -----------------------------
def clean_special_values(data, headers):
    h = list(headers)

    # experience column
    if "experience" in h:
        i = h.index("experience")
        col = np.char.strip(data[:, i].astype(str))
        col = np.where(col == "<1", "0", col)
        col = np.where(col == ">20", "21", col)
        col = np.where(col == "", "nan", col)
        data[:, i] = col

    # last_new_job
    if "last_new_job" in h:
        i = h.index("last_new_job")
        col = np.char.strip(data[:, i].astype(str))
        col = np.where(col == "never", "0", col)
        col = np.where(col == ">4", "5", col)
        col = np.where(col == "", "nan", col)
        data[:, i] = col

# -----------------------------
# Label encoding
# -----------------------------
def label_encode_np(col):
    vals = np.unique(col.astype(str))
    mapping = {v: i for i, v in enumerate(vals)}
    encoded = np.array([mapping[v] for v in col.astype(str)], dtype=int)
    return encoded, vals

# -----------------------------
# Outlier handling (IQR clipping)
# -----------------------------
def clip_iqr_np(arr):
    arr = np.asarray(arr, float)
    q1 = np.nanpercentile(arr, 25)
    q3 = np.nanpercentile(arr, 75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    out = arr.copy()
    mask = ~np.isnan(out)
    out[mask] = np.clip(out[mask], low, high)
    return out

# -----------------------------
# Scaling
# -----------------------------
def minmax_np(arr):
    arr = np.asarray(arr, float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn < 1e-12:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

def standardize_np(arr):
    arr = np.asarray(arr, float)
    mu = np.nanmean(arr)
    sd = np.nanstd(arr)
    if sd < 1e-12: sd = 1.0
    return (arr - mu) / sd

# -----------------------------
# Feature engineering
# -----------------------------
def add_feature_cdi_x_training(data, headers):
    h = list(headers)
    if ("city_development_index" in h) and ("training_hours" in h):
        i_cdi = h.index("city_development_index")
        i_tr = h.index("training_hours")
        cdi = to_numeric_col(data[:, i_cdi])
        tr = to_numeric_col(data[:, i_tr])
        new = (cdi * tr).reshape(-1,1)
        data_new = np.hstack([data, new.astype(object)])
        headers_new = np.append(headers, "cdi_x_training")
        return data_new, headers_new
    return data, headers

def normal_cdf(x):
    return 0.5 * (1.0 + erf_approx(x / np.sqrt(2)))