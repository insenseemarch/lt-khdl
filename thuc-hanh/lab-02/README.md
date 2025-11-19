# 🔍 Dự đoán Xu Hướng Đổi Việc Của Ứng Viên Data Science  
### *Lab 02 – Data Exploration → Preprocessing → Modeling (100% NumPy)*

Dự án này build trọn pipeline xử lý & phân tích dữ liệu ứng viên Data Science từ Kaggle, bao gồm:
- Khám phá dữ liệu (EDA)
- Tiền xử lý 100% dùng **NumPy**
- Feature Engineering dựa trên EDA
- Xây dựng mô hình ML: Logistic Regression & KNN từ đầu
- Visualization & thống kê kiểm định  
- Đánh giá mô hình bằng precision, recall, F1, confusion matrix, loss curve

---

## 2. Mục lục
1. [Giới thiệu](#giới-thiệu)  
2. [Dataset](#dataset)  
3. [Quy trình & Phương pháp](#quy-trình--phương-pháp)  
4. [Installation](#installation)  
5. [Cách chạy project](#cách-chạy-project)  
6. [Kết quả chính](#kết-quả-chính)  
7. [Cấu trúc project](#cấu-trúc-project)  
8. [Challenges](#challenges)  
9. [Future Improvements](#future-improvements)  
10. [Contributor](#contributor)  
11. [License](#license)

---

## 3. Giới thiệu

Bài toán: **Dự đoán ứng viên có muốn đổi việc hay không**, dựa vào thông tin cá nhân & kinh nghiệm — dùng Logistic Regression & KNN tự implement bằng NumPy.

Ý nghĩa thực tế:
- Hỗ trợ HR phân tích lý do rời bỏ công việc 
- Thấy rõ nhóm ứng viên nào có nguy cơ đổi việc  
- Tối ưu tuyển dụng & đào tạo nhân sự  

Pipeline được chia thành 3 phần tương ứng 3 notebook:
- `01_data_exploration.md` -> EDA  
- `02_preprocessing.md` -> Data cleaning + Encoding + Scaling + Feature Engineering 
- `03_modeling.md` -> Training + Evaluation + Visualization  

---

## 4. Dataset

Nguồn: [*HR Analytics – Job Change of Data Scientists* (Kaggle)](https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists)

**Size:** 2129 hàng × 14 cột (test dataset)

### Các feature chính:
| Feature | Ý nghĩa |
|--------|--------|
| `city` | Thành phố làm việc |
| `city_development_index` | Chỉ số phát triển (CDI) |
| `gender` | Giới tính |
| `relevent_experience` | Có kinh nghiệm liên quan không |
| `education_level` | Trình độ học vấn |
| `major_discipline` | Chuyên ngành học |
| `experience` | Số năm kinh nghiệm: `<1`, `>20` |
| `company_size` | Quy mô công ty |
| `training_hours` | Số giờ training |
| `target` | 1 = muốn đổi việc, 0 = không |

Đặc điểm nổi bật:
- Nhiều cột categorical có missing cao (`gender`, `major_discipline`, `company_size`, `company_type`)
- Numeric gần như đầy đủ, ngoại trừ outlier mạnh ở `training_hours`
- Target imbalance: 73% không đổi việc – 27% đổi việc

---

## 5. Quy trình & Phương pháp

### **5.1. Data Exploration**
Trích từ `01_data_exploration.md`:
- Kiểm tra missing, phân loại numeric/categorical  
- Thống kê mô tả (mean, median, std)  
- Phân phối training hours (bị skew)  
- Top city: **city_103** & **city_21**  
- Chênh lệch giới tính cực lớn (90% Male)  
- Phân tích theo:
  - Giới tính
  - Học vấn
  - Chuyên ngành
  - Kinh nghiệm
  - Lịch sử đổi job
  - CDI, training hours, relevant experience  
- Xác định yếu tố rủi ro cao:
  - CDI thấp -> dễ đổi việc  
  - Không có kinh nghiệm liên quan -> dễ đổi  
  - Training hours gần như không ảnh hưởng  

---

### **5.2. Preprocessing**  
Trích từ `02_preprocessing.md` (100% NumPy):

#### Làm sạch dữ liệu:
- Chuẩn hoá `<1 → 0` và `>20 = 21` cho `experience`
- `never = 0`, `>4 = 5` cho `last_new_job`

#### Xử lý missing:
- Numeric -> fill median  
- Categorical → fill mode  

#### Label Encoding NumPy:
Tự implement `label_encode_np()` cho các cột string.

#### Outlier handling:
- IQR clipping cho numeric  
- `training_hours` có outlier ~6% -> được clip phù hợp  

#### Normalization:
- Min-max cho `city_development_index` & `training_hours`  
- GIữ nguyên numeric đã clean nếu không cần scale (VD: `experience`)  

#### Feature Engineering:
1. **`cdi_x_training`**: tương tác hữu ích  
2. **`experience_level`**: binning theo insight của EDA  
3. **`job_stability_score`**: từ lịch sử đổi việc  

#### Save processed data:
`.npy` files cho modeling.

---

### **5.3. Modeling**  
Trích từ `03_modeling.md`:

#### Train / Val / Test split:
- 80% train  
- 10% validation  
- 10% test  
- Standardize theo train  

---

### **Logistic Regression – Implement bằng NumPy**

\[
p = \sigma(Wx + b)
\]
\[
Loss = - (y\ln p + (1-y)\ln(1-p)) + \frac{\lambda}{2}\|W\|^2
\]

Gradient descent + mini-batch.
- epochs = 200  
- lr = 0.1  
- L2 = 1e-3  

#### **Kết quả test:**
- Accuracy: **1.0**  
- Precision: **1.0**  
- Recall: **1.0**  
- F1-score: **1.0**  
- Confusion matrix:  
  - TP = 59  
  - FP = 0  
  - FN = 0  
  - TN = 154  

---

### KNN (NumPy)
K = 5  
- Accuracy: **98.6%**  
- Recall hơi thấp hơn LR  

---

### Visualization:
- Loss curve -> hội tụ đẹp  
- Precision–Recall curve -> rất cao, cân bằng  
- Confusion Matrix -> perfect  
- So sánh KNN vs Logistic -> Logistic thắng về sự ổn định  

---

## 6. Installation
```bash
conda create -n lab2 python=3.10 -y
conda activate lab2
conda install numpy matplotlib
conda install jupyter
jupyter notebook
```

---

## 7. Usage

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

---

## 8. Results

### 8.1. Kết quả Logistic Regression (NumPy)
- Accuracy: 1.00
- Precision: 1.00
- Recall: 1.00
- F1-score: 1.00
- Confusion Matrix:
    - TP = 59
    - FP = 0
    - FN = 0
    - TN = 154
Loss curve hội tụ rất ổn định qua 200 epochs.

---

### 8.2. Visualization
- Histogram từng feature
- Bar chart phân phối job change theo giới tính / kinh nghiệm / học vấn
- Heatmap correlation
- Scatter plot
- Loss Curve & PR Visualization

---

### 8.3. So sánh mô hình
| Model                     | Accuracy | Ưu điểm                |
|---------------------------|----------|-------------------------|
| Logistic Regression (NumPy) | **1.00**   | ổn định, nhanh         |
| KNN (K=5)                 | 0.986    | đơn giản, baseline tốt |

---

## 9. Project Structure

23120172/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── visualization.py
│   └── models.py

---

## 10. Challenges & Solutions

### 10.1. Không dùng Pandas

Tự viết:
- CSV loader
- Missing-value handler
- IQR outlier detector
- Label encoder

### 10.2. Không dùng sklearn

Logistic regression tự viết toàn bộ (loss, grad, update).

### 10.3. Không dùng SciPy

Viết lại `normal_cdf()`.

### 10.4. Không dùng seaborn

Tự build heatmap & biểu đồ bằng matplotlib raw.

---

## 11. Future Improvements
- Thử thêm các mô hình khác: Naive Bayes, SVM (thuần NumPy).

- Thêm PCA để giảm chiều dữ liệu.

- Early stopping cho Logistic Regression
---

## 12. Contributors
- MSSV: 23120172
- Họ và tên: Trần Thị Thủy Tiên
- Lớp: Chính quy, chiều thứ 2
- Email: 23120172@student.hcmus.edu.vn

---

## 11. License

No License