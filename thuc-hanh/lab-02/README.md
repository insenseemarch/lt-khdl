# Lab 02 – Data Exploration, Preprocessing and Logistic Regression with NumPy

Dự án thực hiện quy trình xử lý dữ liệu và xây dựng mô hình Logistic Regression bằng NumPy, bao gồm:  
- Khám phá dữ liệu (EDA)  
- Tiền xử lý dữ liệu  
- Xây dựng mô hình học máy  
- Đánh giá mô hình và trực quan hoá  
- Kiểm định thống kê  

Dataset sử dụng: HR Analytics – Job Change of Data Scientists (Kaggle).

---

## Mục lục
1. Giới thiệu  
2. Dataset  
3. Method  
4. Installation & Setup  
5. Usage  
6. Results  
7. Project Structure  
8. Challenges & Solutions  
9. Future Improvements  
10. Contributors  
11. License  

---

## 1. Giới thiệu

### Bài toán
Dự đoán ứng viên trong lĩnh vực Data Science có mong muốn đổi việc hay không (binary classification), dựa trên dữ liệu nhân khẩu học, học vấn và kinh nghiệm làm việc.

### Động lực & Ứng dụng
- Hỗ trợ doanh nghiệp giữ chân nhân sự.  
- Phân tích yếu tố ảnh hưởng đến quyết định rời bỏ.  
- Tối ưu chiến lược nhân sự và đào tạo.

### Mục tiêu
- Thực hiện đầy đủ pipeline xử lý dữ liệu bằng NumPy.  
- Cài đặt Logistic Regression từ đầu mà không dùng sklearn.  
- Hiểu sâu dữ liệu thông qua EDA và trực quan hoá.  
- Đánh giá mô hình bằng các chỉ số và kiểm định thống kê.

---

## 2. Dataset

**Nguồn:** Kaggle  
Link: https://www.kaggle.com/datasets/arashnic/hr-analytics-job-change-of-data-scientists

### Kích thước
- 19,158 dòng  
- 14 features + 1 target  

### Mô tả các features
| Feature | Mô tả |
|--------|-------|
| enrollee_id | ID ứng viên |
| city | Mã thành phố |
| city_development_index | Chỉ số phát triển của thành phố |
| gender | Giới tính |
| relevent_experience | Kinh nghiệm liên quan |
| enrolled_university | Loại chương trình học |
| education_level | Trình độ học vấn |
| major_discipline | Chuyên ngành học |
| experience | Số năm kinh nghiệm |
| company_size | Quy mô công ty |
| company_type | Loại công ty |
| last_new_job | Thời điểm đổi việc gần nhất |
| training_hours | Số giờ đào tạo |
| target | 1 = muốn đổi việc, 0 = không |

---

## 3. Method

### Quy trình xử lý dữ liệu
1. Khám phá dữ liệu (EDA).  
2. Làm sạch giá trị không hợp lệ (“<1”, “>20”, giá trị rỗng).  
3. Xử lý missing values:  
   - Numeric: median  
   - Categorical: mode  
4. Phát hiện và xử lý outlier bằng IQR clipping.  
5. Label Encoding cho categorical features.  
6. Scaling: Min-Max và Standardization.  
7. Feature engineering: tạo feature mới `cdi_x_training`.

### Logistic Regression (NumPy)

**Sigmoid:**

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

**Binary Cross-Entropy Loss:**

\[
L = -\frac{1}{n}\sum_{i=1}^n \left[y_i\log(p_i) + (1 - y_i)\log(1 - p_i)\right]
\]

**Gradient Descent:**

\[
w := w - \alpha \cdot \frac{X^T (p - y)}{n}
\]

**Thêm L2 Regularization:**

\[
w := w - \alpha \left( \frac{X^T (p - y)}{n} + \lambda w \right)
\]

---

## 4. Installation & Setup

```bash
git clone <repository-url>
cd lab-02
pip install -r requirements.txt
```

---

## 5. Usage

```bash
jupyter notebook notebooks/01_data_exploration.ipynb
jupyter notebook notebooks/02_preprocessing.ipynb
jupyter notebook notebooks/03_modeling.ipynb
```

---

## 6. Results

---

## 7. Project Structure

lab-02/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
└── src/
    ├── data_processing.py
    ├── models.py
    └── visualization.py

---

## 8. Challenges & Solutions

---

## 9. Future Improvements
- Thử thêm các mô hình khác: KNN, Naive Bayes, SVM (thuần NumPy).

- Thêm PCA để giảm chiều dữ liệu.

- ROC-AUC evaluation.

- Sử dụng SMOTE để giải quyết imbalance.

- Hyperparameter tuning cho logistic regression.
---

## 10. Contributors

---

## 11. License