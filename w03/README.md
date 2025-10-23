# Báo cáo so sánh hiệu năng nhân ma trận

Đây là bài tập báo cáo so sánh tốc độ thực thi của phép nhân ma trận vuông bằng ba phương pháp khác nhau: C, Python thuần túy và NumPy.

## Cấu trúc thư mục

- `matrix_mult.c`: Mã nguồn C cho việc nhân ma trận.
- `matrix_mult.exe`: File thực thi cần thiết cho việc chạy code C trong Jupyter Notebook (**không khuyến khích xóa**).
- `23120172.ipynb`: File Jupyter Notebook chứa báo cáo chi tiết, mã nguồn Python, kết quả và phân tích.
- `README.md`: File hướng dẫn này.

## Hướng dẫn chạy chương trình

### Yêu cầu về môi trường

- Trình biên dịch C (ví dụ: `gcc`)  — có thể dùng MSYS2 UCRT64 hoặc MinGW-w64.
- Python 3+.
- Các thư viện Python: `numpy`, `pandas`, `matplotlib`, `seaborn`.

### Các bước thực thi

1.  **Cài đặt các thư viện Python cần thiết:**
    ```bash
    pip install numpy pandas matplotlib seaborn
    ```

2.  **Biên dịch file C và chạy Jupyter Notebook:**
    - Mở file `23120172.ipynb` bằng Jupyter Notebook.
    - Chạy code trong notebook bằng cách chọn `Kernel > Restart & Run All`. Đoạn script Python sẽ tự động thực hiện các công việc sau:
        - Biên dịch file `matrix_mult.c` thành file thực thi (`matrix_mult.exe` trên Windows hoặc `matrix_mult` trên Linux/macOS).
        - Thực thi file C để lấy kết quả thời gian chạy.
        - Chạy các phiên bản Python thuần và NumPy.

        - In ra bảng so sánh kết quả và vẽ biểu đồ trực quan hóa.
