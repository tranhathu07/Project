# Phân Loại Email (Email Classification)

Bài tập phân loại Email dựa trên mô hình **Gaussian Naive Bayes (GaussianNB)** và các bước tiền xử lý văn bản (preprocessing) đơn giản.

## Giới thiệu dự án
Dự án này ứng dụng thuật toán học máy cơ bản để giải quyết bài toán phân loại văn bản (Text Classification). Mục tiêu là xây dựng một mô hình có khả năng phân tích nội dung văn bản và dự đoán chính xác nhãn của email (ví dụ: Thư rác - Spam hay Thư thường - Ham).

## Thư viện và Kỹ thuật sử dụng
* **Ngôn ngữ & Môi trường:** Python, Jupyter Notebook (Kaggle).
* **Mô hình cốt lõi:** `GaussianNB` (từ thư viện `scikit-learn`).
* **Tiền xử lý (Preprocessing):** 
  * Chuyển đổi chữ thường, loại bỏ dấu câu và ký tự đặc biệt.
  * Lọc các từ dừng (stop words).
* **Trích xuất đặc trưng (Feature Creation):** 
  * Biến đổi văn bản thành các vector số học dựa trên từ điển (Dictionary) để tương thích với đầu vào của GaussianNB.
* **Xử lý nhãn (Label Encoding):** 
  * Sử dụng `LabelEncoder` để mã hóa nhãn huấn luyện và dịch ngược (`inverse_transform`) kết quả dự đoán.

## Cấu trúc thư mục
* `Naiive_Email_Classification.ipynb`: Mã nguồn chính (Notebook) chứa toàn bộ quá trình đọc dữ liệu, tiền xử lý, huấn luyện và đánh giá mô hình.
* `README.md`: Tài liệu mô tả dự án.
* `Dataset.zip`: Dataset được sử dụng
