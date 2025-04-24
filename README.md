# Phát Hiện Ung Thư Phổi Từ Ảnh CT Scan

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/)
[![Libraries](https://img.shields.io/badge/libraries-OpenCV%2CSklearn%2CPIL-yellow.svg)](https://pypi.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Giới Thiệu

Đề tài này xây dựng một hệ thống phát hiện ung thư phổi từ ảnh chụp cắt lớp vi tính (CT scan) sử dụng các kỹ thuật xử lý ảnh và học máy. Hệ thống bao gồm một script để huấn luyện mô hình phân loại và một ứng dụng giao diện người dùng đồ họa (GUI) đơn giản để thực hiện dự đoán trên ảnh mới.

## Các Thư Viện Sử Dụng

Dự án này sử dụng các thư viện Python sau:

* `opencv-python` (cv2)
* `scikit-learn` (sklearn)
* `scikit-image`
* `joblib`
* `tqdm`
* `argparse`
* `matplotlib.pyplot`
* `imagehash`
* `PIL` (Pillow)
* `tkinter`

## Cách Hoạt Động Của Code

### 1. Huấn luyện mô hình

Script huấn luyện (`train.py`) thực hiện các bước sau:

1.  **Chuẩn bị dữ liệu:** Tải và tiền xử lý ảnh từ thư mục `dataset`, chia thành thư mục `normal` và `cancer`. Tùy chọn tăng cường dữ liệu (`--augment`) có thể được sử dụng.
2.  **Trích xuất đặc trưng:** Sử dụng Histogram of Oriented Gradients (HOG) và Local Binary Patterns (LBP) để trích xuất các đặc trưng từ ảnh. Mức độ chi tiết của việc trích xuất có thể được điều chỉnh bằng tham số `--detailed`.
3.  **Chuẩn bị dữ liệu:** Chia dữ liệu thành tập huấn luyện và tập kiểm thử. Có tùy chọn phân chia dữ liệu theo bệnh nhân (`--group_split`) để tránh rò rỉ dữ liệu. Dữ liệu được chuẩn hóa bằng `StandardScaler` và giảm chiều bằng Principal Component Analysis (PCA).
4.  **Huấn luyện mô hình:** Huấn luyện mô hình Support Vector Machine (SVM). Có các tùy chọn để sử dụng tìm kiếm lưới (`--grid_search`) để tìm tham số tối ưu hoặc sử dụng class weights tùy chỉnh (`--custom_weights`) để xử lý dữ liệu không cân bằng.
5.  **Đánh giá mô hình:** Đánh giá hiệu suất của mô hình trên tập huấn luyện và kiểm thử.
6.  **Lưu mô hình:** Lưu mô hình đã huấn luyện, scaler và PCA vào thư mục `models` để sử dụng cho việc dự đoán.
7.  **Ghi log:** Ghi lại thông tin và lỗi trong quá trình huấn luyện vào file `app.log`.

#### Xử lý tham số dòng lệnh

Script huấn luyện sử dụng thư viện `argparse` để xử lý các tham số dòng lệnh, cho phép người dùng tùy chỉnh quá trình huấn luyện và dự đoán. Dưới đây là mô tả các tham số:

* `--grid_search`: Kích hoạt quá trình tìm kiếm tham số tối ưu cho mô hình SVM bằng `GridSearchCV`. Nếu được sử dụng, script sẽ thử nghiệm nhiều tổ hợp tham số để tìm ra cấu hình tốt nhất cho mô hình.
* `--visualize_features`: Nếu được kích hoạt, script sẽ hiển thị trực quan các bản đồ đặc trưng HOG và LBP của một số ảnh mẫu từ dataset. Điều này giúp hiểu rõ hơn về các đặc trưng mà mô hình đang học.
* `--augment`: Bật tính năng tăng cường dữ liệu (Data Augmentation). Khi được sử dụng, script sẽ tạo thêm các phiên bản biến đổi của ảnh huấn luyện (ví dụ: lật, xoay nhẹ) để tăng kích thước và tính đa dạng của dataset, giúp mô hình tổng quát hóa tốt hơn.
* `--custom_weights`: Kích hoạt việc sử dụng class weights tùy chỉnh cho mô hình SVM. Điều này hữu ích khi dataset bị mất cân bằng giữa các lớp (ví dụ: số lượng ảnh ung thư và bình thường khác nhau đáng kể).
* `--group_split`: Sử dụng phương pháp `GroupShuffleSplit` để chia dữ liệu thành tập huấn luyện và kiểm thử dựa trên ID bệnh nhân (giả định ID bệnh nhân là phần đầu của tên file ảnh). Điều này đảm bảo rằng không có dữ liệu từ cùng một bệnh nhân xuất hiện ở cả hai tập, tránh rò rỉ dữ liệu.
* `--detailed`: Sử dụng các tham số chi tiết hơn khi trích xuất đặc trưng HOG và LBP. Điều này có thể giúp mô hình nắm bắt các đặc trưng vi mô hơn trong ảnh.
* `--predict <đường dẫn ảnh hoặc thư mục>`: Chuyển script sang chế độ dự đoán. Nếu bạn cung cấp đường dẫn đến một file ảnh hoặc một thư mục chứa ảnh, script sẽ tải mô hình đã được huấn luyện và thực hiện dự đoán trên các ảnh này, in ra nhãn dự đoán và độ tin cậy.

**Cách chạy huấn luyện với các tham số:**

Mở terminal hoặc command prompt, điều hướng đến thư mục chứa file `train.py` và chạy lệnh `python train.py` kèm theo các tham số mong muốn. Ví dụ:

* Chạy huấn luyện với tìm kiếm lưới và tăng cường dữ liệu:
    ```bash
    python train.py --grid_search --augment
    ```
* Chạy huấn luyện với phân chia theo nhóm bệnh nhân và sử dụng class weights tùy chỉnh:
    ```bash
    python train.py --group_split --custom_weights
    ```

### 2. Giao diện người dùng

Script giao diện (`gui.py`) cung cấp một ứng dụng đồ họa đơn giản để người dùng có thể:

1.  **Tải ảnh:** Chọn một file ảnh CT scan từ máy tính.
2.  **Hiển thị ảnh:** Xem ảnh gốc và ảnh đã qua xử lý (resize) trên giao diện.
3.  **Dự đoán:** Sử dụng mô hình đã được huấn luyện (đã lưu trong thư mục `models`) để dự đoán xem ảnh có dấu hiệu ung thư phổi hay không.
4.  **Hiển thị kết quả:** Hiển thị nhãn dự đoán ("BÌNH THƯỜNG" hoặc "UNG THƯ") và độ tin cậy của dự đoán.

**Cách chạy giao diện:**

Đảm bảo bạn đã huấn luyện mô hình thành công và các file model (`lung_cancer_svm.joblib`, `scaler.joblib`, `pca.joblib`) tồn tại trong thư mục `models`. Sau đó, mở terminal hoặc command prompt, điều hướng đến thư mục chứa file `gui.py` và chạy lệnh:

```bash
[1] Cài đặt thư viện
    pip install opencv-python scikit-learn scikit-image joblib tqdm argparse matplotlib imagehash
[2] Chạy file train
    python training_script.py
[3] chạy file app
    python inference_script.py 
```
Demo:
![image](https://github.com/user-attachments/assets/54820537-7815-4ce0-9b33-c781c3af28a7)
Hình 1:  test trên tập dữ liệu mới mẫu 1
![image](https://github.com/user-attachments/assets/19818075-4c5d-462c-b3ac-137e4eea74e2)
Hình 2: test trên dữ liệu mới mẫu 2

