import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import numpy as np
import cv2
from joblib import load
from skimage.feature import hog, local_binary_pattern

class LungCancerApp:
    def __init__(self, master):
        self.master = master
        master.title("Phát Hiện Ung Thư Phổi")
        master.geometry("800x600")
        
        # Load model, scaler và PCA
        try:
            self.model = load("models/lung_cancer_svm.joblib")
            self.scaler = load("models/scaler.joblib")
            self.pca = load("models/pca.joblib")
        except Exception as e:
            print("Lỗi tải model, scaler hoặc PCA:", e)
            exit()
        
        # Tạo giao diện
        self.create_widgets()
        
    def create_widgets(self):
        # Frame chính
        main_frame = ttk.Frame(self.master, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Phần upload ảnh
        upload_frame = ttk.LabelFrame(main_frame, text="Tải lên ảnh CT Scan", padding=15)
        upload_frame.pack(fill=tk.X, pady=10)
        
        self.upload_btn = ttk.Button(
            upload_frame,
            text="Chọn Ảnh",
            command=self.load_image
        )
        self.upload_btn.pack(pady=10)
        
        # Tạo 2 LabelFrame riêng cho ảnh gốc và ảnh sau xử lý
        images_frame = ttk.Frame(upload_frame)
        images_frame.pack(pady=10, fill=tk.X)
        
        orig_frame = ttk.LabelFrame(images_frame, text="Ảnh gốc", padding=10)
        orig_frame.pack(side=tk.LEFT, expand=True, padx=10)
        self.image_label = ttk.Label(orig_frame)
        self.image_label.pack()
        
        proc_frame = ttk.LabelFrame(images_frame, text="Ảnh sau xử lý", padding=10)
        proc_frame.pack(side=tk.LEFT, expand=True, padx=10)
        self.processed_image_label = ttk.Label(proc_frame)
        self.processed_image_label.pack()
        
        # Phần kết quả
        result_frame = ttk.LabelFrame(main_frame, text="Kết Quả Dự Đoán", padding=15)
        result_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.result_text = tk.StringVar()
        self.result_label = ttk.Label(
            result_frame,
            textvariable=self.result_text,
            font=('Helvetica', 14),
            wraplength=500
        )
        self.result_label.pack(pady=20)
        
        # Phần hướng dẫn
        info_text = """
        Hướng dẫn sử dụng:
        1. Nhấn 'Chọn Ảnh' để tải lên ảnh chụp CT
        2. Định dạng hỗ trợ: JPG, PNG
        3. Kết quả sẽ hiển thị dự đoán cuối cùng
        """
        info_label = ttk.Label(main_frame, text=info_text, justify=tk.LEFT)
        info_label.pack(fill=tk.X)
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Tệp ảnh", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            try:
                # Lấy vector đặc trưng của ảnh đã qua xử lý dùng cho dự đoán
                processed_features = self.process_image(file_path)
                
                # Hiển thị ảnh gốc
                original_img = Image.open(file_path)
                original_img.thumbnail((300, 300))
                photo_orig = ImageTk.PhotoImage(original_img)
                self.image_label.configure(image=photo_orig)
                self.image_label.image = photo_orig
                
                # Hiển thị ảnh đã qua xử lý
                proc_img = self.get_processed_image_for_display(file_path)
                proc_img.thumbnail((300, 300))
                photo_proc = ImageTk.PhotoImage(proc_img)
                self.processed_image_label.configure(image=photo_proc)
                self.processed_image_label.image = photo_proc
                
                # Dự đoán
                p_normal, p_cancer = self.predict_image(processed_features)
                if p_normal > p_cancer:
                    result = f"Dự đoán: BÌNH THƯỜNG ({p_normal*100:.1f}%)"
                    result_color = "#00aa00"
                else:
                    result = f"Dự đoán: UNG THƯ ({p_cancer*100:.1f}%)"
                    result_color = "#ff0000"
                    
                self.result_text.set(result)
                self.result_label.configure(foreground=result_color)
                
            except Exception as e:
                self.result_text.set(f"Lỗi: {str(e)}")
                self.result_label.configure(foreground="#000000")
    
    def process_image(self, file_path):
        """
        Hàm trích xuất đặc trưng từ ảnh dùng cho dự đoán.
        Quy trình: đọc ảnh ở chế độ grayscale, resize về 512x512 và chuẩn hóa (chia 255).
        Sau đó, trích xuất HOG và LBP rồi kết hợp lại.
        """
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Tệp ảnh không hợp lệ")
            
        img = cv2.resize(img, (512, 512))
        img = img / 255.0
        
        hog_feat = hog(
            img,
            orientations=16,
            pixels_per_cell=(16, 16),
            cells_per_block=(3, 3),
            channel_axis=None
        )
        
        lbp = local_binary_pattern(img, 24, 3, method="uniform")
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 24+3), range=(0, 24+2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        
        combined_features = np.hstack([hog_feat, hist])
        return combined_features
    
    def get_processed_image_for_display(self, file_path):
        """
        Hàm xử lý ảnh để hiển thị ảnh đã qua xử lý.
        Quy trình: đọc ảnh ở chế độ grayscale, resize về 512x512, chuẩn hóa như dùng cho mô hình,
        sau đó chuyển về dạng uint8 để hiển thị.
        """
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Tệp ảnh không hợp lệ")
        img = cv2.resize(img, (512, 512))
        img = (img / 255.0 * 255).astype(np.uint8)
        return Image.fromarray(img)
    
    def predict_image(self, features):
        scaled_features = self.scaler.transform([features])
        pca_features = self.pca.transform(scaled_features)
        proba = self.model.predict_proba(pca_features)
        p_normal = proba[0][0]
        p_cancer = proba[0][1]
        return p_normal, p_cancer

if __name__ == "__main__":
    root = tk.Tk()
    app = LungCancerApp(root)
    root.mainloop()
