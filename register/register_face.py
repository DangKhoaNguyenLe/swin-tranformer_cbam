import cv2
import os
import time
from face_detection.detect_face import detect_face

# Nhận tên người dùng để tạo thư mục lưu ảnh
name = input("Nhập tên của bạn: ")
path = f"dataset/persons/{name}"

# Tạo thư mục tự động nếu chưa có
os.makedirs(path, exist_ok=True)

# Khởi động webcam
cap = cv2.VideoCapture(0)
count = 0

# Các bước đăng ký theo yêu cầu
steps = [
    ("Nhin thang (Straight)", 20),
    ("Quay trai (Turn Left)", 20),
    ("Quay phai (Turn Right)", 20)
]

print(f"\n--- Bat dau dang ky cho: {name} ---")
print("He thong se tu dong nhan dien khuon mat va chup anh.")

for step_name, target_count in steps:
    print(f"\nYeu cau: {step_name}")
    
    # 1. Giai đoạn chuẩn bị (Giu trong 2 giây khi thay mat)
    start_time = None
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # Lat ngang cho de canh chinh
        faces = detect_face(frame)
        
        h, w, _ = frame.shape
        if faces:
            # Lay khuon mat lon nhat
            (x1, y1, x2, y2) = max(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            if start_time is None:
                start_time = time.time()
            
            elapsed = time.time() - start_time
            remaining = max(0, 2 - elapsed)
            
            cv2.putText(frame, f"Giu nguyen: {step_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Bat dau sau: {remaining:.1f}s", (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if elapsed >= 2:
                break
        else:
            start_time = None
            cv2.putText(frame, "KHONG THAY MAT - Vui long dua mat vao khung hinh", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Dang Ky Khuon Mat", frame)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            exit()

    # 2. Giai đoạn chụp ảnh (10 ảnh)
    print(f"Dang tu dong chup 10 anh...")
    captured = 0
    while captured < target_count:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        faces = detect_face(frame)
        
        if faces:
            # Lay khuon mat va cat (crop)
            (x1, y1, x2, y2) = max(faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
            
            # Them margin 20%
            mw = int((x2 - x1) * 0.2)
            mh = int((y2 - y1) * 0.2)
            face_img = frame[max(0, y1-mh):min(h, y2+mh), max(0, x1-mw):min(w, x2+mw)]
            
            if face_img.size > 0:
                cv2.imwrite(f"{path}/{count}.jpg", face_img)
                count += 1
                captured += 1
                
                # Hien thi phan hoi
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, f"DANG CHUP: {captured}/{target_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                
                cv2.imshow("Dang Ky Khuon Mat", frame)
                cv2.waitKey(100) # Do tre giua cac anh

print(f"\nHoan tat! Da luu {count} anh vao {path}")
cap.release()
cv2.destroyAllWindows()
