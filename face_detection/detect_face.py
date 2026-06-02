import os
from ultralytics import YOLO

# Khởi tạo mô hình YOLO chuyên nhận diện khuôn mặt (đã được tải về)
DIR = os.path.dirname(os.path.abspath(__file__))
# Lấy file yolov8n-face.pt ở thư mục gốc (mask-detection)
model_path = os.path.join(DIR, "..", "yolov8n-face.pt")
model = YOLO(model_path)

def detect_face(frame, conf_threshold=0.5):
    """
    Nhận diện khuôn mặt trong ảnh dùng YOLOv8-Face.
    Trả về danh sách các bounding box (x1, y1, x2, y2).
    """
    # Vì đây là model Face chuyên dụng, class 0 chính là khuôn mặt
    results = model(frame, imgsz=640, conf=conf_threshold, classes=[0], verbose=False)
    faces = []

    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            
            # Đảm bảo box hợp lệ
            if x2 > x1 and y2 > y1:
                faces.append((x1, y1, x2, y2))

    return faces