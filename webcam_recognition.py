import cv2
import pickle

from face_detection.detect_face import detect_face
from face_recognition.swin_embedding import get_embedding
from utils.similarity import compare

# Tải cơ sở dữ liệu (từ file pkl) chứa danh sách embedding đã lưu
database = pickle.load(open("database/embeddings.pkl", "rb"))

# Mở webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Phát hiện tất cả khuôn mặt trong khung hình
    faces = detect_face(frame)

    for (x1, y1, x2, y2) in faces:
        face = frame[y1:y2, x1:x2]
        
        # Trích xuất embedding cho khuôn mặt hiện tại
        emb = get_embedding(face)
        name = "Unknown"
        best = 0

        # So sánh và tìm người khớp nhất
        person_results = []
        for person in database:
            all_scores = [compare(emb, db_emb) for db_emb in database[person]]
            top_scores = sorted(all_scores, reverse=True)[:3]
            avg_score = sum(top_scores) / len(top_scores) if top_scores else 0
            person_results.append((avg_score, person))

        person_results.sort(reverse=True)
        name = "Unknown"
        best = 0

        if person_results:
            best, name = person_results[0]
            
            # Kiểm tra ngưỡng và khoảng cách an toàn (Margin)
            if best < 0.70:
                name = "Unknown"
            elif len(person_results) > 1:
                second_best, _ = person_results[1]
                if (best - second_best) < 0.05:
                    name = f"Wait ({name}?)"

        # Vẽ khung và tên quanh khuôn mặt
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({best:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Nhấn ESC để thoát
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()