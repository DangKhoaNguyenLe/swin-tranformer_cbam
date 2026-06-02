#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Điểm khởi đầu chính của Hệ thống Nhận diện Gương mặt.
Hỗ trợ các chức năng: Đăng ký, Nhận diện và Huấn luyện.
"""

# ── Thư viện chuẩn ──────────────────────────────────────────────
import pickle
import shutil
import time
from pathlib import Path

# ── Thư viện bên thứ ba ─────────────────────────────────────────
import cv2
import numpy as np
# ── Module nội bộ dự án ─────────────────────────────────────────
from face_detection.detect_face import detect_face
from face_recognition.swin_embedding import get_embedding
from utils.similarity import compare


# ══════════════════════════════════════════════════════════════
# CẤU HÌNH THU THẬP DỮ LIỆU
# Định dạng: (tên_góc, hướng_dẫn_hiển_thị, số_ảnh_mục_tiêu)
# ══════════════════════════════════════════════════════════════

# ── Đăng ký 3 góc theo yêu cầu: 40 ảnh gốc/góc ──────────
REGISTER_ANGLES = [
    ("CENTERED",    "Nhin THANG (Hay cu dong co mat nhe)",  40),
    ("LEFT",        "Quay TRAI (Gat dau nhe len xuong)",     40),
    ("RIGHT",       "Quay PHAI (Gat dau nhe len xuong)",     40),
]

def save_clean_faces(face_img, base_path, base_name, index):
    """Lưu ảnh gốc và ảnh lật ngang (Augment an toàn nhất cho DB)"""
    # Không dùng Blur hay Brightness ở đây vì Database gốc cần ảnh cực nét.
    # Nếu nạp ảnh Blur vào DB, khi có người lạ đứng mờ mờ, AI sẽ nhận nhầm.
    
    cv2.imwrite(f"{base_path}/{base_name}_{index:04d}_orig.jpg", face_img)
    
    # Lật ngang (Horizontal Flip)
    flip = cv2.flip(face_img, 1)
    cv2.imwrite(f"{base_path}/{base_name}_{index:04d}_flip.jpg", flip)
    
    return 2

# ── Bổ sung lần sau: 10 ảnh tập trung vào đa dạng biểu cảm ──────
AUGMENT_ANGLES = [
    # 4 ảnh thẳng với biểu cảm đa dạng là điểm mấu chốt
    ("CENTERED",   "Nhìn thẳng – MỈM CƯỜI tự nhiên",                 2),
    ("CENTERED",   "Nhìn thẳng – NGHIÊM TÚC, không cười",            2),
    # Ngang xa hơn để mô hình học được góc cực đoan
    ("RIGHT",      "Quay PHẢI thật xa (60-70°, gần như nghiêng hẳn)",2),
    ("LEFT",       "Quay TRÁI thật xa (60-70°, gần như nghiêng hẳn)",2),
    # Góc chéo bổ sung
    ("UP_LEFT",    "Ngẩng lên + quay TRÁI (góc chéo)",               1),
    ("DOWN_RIGHT", "Cúi xuống + quay PHẢI (góc chéo)",              1),
]

class HeadPoseDetector:
    """Phát hiện 9 góc xoay đầu bằng OpenCV Haarcascades.

    Góc được hỗ trợ:
        Thẳng  : CENTERED
        Ngang  : LEFT, RIGHT
        Dọc    : UP, DOWN
        Chéo   : UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT

    Chiến lược:
      - Mặt trực diện  → dùng vị trí MẮT (avg_eye_y_norm) xác định UP/DOWN/CENTERED,
        kết hợp trọng tâm khuôn mặt theo X để bắt góc chéo.
      - Mặt nghiêng    → profile cascade xác định LEFT/RIGHT (và ước lượng UP/DOWN
        từ vị trí Y của hộp mặt so với khung hình).
    """

    # Ngưỡng phân loại UP/DOWN dựa trên avg_eye_y_norm (tỉ lệ trên chiều cao mặt)
    _EYE_UP_THR   = 0.38   # mắt cao → đầu ngẩng
    _EYE_DOWN_THR = 0.50   # mắt thấp → đầu cúi

    # Ngưỡng phân loại LEFT/RIGHT dựa trên vị trí X tâm mặt trong khung hình
    # (tính theo tỉ lệ 0-1 của chiều rộng khung)
    _FACE_LEFT_THR  = 0.42  # tâm mặt thiên về bên phải màn hình (mirror) → quay PHẢI
    _FACE_RIGHT_THR = 0.58  # tâm mặt thiên về bên trái màn hình (mirror) → quay TRÁI

    def __init__(self):
        self.face_cascade    = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml')
        self.eye_cascade     = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')

    # ── helpers ──────────────────────────────────────────────────
    @staticmethod
    def _eye_y_norm(roi_gray: np.ndarray, eyes, face_h: int) -> float | None:
        """Trả về vị trí Y trung bình của mắt chuẩn hoá theo chiều cao mặt."""
        if len(eyes) == 0:
            return None
        return sum(ey + eh / 2.0 for _, ey, _, eh in eyes) / len(eyes) / face_h

    def _classify_vertical(self, eye_y: float | None,
                            roi_gray: np.ndarray, face_h: int) -> str:
        """Trả về 'UP' / 'DOWN' / 'CENTERED' từ vị trí mắt (hoặc Canny dự phòng)."""
        if eye_y is not None:
            if eye_y < self._EYE_UP_THR:   return "UP"
            if eye_y > self._EYE_DOWN_THR: return "DOWN"
            return "CENTERED"
        # Dự phòng Canny khi không thấy mắt (đeo kính loá, nhắm mắt…)
        edges = cv2.Canny(roi_gray, 50, 150)
        M = cv2.moments(edges)
        if M["m00"] != 0:
            cY = int(M["m01"] / M["m00"])
            norm_y = (cY / face_h) * 2 - 1
            if norm_y < -0.05: return "UP"
            if norm_y >  0.06: return "DOWN"
        return "CENTERED"

    def _classify_horizontal(self, face_cx_norm: float) -> str:
        """Trả về 'LEFT' / 'RIGHT' / '' từ vị trí tâm mặt theo chiều ngang."""
        if face_cx_norm < self._FACE_LEFT_THR:  return "RIGHT"  # mirror đảo chiều
        if face_cx_norm > self._FACE_RIGHT_THR: return "LEFT"
        return ""

    # ── API chính ────────────────────────────────────────────────
    def get_angle(self, frame: np.ndarray) -> tuple[str, tuple]:
        """Trả về (tên_góc, (x,y,w,h)) của khuôn mặt trong frame đã flip."""
        img_h, img_w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # ── Bước 1: thử phát hiện mặt trực diện ─────────────────
        front_faces = self.face_cascade.detectMultiScale(gray, 1.2, 5,
                                                         minSize=(60, 60))
        if len(front_faces) > 0:
            x, y, w, h = max(front_faces, key=lambda r: r[2] * r[3])  # chọn mặt to nhất
            roi_gray = gray[y:y+h, x:x+w]

            # Xác định hướng DỌC từ vị trí mắt
            eyes  = self.eye_cascade.detectMultiScale(roi_gray, 1.05, 4,
                                                      minSize=(15, 15))
            ey_n  = self._eye_y_norm(roi_gray, eyes, h)
            vert  = self._classify_vertical(ey_n, roi_gray, h)

            # Xác định hướng NGANG từ vị trí tâm mặt trong khung hình
            face_cx_norm = (x + w / 2.0) / img_w
            horiz = self._classify_horizontal(face_cx_norm)

            # Ghép thành 9 nhãn
            if horiz and vert != "CENTERED":
                return f"{vert}_{horiz}", (x, y, w, h)   # VD: UP_LEFT, DOWN_RIGHT
            if horiz:
                return horiz, (x, y, w, h)               # LEFT / RIGHT
            return vert, (x, y, w, h)                    # UP / DOWN / CENTERED

        # ── Bước 2: mặt nghiêng (profile) ────────────────────────
        # OpenCV profile cascade bắt mặt nhìn sang trái (trong ảnh KHÔNG flip).
        # Vì frame đã flip(1) thành mirror, nhãn được đảo ngược.
        profile_l = self.profile_cascade.detectMultiScale(gray, 1.2, 5,
                                                          minSize=(60, 60))
        if len(profile_l) > 0:
            x, y, w, h = max(profile_l, key=lambda r: r[2] * r[3])
            # Ước lượng góc DỌC từ vị trí Y hộp mặt trong khung hình
            face_cy_norm = (y + h / 2.0) / img_h
            if face_cy_norm < 0.38: return "UP_LEFT",   (x, y, w, h)
            if face_cy_norm > 0.62: return "DOWN_LEFT", (x, y, w, h)
            return "LEFT", (x, y, w, h)

        flipped = cv2.flip(gray, 1)
        profile_r = self.profile_cascade.detectMultiScale(flipped, 1.2, 5,
                                                          minSize=(60, 60))
        if len(profile_r) > 0:
            x, y, w, h = max(profile_r, key=lambda r: r[2] * r[3])
            actual_x   = img_w - (x + w)  # đổi tọa độ sau flip
            face_cy_norm = (y + h / 2.0) / img_h
            if face_cy_norm < 0.38: return "UP_RIGHT",   (actual_x, y, w, h)
            if face_cy_norm > 0.62: return "DOWN_RIGHT", (actual_x, y, w, h)
            return "RIGHT", (actual_x, y, w, h)

        return "NONE", None




class FaceRecognitionSystem:
    """Hệ thống nhận diện gương mặt đa góc độ."""

    # ── Hằng số mặc định ────────────────────────────────────────
    _BASE_DIR        = Path(__file__).resolve().parent
    DATABASE_PATH    = str(_BASE_DIR / "database" / "embeddings.pkl")
    DATASET_PATH     = str(_BASE_DIR / "dataset" / "persons")
    RECOG_THRESHOLD  = 0.75   # Điểm tối thiểu để nhận ra (Tăng lên để tránh nhận nhầm)
    CAPTURE_INTERVAL = 2

    def __init__(self):
        # Khởi tạo đường dẫn từ hằng số (dễ ghi đè trong subclass)
        self.database_path = self.DATABASE_PATH
        self.dataset_path  = self.DATASET_PATH
        self.recognition_threshold = self.RECOG_THRESHOLD

        # Tải database embeddings từ file (nếu đã có)
        self.database: dict = self._load_database()

    # ════════════════════════════════════════════════════════════
    # Quản lý Database
    # ════════════════════════════════════════════════════════════

    def _load_database(self) -> dict:
        """Tải database embeddings từ file pickle.

        Returns:
            dict: {tên_người: [embedding, ...]} hoặc {} nếu chưa có.
        """
        db_path = Path(self.database_path)
        if not db_path.exists():
            return {}

        try:
            # Dùng context manager để đảm bảo file được đóng đúng cách
            with open(db_path, "rb") as f:
                data = pickle.load(f)
            print(f"Đã tải database: {len(data)} người")
            return data
        except Exception as e:
            print(f"Lỗi khi tải database: {e}")
            return {}

    def _save_database(self) -> None:
        """Lưu database embeddings ra file pickle."""
        db_path = Path(self.database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(db_path, "wb") as f:
                pickle.dump(self.database, f)
            print("Database đã được lưu!")
        except Exception as e:
            print(f"Lỗi khi lưu database: {e}")

    # ════════════════════════════════════════════════════════════
    # Đăng ký gương mặt
    # ════════════════════════════════════════════════════════════

    def register_face(self) -> None:
        """Đăng ký gương mặt mới với hướng dẫn 4 góc (đúng yêu cầu 2s hold + 10 ảnh)."""
        print("\n" + "=" * 50)
        print("ĐĂNG KÝ GƯƠNG MẶT MỚI")
        print("=" * 50)

        name = input("Nhập tên của bạn: ").strip()
        if not name:
            print("Tên không được để trống!")
            return

        save_dir = Path(self.dataset_path) / name
        save_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return

        detector = HeadPoseDetector()
        summary   = []
        cancelled = False

        print("\nBẮT ĐẦU ĐĂNG KÝ KHUÔN MẶT")
        print("   Ghi chú: Giữ đúng tư thế trong 2 giây để bắt đầu chụp tự động.")

        for angle_name, instruction, target in REGISTER_ANGLES:
            captured      = 0   
            start_hold_time = None
            wrong_count = 0  # Bộ đếm để khử nhiễu (persistence)

            print(f"\nXin mời {instruction} - cần {target} ảnh")

            while captured < target:
                ret, frame = cap.read()
                if not ret: break

                frame = cv2.flip(frame, 1)   
                h, w = frame.shape[:2]
                current_angle, face_box = detector.get_angle(frame)
                
                # Xác định xem có đang giữ đúng tư thế không
                is_correct = (current_angle == angle_name and face_box is not None)
                
                if is_correct:
                    wrong_count = 0 # Reset bộ đếm nếu đúng
                    if start_hold_time is None:
                        start_hold_time = time.time()
                    
                    elapsed = time.time() - start_hold_time
                    border_color = (0, 255, 0) # Xanh lá khi đúng góc
                    
                    if elapsed >= 1:
                        # Đã giữ đủ 1 giây, bắt đầu chụp tự động cực nhanh
                        x, y, box_w, box_h = face_box
                        margin = 0.25
                        x1 = max(0, int(x - box_w * margin))
                        y1 = max(0, int(y - box_h * margin))
                        x2 = min(w, int(x + box_w * (1 + margin)))
                        y2 = min(h, int(y + box_h * (1 + margin + 0.1)))
                        
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            num_generated = save_clean_faces(face_crop, str(save_dir), angle_name, captured)
                            captured += 1
                            print(f"  Da chup {angle_name}: {captured}/{target} (-> sinh ra {captured * num_generated} anh net)")
                            cv2.waitKey(10) # Chụp liên thanh tốc độ bàn thờ
                    else:
                        remaining = 1 - elapsed
                        cv2.putText(frame, f"Giu nguyen... {remaining:.1f}s", (w//2-100, h-50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    # Nếu sai góc, tăng bộ đếm nhiễu
                    wrong_count += 1
                    # Chỉ reset timer nếu sai liên tiếp quá 5 frame (khoảng 0.2-0.3s)
                    if wrong_count > 5:
                        start_hold_time = None
                    
                    border_color = (0, 0, 255) if current_angle != "NONE" else (0, 0, 0)

                # Vẽ HUD
                cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness=15)
                cv2.putText(frame, f"Yeu cau: {instruction}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Tien do: {captured}/{target}", (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                cv2.imshow("Dang Ky - Main System", frame)
                key = cv2.waitKey(1)
                if key == 27: cancelled = True; break
                if key == 32: break

            summary.append((instruction, captured))
            if cancelled:
                break

        # ── Dọn dẹp ────────────────────────────────────────────
        cap.release()
        cv2.destroyAllWindows()

        # ── Tổng kết ────────────────────────────────────────────
        total = sum(c for _, c in summary)
        print(f"\nĐã đăng ký {total} ảnh cho '{name}'")
        print("\nChi tiết:")
        for label, count in summary:
            if count > 0:
                print(f"   {label}: {count} ảnh")

        if total > 0:
            # Tự động huấn luyện ngay sau khi đăng ký xong
            self.train_embeddings()


    # Huấn luyện Embeddings
    # ════════════════════════════════════════════════════════════

    def train_embeddings(self) -> None:
        """Sinh embedding từ toàn bộ ảnh trong dataset và lưu vào database.

        Quét qua từng thư mục người dùng trong dataset_path, dùng mô hình
        Swin Transformer để tạo vector đặc trưng, rồi lưu vào file pickle.
        """
        print("\n" + "=" * 50)
        print("HUẤN LUYỆN EMBEDDINGS")
        print("=" * 50)
        print("Đang xử lý ảnh và tạo embeddings...")

        self.database = {}      # Xóa dữ liệu cũ trước khi huấn luyện lại
        total_images  = 0

        dataset_root = Path(self.dataset_path)
        if not dataset_root.exists():
            print(f"Thư mục dataset không tồn tại: {dataset_root}")
            return

        for person_dir in dataset_root.iterdir():
            if not person_dir.is_dir():
                continue

            person_name = person_dir.name
            embeddings: list = []
            img_count  = 0

            for img_path in person_dir.iterdir():
                # Chỉ xử lý các định dạng ảnh phổ biến
                if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                try:
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    faces = detect_face(image)
                    if not faces:
                        # Fallback: Nếu không tìm thấy (do ảnh đã crop quá sát), coi toàn bộ ảnh là mặt
                        h_img, w_img = image.shape[:2]
                        faces = [(0, 0, w_img, h_img)]

                    for (x1, y1, x2, y2) in faces:
                        face_crop  = image[y1:y2, x1:x2]
                        embedding  = get_embedding(face_crop)
                        embeddings.append(embedding)
                        img_count += 1

                except Exception as e:
                    print(f"Lỗi xử lý {img_path.name}: {e}")

            if embeddings:
                self.database[person_name] = embeddings
                total_images += img_count
                print(f"{person_name}: {img_count} embeddings")

        self._save_database()
        print(f"\nHuấn luyện hoàn tất! Tổng cộng {total_images} embeddings")

    # ════════════════════════════════════════════════════════════
    # Nhận diện Gương mặt
    # ════════════════════════════════════════════════════════════

    def _find_best_match(self, embedding: np.ndarray) -> tuple[str, float]:
        """Tìm người khớp nhất trong database với chiến lược bảo vệ đa lớp:
          1. Top-3 Average: Tránh bị nhiễu bởi 1 tấm ảnh xấu trong DB.
          2. Absolute Threshold: Ngưỡng tối thiểu (0.70).
          3. Margin Check: Khoảng cách giữa người thứ 1 và thứ 2 phải đủ lớn.
        """
        person_results = []
        for person, emb_list in self.database.items():
            # Tính điểm tương đồng với tất cả ảnh của người này
            all_scores = [compare(embedding, db_emb) for db_emb in emb_list]
            
            # Lấy 3 kết quả tốt nhất (Top-3)
            top_scores = sorted(all_scores, reverse=True)[:3]
            if top_scores:
                max_score = top_scores[0]
                mean_score = sum(top_scores) / len(top_scores)
                # Mẹo tăng vài % điểm: Ưu tiên 70% trọng số cho bức ảnh giống nhất, 30% cho trung bình.
                # Cách này giúp đẩy điểm lên cao hơn so với trung bình cộng thuần túy.
                final_score = 0.7 * max_score + 0.3 * mean_score
            else:
                final_score = 0
                
            person_results.append((final_score, person))

        if not person_results:
            return "Unknown", 0.0

        # Sắp xếp theo điểm giảm dần
        person_results.sort(reverse=True)
        best_score, best_name = person_results[0]

        # 1. Kiểm tra ngưỡng tối thiểu (Yêu cầu khắt khe: phải trên 80% mới nhận diện)
        if best_score < 0.80:
            return "Unknown", best_score

        # 2. Kiểm tra khoảng cách (Margin) với người đứng thứ hai
        if len(person_results) > 1:
            second_score, second_name = person_results[1]
            # Tăng Margin lên 0.15 (15%): Phải giống người thứ nhất hơn người thứ hai ít nhất 15% mới dám khẳng định
            if (best_score - second_score) < 0.15:
                return f"Wait ({best_name}?)", best_score

        return best_name, best_score

    def recognize_faces(self) -> None:
        """Nhận diện gương mặt theo thời gian thực từ webcam.

        Mỗi frame sẽ:
        1. Phát hiện tất cả khuôn mặt.
        2. Tính embedding và so sánh với database.
        3. Hiển thị tên + điểm tương đồng lên frame.
        """
        print("\n" + "=" * 50)
        print("NHẬN DIỆN GƯƠNG MẶT")
        print("=" * 50)

        if not self.database:
            print("Database trống! Vui lòng đăng ký gương mặt trước.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return

        print("\nNhấn 'ESC' để dừng")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lỗi khi đọc frame từ webcam!")
                break

            faces = detect_face(frame)

            for (x1, y1, x2, y2) in faces:
                face_crop = frame[y1:y2, x1:x2]
                try:
                    emb              = get_embedding(face_crop)
                    name, best_score = self._find_best_match(emb)

                    # ── Debug: in điểm tất cả người ra terminal (mỗi 30 frame) ──
                    if not hasattr(self, '_dbg_frame'): self._dbg_frame = 0
                    self._dbg_frame += 1
                    if self._dbg_frame % 30 == 0:
                        scores_info = []
                        # Chúng ta tính lại Top-3 Avg giống như trong _find_best_match để debug chính xác
                        person_avgs = []
                        for person, emb_list in self.database.items():
                            from utils.similarity import compare as _cmp
                            all_s = sorted([_cmp(emb, e) for e in emb_list], reverse=True)
                            top_3 = all_s[:3]
                            avg_s = sum(top_3) / len(top_3) if top_3 else 0.0
                            person_avgs.append((avg_s, person))
                        
                        person_avgs.sort(reverse=True)
                        for s, p in person_avgs:
                            scores_info.append(f"{p}={s:.3f}")
                        
                        delta_str = ""
                        if len(person_avgs) > 1:
                            delta = person_avgs[0][0] - person_avgs[1][0]
                            delta_str = f" | Δ={delta:.4f}"
                        
                        print(f"[DEBUG] {' | '.join(scores_info)}{delta_str}  → {name}")

                    # Màu xanh nếu nhận ra, đỏ nếu không
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # Hiển thị score dù Unknown để dễ calibrate
                    label = f"{name} ({best_score:.2f})"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                except Exception as e:
                    print(f"Lỗi nhận diện: {e}")

            cv2.imshow("Nhận diện Gương mặt", frame)

            if cv2.waitKey(1) == 27:    # ESC – thoát
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Kết thúc nhận diện")

    # ════════════════════════════════════════════════════════════
    # Quản lý Người dùng
    # ════════════════════════════════════════════════════════════

    def view_registered_persons(self) -> None:
        """Hiển thị danh sách tất cả người dùng đã đăng ký kèm thống kê."""
        print("\n" + "=" * 50)
        print("DANH SÁCH NGƯỜI ĐÃ ĐĂNG KÝ")
        print("=" * 50)

        dataset_root = Path(self.dataset_path)
        if not dataset_root.exists():
            print("Thư mục dataset không tồn tại!")
            return

        # Lấy danh sách thư mục con (mỗi thư mục = một người)
        persons = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())

        if not persons:
            print("Chưa có ai đăng ký!")
            return

        print(f"\nTổng cộng: {len(persons)} người\n")
        for i, person in enumerate(persons, start=1):
            person_dir  = dataset_root / person
            # Đếm số file ảnh
            img_count   = sum(1 for f in person_dir.iterdir()
                              if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            db_count    = len(self.database.get(person, []))

            print(f"  {i}. {person}")
            print(f"     - Ảnh       : {img_count}")
            print(f"     - Embeddings: {db_count}")

    def delete_person(self) -> None:
        """Xóa toàn bộ dữ liệu (ảnh + embedding) của một người dùng."""
        print("\n" + "=" * 50)
        print("XÓA NGƯỜI DÙNG")
        print("=" * 50)

        dataset_root = Path(self.dataset_path)
        persons = [p.name for p in dataset_root.iterdir() if p.is_dir()]

        if not persons:
            print("Chưa có ai đăng ký!")
            return

        self.view_registered_persons()
        name = input("\nNhập tên người dùng cần xóa: ").strip()

        person_dir = dataset_root / name
        if person_dir.exists():
            shutil.rmtree(person_dir)           # Xóa toàn bộ thư mục ảnh

            if name in self.database:
                del self.database[name]          # Xóa embedding khỏi database
                self._save_database()

            print(f"Đã xóa {name}")
        else:
            print(f"Không tìm thấy '{name}'")

    def augment_face(self) -> None:
        """Bổ sung thêm 10 ảnh/người cho người đã đăng ký để giảm nhầm lẫn.

        Chỉ ghi thêm ảnh mới vào thư mục hiện tại (không xóa ảnh cũ).
        Tên file tiếp nối theo số thứ tự đã có để tránh ghi đè.
        Sau khi chụp xong, tự động huấn luyện lại embeddings.
        """
        print("\n" + "=" * 50)
        print("BỔ SUNG ẢNH CHO NGƯỜI ĐÃ ĐĂNG KÝ")
        print("=" * 50)

        dataset_root = Path(self.dataset_path)
        persons = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())

        if not persons:
            print("Chưa có ai đăng ký!")
            return

        print("Danh sách người đã đăng ký:")
        for i, p in enumerate(persons, 1):
            print(f"  {i}. {p}")

        name = input("\nNhập tên người cần bổ sung ảnh: ").strip()
        if name not in persons:
            print(f"Không tìm thấy '{name}' trong hệ thống!")
            return

        save_dir = dataset_root / name

        # Đếm số thứ tự ảnh hiện có theo từng góc để tiếp nối (không ghi đè)
        existing_counts: dict[str, int] = {}
        for angle_name, _, _ in AUGMENT_ANGLES:
            existing_counts[angle_name] = sum(
                1 for f in save_dir.glob(f"{angle_name}_*.jpg")
            )

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return

        detector = HeadPoseDetector()
        summary   = []
        cancelled = False

        total_target = sum(t for _, _, t in AUGMENT_ANGLES)
        print(f"\nBẮT ĐẦU BỔ SUNG ẢNH cho '{name}' — mục tiêu: {total_target} ảnh")
        print("   Hệ thống sẽ chỉ chụp khi bạn quay đúng góc yêu cầu!")
        print("   ESC: dừng  |  SPACE: bỏ qua góc hiện tại")

        for angle_name, instruction, target in AUGMENT_ANGLES:
            captured      = 0
            correct_angle_frames = 0
            start_idx     = existing_counts.get(angle_name, 0)  # Tiếp nối từ index hiện có

            print(f"\nXin mời {instruction} – cần thêm {target} ảnh")

            while captured < target:
                ret, frame = cap.read()
                if not ret:
                    print("Lỗi đọc frame từ webcam!")
                    cancelled = True
                    break

                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                current_angle, face_box = detector.get_angle(frame)

                border_color = (0, 0, 0)
                if current_angle != "NONE":
                    border_color = (0, 200, 255)

                if current_angle == angle_name and face_box is not None:
                    border_color = (0, 200, 0)
                    correct_angle_frames += 1
                    if correct_angle_frames % self.CAPTURE_INTERVAL == 0:
                        x, y, box_w, box_h = face_box
                        margin_x = int(box_w * 0.25)
                        margin_y = int(box_h * 0.25)
                        crop_x1 = max(0, x - margin_x)
                        crop_y1 = max(0, y - margin_y)
                        crop_x2 = min(w, x + box_w + margin_x)
                        crop_y2 = min(h, y + box_h + int(margin_y * 1.5))
                        face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                        if face_crop.size > 0:
                            img_path = save_dir / f"{angle_name}_{start_idx + captured}.jpg"
                            cv2.imwrite(str(img_path), face_crop)
                            captured += 1
                            print(f"  Đã chụp bổ sung góc {angle_name}: {captured}/{target}")
                else:
                    correct_angle_frames = 0

                # Vẽ HUD
                cv2.rectangle(frame, (0, 0), (w, h), border_color, thickness=20)
                # Tiêu đề
                cv2.putText(frame, f"BO SUNG ANH - {name}", (30, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                text_info = f"Goc can thu: {angle_name} ({captured}/{target})"
                cv2.putText(frame, text_info, (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Hien tai he thong thay: {current_angle}", (30, 105),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

                cv2.imshow("Bo Sung Anh - Khung Viec", frame)
                key = cv2.waitKey(1)

                if key == 27:   # ESC
                    cancelled = True
                    break
                if key == 32:   # SPACE – bỏ qua góc này
                    break

            summary.append((instruction, captured))
            if cancelled:
                break

        cap.release()
        cv2.destroyAllWindows()

        total = sum(c for _, c in summary)
        print(f"\nĐã bổ sung {total} ảnh cho '{name}'")
        for label, count in summary:
            if count > 0:
                print(f"   {label}: {count} ảnh")

        if total > 0:
            print("\nTự động huấn luyện lại embeddings...")
            self.train_embeddings()

    def attendance(self) -> None:
        """Điểm danh: Quét khuôn mặt, chụp 1 ảnh và trả ra danh tính."""
        print("\n" + "=" * 50)
        print("CHỨC NĂNG ĐIỂM DANH")
        print("=" * 50)

        if not self.database:
            print("Database trống! Vui lòng đăng ký gương mặt trước.")
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Không thể mở webcam!")
            return

        print("\nĐang mở camera để điểm danh. Nhấn 'ESC' để hủy.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Lỗi khi đọc frame từ webcam!")
                break

            display_frame = frame.copy()
            cv2.putText(display_frame, "Dang quet diem danh...", (30, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.imshow("Diem Danh", display_frame)

            key = cv2.waitKey(1)
            if key == 27: # ESC
                print("Đã hủy điểm danh.")
                break
                
            # Xử lý nhận diện ngầm
            faces = detect_face(frame)
            if not faces:
                continue
                
            # Lấy khuôn mặt to nhất (gần cam nhất)
            faces = sorted(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
            x1, y1, x2, y2 = faces[0]
            face_crop = frame[y1:y2, x1:x2]
            
            try:
                emb = get_embedding(face_crop)
                name, score = self._find_best_match(emb)
                
                # Chấp nhận nếu nhận diện thành công (không phải Unknown hay Wait)
                if name != "Unknown" and not name.startswith("Wait"):
                    print(f"\n=> [THÀNH CÔNG] Điểm danh xác nhận: {name} (Độ tự tin: {score*100:.1f}%)")
                    
                    # Hiện thông báo lên màn hình trong 3 giây
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Diem Danh OK: {name}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.imshow("Diem Danh", display_frame)
                    cv2.waitKey(2000) # Dừng 2 giây để user nhìn thấy kết quả
                    break
            except Exception as e:
                pass

        cap.release()
        cv2.destroyAllWindows()

    # ════════════════════════════════════════════════════════════
    # Giao diện Menu
    # ════════════════════════════════════════════════════════════

    def _display_menu(self) -> None:
        """In menu chính ra terminal."""
        print("\n" + "=" * 50)
        print("HỆ THỐNG NHẬN DIỆN GƯƠNG MẶT")
        print("=" * 50)
        print("1. Đăng ký gương mặt mới")
        print("2. Huấn luyện embeddings")
        print("3. Nhận diện gương mặt")
        print("4. Xem danh sách người dùng")
        print("5. Xóa người dùng")
        print("6. Bổ sung ảnh (giảm nhầm lẫn giữa các người)")
        print("7. Điểm danh (Quét 1 ảnh & Báo danh tính)")
        print("0. Thoát chương trình")
        print("=" * 50)

    def run(self) -> None:
        """Vòng lặp chính của chương trình – xử lý lựa chọn menu."""
        # Bảng ánh xạ lựa chọn → phương thức tương ứng
        actions = {
            "1": self.register_face,
            "2": self.train_embeddings,
            "3": self.recognize_faces,
            "4": self.view_registered_persons,
            "5": self.delete_person,
            "6": self.augment_face,
            "7": self.attendance,
        }

        while True:
            self._display_menu()
            choice = input("Chọn chức năng (0-7): ").strip()

            if choice == "0":
                print("\nCảm ơn bạn đã sử dụng! Tạm biệt!")
                break

            action = actions.get(choice)
            if action:
                action()
            else:
                print("Lựa chọn không hợp lệ! Vui lòng thử lại.")


# ── Điểm khởi chạy chương trình ─────────────────────────────────
if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()