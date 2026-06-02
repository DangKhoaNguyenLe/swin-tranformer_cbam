import os
import sys
import time
import json
import cv2
import pickle
import threading
from pathlib import Path
from flask import Flask, render_template, Response, request, jsonify

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from face_detection.detect_face import detect_face
from face_recognition.swin_embedding import get_embedding
from utils.similarity import compare
from main import HeadPoseDetector, save_clean_faces, FaceRecognitionSystem

WEB_REGISTER_ANGLES_NORMAL = [
    ("CENTERED",   "Nhin THANG (Cu dong co mat nhe)", 5),
    ("LEFT",       "Quay TRAI (Gat dau nhe)", 5),
    ("RIGHT",      "Quay PHAI (Gat dau nhe)", 5)
]

WEB_REGISTER_ANGLES_MASKED = [
    ("CENTERED",   "DEO KHAU TRANG: Nhin THANG", 5),
    ("LEFT",       "DEO KHAU TRANG: Quay TRAI", 5),
    ("RIGHT",      "DEO KHAU TRANG: Quay PHAI", 5)
]

app = Flask(__name__)

class AppState:
    def __init__(self):
        self.mode = "IDLE" # IDLE, REGISTER, ATTENDANCE
        self.camera_active = False
        self.cap = None
        self.lock = threading.Lock()
        
        # Async Frame Buffers
        self.raw_frame = None
        self.current_faces = []
        self.ai_results = {}
        
        # Registration state
        self.reg_name = ""
        self.reg_type = "normal"
        self.reg_angle_idx = 0
        self.reg_captured = 0
        self.reg_start_hold_time = None
        self.reg_wrong_count = 0
        self.reg_summary = []
        self.reg_done = False
        self.reg_last_capture_time = 0
        
        # Attendance state
        self.attendance_history = []
        
        # System
        self.detector = HeadPoseDetector()
        self.face_system = FaceRecognitionSystem()
        if not self.face_system.database:
            print("Warning: Database is empty!")
            
        # Start background threads
        threading.Thread(target=self.camera_loop, daemon=True).start()
        threading.Thread(target=self.tracker_loop, daemon=True).start()
        threading.Thread(target=self.ai_loop, daemon=True).start()

    def camera_loop(self):
        """Liên tục đọc camera ở 30 FPS để luồng video luôn mượt (Không bị block bởi AI)"""
        while True:
            if not self.camera_active:
                if self.cap is not None:
                    self.cap.release()
                    self.cap = None
                time.sleep(0.1)
                continue
                
            if self.cap is None or not self.cap.isOpened():
                self.cap = cv2.VideoCapture(0)
                
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                with self.lock:
                    self.raw_frame = frame
            else:
                time.sleep(0.01)

    def tracker_loop(self):
        """Chạy detect_face (YOLO) cực nhanh ở luồng riêng để cập nhật tọa độ box mượt mà"""
        while True:
            if not self.camera_active or self.raw_frame is None:
                time.sleep(0.05)
                continue
                
            # Chỉ detect_face trong ATTENDANCE
            if self.mode == "ATTENDANCE":
                frame = self.raw_frame.copy()
                c_faces = detect_face(frame)
                with self.lock:
                    self.current_faces = c_faces
            else:
                time.sleep(0.1)
                continue
                
            time.sleep(0.01)

    def ai_loop(self):
        """Xử lý AI ngầm định kỳ trên frame mới nhất (nhận diện, góc xoay)"""
        while True:
            if not self.camera_active or self.raw_frame is None or self.mode == "IDLE":
                time.sleep(0.1)
                continue
                
            # Lấy bản sao frame mới nhất để AI xử lý
            frame = self.raw_frame.copy()
            
            with self.lock:
                mode = self.mode
                reg_done = self.reg_done
                
            if mode == "REGISTER" and not reg_done:
                self._ai_register_step(frame)
            elif mode == "ATTENDANCE":
                self._ai_attendance_step(frame)
                
            # Tránh 100% CPU usage
            time.sleep(0.02)
            
    def _ai_register_step(self, frame):
        with self.lock:
            angles_list = WEB_REGISTER_ANGLES_MASKED if self.reg_type == "masked" else WEB_REGISTER_ANGLES_NORMAL
            if self.reg_angle_idx >= len(angles_list):
                if not self.reg_done:
                    self.reg_done = True
                    threading.Thread(target=self.face_system.train_embeddings).start()
                return
                
            angle_name, instruction, target = angles_list[self.reg_angle_idx]
            
        h, w = frame.shape[:2]
        
        # AI Detect
        if self.reg_type == "masked": 
            # Khi đeo khẩu trang, HaarCascade profile thất bại. 
            # Dùng YOLO lấy box mặt, sau đó dùng HaarCascade mắt ở nửa trên mặt để đoán góc xoay.
            faces = detect_face(frame)
            if len(faces) > 0:
                faces = sorted(faces, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
                x1, y1, x2, y2 = faces[0]
                w_box = x2 - x1
                h_box = y2 - y1
                face_box = (x1, y1, w_box, h_box)
                
                # Cắt nửa trên khuôn mặt (tránh phần khẩu trang) để tìm mắt
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_gray = gray[max(0, y1):max(0, y1) + int(h_box*0.6), max(0, x1):max(0, x2)]
                
                eyes = self.detector.eye_cascade.detectMultiScale(roi_gray, 1.1, 3, minSize=(15, 15))
                
                if len(eyes) > 0:
                    # Tính trung bình vị trí X của các mắt so với chiều rộng khuôn mặt
                    avg_eye_x = sum([ex + ew/2.0 for (ex, ey, ew, eh) in eyes]) / len(eyes)
                    eye_cx_norm = avg_eye_x / w_box
                    
                    # Ngưỡng nhạy hơn (0.45 và 0.55)
                    if eye_cx_norm < 0.45:
                        current_angle = "LEFT"
                    elif eye_cx_norm > 0.55:
                        current_angle = "RIGHT"
                    else:
                        current_angle = "CENTERED"
                else:
                    # Nếu không thấy mắt, tạm thời chấp nhận luôn góc yêu cầu để không block user
                    current_angle = angle_name
            else:
                face_box = None
                current_angle = "NONE"
        else:
            current_angle, face_box = self.detector.get_angle(frame)
        
        with self.lock:
            is_correct = (current_angle == angle_name and face_box is not None)
            
            if is_correct:
                self.reg_wrong_count = 0
                if self.reg_start_hold_time is None:
                    self.reg_start_hold_time = time.time()
                    
                elapsed = time.time() - self.reg_start_hold_time
                
                if elapsed >= 1:
                    x, y, box_w, box_h = face_box
                    margin = 0.25
                    
                    if time.time() - self.reg_last_capture_time > 0.15:
                        x1 = max(0, int(x - box_w * margin))
                        y1 = max(0, int(y - box_h * margin))
                        x2 = min(w, int(x + box_w * (1 + margin)))
                        y2 = min(h, int(y + box_h * (1 + margin + 0.1)))
                        
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size > 0:
                            save_dir = Path(self.face_system.dataset_path) / self.reg_name
                            save_dir.mkdir(parents=True, exist_ok=True)
                            save_name = f"MASKED_{angle_name}" if self.reg_type == "masked" else angle_name
                            save_clean_faces(face_crop, str(save_dir), save_name, self.reg_captured)
                            self.reg_captured += 1
                            self.reg_last_capture_time = time.time()
                        
                    if self.reg_captured >= target:
                        self.reg_summary.append((instruction, self.reg_captured))
                        self.reg_angle_idx += 1
                        self.reg_captured = 0
                        self.reg_start_hold_time = None
            else:
                self.reg_wrong_count += 1
                if self.reg_wrong_count > 5:
                    self.reg_start_hold_time = None
                    
            # Ghi kết quả AI cho MJPEG vẽ
            self.ai_results = {
                "type": "register",
                "face_box": face_box,
                "current_angle": current_angle,
                "is_correct": is_correct,
                "elapsed": time.time() - self.reg_start_hold_time if self.reg_start_hold_time else 0,
                "instruction": instruction,
                "progress": f"{self.reg_captured}/{target}"
            }

    def _ai_attendance_step(self, frame):
        with self.lock:
            faces = self.current_faces.copy()
            
        results = []
        for (x1, y1, x2, y2) in faces:
            h, w = frame.shape[:2]
            y1_c, y2_c = max(0, y1), min(h, y2)
            x1_c, x2_c = max(0, x1), min(w, x2)
            face_crop = frame[y1_c:y2_c, x1_c:x2_c]
            
            if face_crop.size == 0:
                continue
                
            try:
                emb = get_embedding(face_crop)
                name, score = self.face_system._find_best_match(emb)
                
                color = (0, 255, 0) if name != "Unknown" and not name.startswith("Wait") else (0, 0, 255)
                if name.startswith("Wait"):
                    color = (0, 255, 255)
                    
                results.append((x1, y1, x2, y2, name, score, color))
                
                if name != "Unknown" and not name.startswith("Wait"):
                    with self.lock:
                        if not self.attendance_history or self.attendance_history[-1]['name'] != name or (time.time() - self.attendance_history[-1]['time'] > 5):
                            self.attendance_history.append({
                                'name': name,
                                'time': time.time(),
                                'score': score
                            })
            except Exception as e:
                pass
                
        with self.lock:
            self.ai_results = {
                "type": "attendance",
                "faces": results
            }

state = AppState()

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def get_frame():
    # MJPEG stream generator - Vẽ UI ở 30 FPS mượt mà
    while True:
        if not state.camera_active or state.raw_frame is None:
            time.sleep(0.1)
            continue
            
        frame = state.raw_frame.copy()
        display_frame = frame
        
        with state.lock:
            mode = state.mode
            results = state.ai_results.copy()
            reg_done = state.reg_done
            current_faces = state.current_faces.copy() if hasattr(state, 'current_faces') else []
            
        if mode == "REGISTER" and not reg_done:
            h, w = display_frame.shape[:2]
            
            if results and results.get("type") == "register":
                face_box = results.get("face_box")
                is_correct = results.get("is_correct")
                elapsed = results.get("elapsed", 0)
                
                border_color = (0, 0, 0) if results.get("current_angle") == "NONE" else (0, 0, 255)
                if is_correct: border_color = (0, 255, 0)
                
                cv2.rectangle(display_frame, (0, 0), (w, h), border_color, thickness=15)
                
                if face_box is not None:
                    fx, fy, fw, fh = face_box
                    cv2.rectangle(display_frame, (fx, fy), (fx+fw, fy+fh), border_color, 2)
                    
                if is_correct and elapsed > 0 and elapsed < 1:
                    remaining = 1 - elapsed
                    cv2.putText(display_frame, f"Giu nguyen... {remaining:.1f}s", (w//2-100, h-50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                                
                cv2.putText(display_frame, f"Yeu cau: {results.get('instruction')}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Tien do: {results.get('progress')}", (30, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                            
        elif mode == "ATTENDANCE":
            if results and results.get("type") == "attendance":
                ai_faces = results.get("faces", [])
                for (cx1, cy1, cx2, cy2) in current_faces:
                    best_iou = 0
                    best_info = ("Scanning...", 0.0, (0, 255, 255))
                    
                    for (ax1, ay1, ax2, ay2, name, score, color) in ai_faces:
                        iou = compute_iou((cx1, cy1, cx2, cy2), (ax1, ay1, ax2, ay2))
                        if iou > best_iou:
                            best_iou = iou
                            best_info = (name, score, color)
                            
                    name, score, color = best_info if best_iou > 0.15 else ("Scanning...", 0.0, (0, 255, 255))
                    
                    cv2.rectangle(display_frame, (cx1, cy1), (cx2, cy2), color, 2)
                    label = f"{name} ({score:.2f})" if name != "Scanning..." else name
                    cv2.putText(display_frame, label, (cx1, cy1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                                
        # Encode and Yield
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        else:
            time.sleep(0.03)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/state')
def get_state():
    info = {
        "mode": state.mode,
        "camera_active": state.camera_active
    }
    if state.mode == "REGISTER":
        info["reg_done"] = state.reg_done
        info["reg_name"] = state.reg_name
        angles_list = WEB_REGISTER_ANGLES_MASKED if state.reg_type == "masked" else WEB_REGISTER_ANGLES_NORMAL
        if state.reg_angle_idx < len(angles_list):
            info["instruction"] = angles_list[state.reg_angle_idx][1]
            info["target"] = angles_list[state.reg_angle_idx][2]
            info["captured"] = state.reg_captured
        else:
            info["instruction"] = "Hoàn tất. Đang huấn luyện hệ thống..."
            info["target"] = 0
            info["captured"] = 0
    elif state.mode == "ATTENDANCE":
        recent = [{"name": x["name"], "time": time.strftime("%H:%M:%S", time.localtime(x["time"]))} 
                  for x in state.attendance_history[-5:]]
        recent.reverse()
        info["history"] = recent
        
    return jsonify(info)

@app.route('/api/start_register', methods=['POST'])
def start_register():
    data = request.json
    name = data.get('name', '').strip()
    reg_type = data.get('type', 'normal')
    if not name:
        return jsonify({"error": "Tên không hợp lệ"}), 400
        
    with state.lock:
        state.mode = "REGISTER"
        state.camera_active = True
        state.reg_name = name
        state.reg_type = reg_type
        state.reg_angle_idx = 0
        state.reg_captured = 0
        state.reg_start_hold_time = None
        state.reg_wrong_count = 0
        state.reg_summary = []
        state.reg_done = False
        state.reg_last_capture_time = 0
        
    return jsonify({"status": "ok"})

@app.route('/api/start_attendance', methods=['POST'])
def start_attendance():
    state.face_system.database = state.face_system._load_database()
    
    with state.lock:
        state.mode = "ATTENDANCE"
        state.camera_active = True
        
    return jsonify({"status": "ok"})

@app.route('/api/stop', methods=['POST'])
def stop():
    with state.lock:
        state.mode = "IDLE"
        state.camera_active = False
        
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
