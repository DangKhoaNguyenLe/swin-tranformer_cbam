import torch
import torch.nn as nn
import torch.optim as optim
import random

# 1. Tạo dữ liệu giả lập (Synthetic Data)
# Đại diện cho lịch sử các lần commit code.
# Cấu trúc: [lines_of_code, complexity, dev_experience, has_unit_tests]
# Dữ liệu chuẩn hóa trong khoảng 0-1 để mạng nơ-ron dễ học
data = []
labels = []

# Tự động tạo ra 2000 mẫu dữ liệu lịch sử
for _ in range(2000):
    loc = random.uniform(0, 1)       # Dòng code bị đổi (Càng dài càng dễ lỗi)
    comp = random.uniform(0, 1)      # Độ phức tạp thuật toán (Cao -> dễ lỗi)
    exp = random.uniform(0, 1)       # Kinh nghiệm Developer (Cao -> ít lỗi)
    tests = random.uniform(0, 1)     # Tỷ lệ coverage của Unit Test (Cao -> ít lỗi)
    
    # Logic ngầm để đánh giá:
    risk_score = loc * 0.4 + comp * 0.4 - exp * 0.2 - tests * 0.3
    
    # Gắn nhãn 1 (Có Bug) và 0 (Không Bug)
    is_buggy = 1.0 if risk_score > 0.1 else 0.0
    
    data.append([loc, comp, exp, tests])
    labels.append([is_buggy])

X = torch.tensor(data, dtype=torch.float32)
y = torch.tensor(labels, dtype=torch.float32)

# 2. Xây dựng mô hình Deep Learning (Multi-Layer Perceptron)
class BugPredictor(nn.Module):
    def __init__(self):
        super(BugPredictor, self).__init__()
        # Lớp đầu vào: Nhận 4 tham số trên
        self.layer1 = nn.Linear(4, 16) 
        self.relu = nn.ReLU()          # Hàm kích hoạt (giúp học logic phức tạp)
        
        # Lớp ẩn: Từ 16 node rút gọn xuống 8 node
        self.layer2 = nn.Linear(16, 8) 
        
        # Lớp đầu ra: Chỉ 1 node, xuất ra xác suất (0 đến 1)
        self.output = nn.Linear(8, 1)  
        self.sigmoid = nn.Sigmoid()    # Ép giá trị đầu ra nằm chuẩn trong 0 - 1

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.output(x))
        return x

model = BugPredictor()

# 3. Cấu hình quy tắc học
criterion = nn.BCELoss() # Binary Cross Entropy (chuẩn cho bài toán phân loại 0/1)
optimizer = optim.Adam(model.parameters(), lr=0.01) # Tốc độ học (Learning Rate)

# 4. Bắt đầu huấn luyện mô hình (Training)
print("="*60)
print(" BẮT ĐẦU HUẤN LUYỆN AI: MÔ HÌNH DỰ ĐOÁN LỖI (BUG PREDICTOR)")
print("="*60)

epochs = 500
for epoch in range(epochs):
    optimizer.zero_grad()      # Xóa rác tính toán của vòng lặp trước
    predictions = model(X)     # B1: Thử dự đoán toàn bộ dữ liệu
    loss = criterion(predictions, y) # B2: So sánh dự đoán với kết quả thực tế (tính sai số)
    loss.backward()            # B3: Tìm cách sửa lỗi (tính đạo hàm)
    optimizer.step()           # B4: Cập nhật sự khôn ngoan cho mô hình
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Sai số (Loss): {loss.item():.4f}")

# 5. Demo sử dụng vào thực tế
print("\n" + "="*60)
print(" DEMO ÁP DỤNG TRONG THỰC TẾ (TESTING PIPELINE)")
print("="*60)

# Giả sử có 3 lập trình viên vừa tạo Pull Request (PR) mới
# Số liệu mô phỏng theo chuẩn 0 - 1

# PR 1: Sửa rất nhiều code (0.9), logic lằng nhằng (0.8), Dev mới ra trường (0.1), Lười viết test (0.0)
pr_1 = torch.tensor([[0.9, 0.8, 0.1, 0.0]], dtype=torch.float32)

# PR 2: Sửa ít code (0.2), logic cơ bản (0.2), Dev trung bình (0.5), Có viết Unit Test đàng hoàng (0.8)
pr_2 = torch.tensor([[0.2, 0.2, 0.5, 0.8]], dtype=torch.float32)

# PR 3: Dev Senior (0.9), nhưng viết đoạn code cực dài (0.9) và phức tạp (0.9), Không viết Test (0.1)
pr_3 = torch.tensor([[0.9, 0.9, 0.9, 0.1]], dtype=torch.float32)

with torch.no_grad(): # Tắt chế độ học, chuyển sang chế độ Test thực tế
    pred_1 = model(pr_1).item() * 100
    pred_2 = model(pr_2).item() * 100
    pred_3 = model(pr_3).item() * 100

print(f"[Pull Request 1] Dài, phức tạp, Dev mới, KO Test")
print(f" => AI Cảnh báo: Tỷ lệ sinh ra Bug là {pred_1:.2f}%\n")

print(f"[Pull Request 2] Ngắn, đơn giản, Dev mid-level, CÓ Test")
print(f" => AI Cảnh báo: Tỷ lệ sinh ra Bug là {pred_2:.2f}%\n")

print(f"[Pull Request 3] Code của Dev Senior nhưng dài, phức tạp, KO Test")
print(f" => AI Cảnh báo: Tỷ lệ sinh ra Bug là {pred_3:.2f}%\n")

print("Kết luận: Hệ thống CI/CD có thể tự động chặn Pull Request 1 và 3, yêu cầu viết thêm Test!")
