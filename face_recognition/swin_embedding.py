import timm
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import sys

# Đảm bảo có thể import SwinWithCBAM từ thư mục gốc
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from importlib.util import spec_from_file_location, module_from_spec
swin_cbam_path = os.path.join(PROJECT_ROOT, "Sw-cbam", "swin_cbam.py")
spec = spec_from_file_location("swin_cbam", swin_cbam_path)
swin_cbam_module = module_from_spec(spec)
spec.loader.exec_module(swin_cbam_module)
SwinWithCBAM = swin_cbam_module.SwinWithCBAM

# Định nghĩa cấu trúc trích xuất đặc trưng giống lúc train
class CustomFaceExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = SwinWithCBAM()
        # Dropout (chỉ dùng lúc train, lúc eval sẽ bị vô hiệu hóa nhưng vẫn cần định nghĩa để load weights)
        self.dropout = nn.Dropout(p=0.5)
        # Bottleneck 512 chiều (Đặc trưng cốt lõi cho Face Recognition)
        self.bottleneck = nn.Sequential(
            nn.Linear(768, 512),
            nn.BatchNorm1d(512),
            nn.PReLU()
        )
        
    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.dropout(features)
        return self.bottleneck(features)

LFW_WEIGHTS = "database/lfw_swin.pth"
CUSTOM_WEIGHTS = "database/custom_swin.pth"

# Khởi tạo mô hình
is_custom = False
if os.path.exists(CUSTOM_WEIGHTS):
    print("Phát hiện file custom_swin.pth. Đang nạp cấu trúc mô hình tuỳ chỉnh (SwinWithCBAM + Bottleneck)...")
    model = CustomFaceExtractor()
    try:
        checkpoint = torch.load(CUSTOM_WEIGHTS, map_location="cpu")
        # File .pth từ train_custom.py thường lưu dưới dạng dictionary
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Bỏ đi phần head (vì ta chỉ lấy embedding)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        
        # Xử lý trường hợp module. prefix từ DataParallel
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict, strict=False)
        print("Đã nạp trọng số custom (Fine-tuned local) thành công! - Embedding Size: 512")
        is_custom = True
    except Exception as e:
        print(f"Lỗi khi nạp trọng số tùy chỉnh: {e}. Hệ thống sẽ tiếp tục dùng weights mặc định.")
        is_custom = False

if not is_custom:
    print("Sử dụng cấu trúc Swin mặc định (768-dim).")
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
    if os.path.exists(LFW_WEIGHTS):
        try:
            model.load_state_dict(torch.load(LFW_WEIGHTS, map_location="cpu"), strict=False)
            print("Đã nạp trọng số LFW Pre-trained thành công! - Embedding Size: 768")
        except Exception as e:
            print(f"Lỗi khi nạp LFW weights: {e}")

model.eval()

def get_embedding(face):
    # 1. Chuyển không gian màu BGR sang RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # 2. Resize về 224x224 cho Swin Transformer (dùng INTER_LINEAR để giữ chi tiết)
    face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)

    # 3. Chuẩn hóa pixel về [0, 1] và ImageNet mean/std
    face = face / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    face = (face - mean) / std

    # 4. Chuyển đổi chiều (H, W, C) -> (C, H, W)
    face = np.transpose(face, (2, 0, 1))

    # 5. Chuyển thành Tensor và thêm batch dimension
    face = torch.tensor(face).float().unsqueeze(0)

    # 6. Trích xuất embedding (không tính gradient)
    with torch.no_grad():
        embedding = model(face).numpy()

    # 7. L2 Normalization (Quan trọng để so sánh Cosine Similarity chính xác)
    norm = np.linalg.norm(embedding, ord=2, axis=1, keepdims=True)
    embedding = embedding / (norm + 1e-6)

    return embedding