import torch
import torch.nn as nn
import timm
import sys
import os

# Thêm đường dẫn gốc vào sys.path để có thể import từ module Attention
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Attention.cbam import CBAM

class SwinWithCBAM(nn.Module):
    def __init__(self, model_name="swin_tiny_patch4_window7_224"):
        super(SwinWithCBAM, self).__init__()
        # Khởi tạo backbone (bỏ lớp phân loại cuối và giữ lại feature map)
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool='')
        
        # Tích hợp CBAM nối tiếp sau MỖI Stage của Swin Transformer
        # Kích thước channel đầu ra của 4 stage trong Swin-Tiny: 96, 192, 384, 768
        # Chỉ dùng Spatial Attention ở Stage 0 và 1. Tắt ở Stage 2 và 3 để bảo vệ đặc trưng không gian.
        self.cbam_stage0 = CBAM(in_planes=96, use_spatial=True)
        self.cbam_stage1 = CBAM(in_planes=192, use_spatial=True)
        self.cbam_stage2 = CBAM(in_planes=384, use_spatial=False)
        self.cbam_stage3 = CBAM(in_planes=768, use_spatial=False)
        
        # Khởi tạo tham số alpha (learnable scaling) bằng 0 để bảo vệ pretrained weights
        self.alpha0 = nn.Parameter(torch.zeros(1))
        self.alpha1 = nn.Parameter(torch.zeros(1))
        self.alpha2 = nn.Parameter(torch.zeros(1))
        self.alpha3 = nn.Parameter(torch.zeros(1))
        
        # Lớp Global Average Pooling để đưa về vector 1D
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def _apply_cbam(self, x, cbam_module, alpha):
        # Swin trả về feature map định dạng (B, H, W, C)
        # CBAM yêu cầu định dạng (B, C, H, W)
        x_permuted = x.permute(0, 3, 1, 2)
        identity = x_permuted
        out = cbam_module(x_permuted)
        
        # Zero Initialization: Nhân out với alpha (khởi tạo = 0)
        # Ở epoch 1, out = identity (giống hệt Swin thuần)
        out = identity + alpha * out 
        
        # Trả lại định dạng (B, H, W, C) để Swin tiếp tục xử lý ở stage sau
        return out.permute(0, 2, 3, 1)

    def forward(self, x):
        # 1. Patch Embedding
        x = self.backbone.patch_embed(x)
        if hasattr(self.backbone, 'absolute_pos_embed') and getattr(self.backbone, 'absolute_pos_embed') is not None:
            x = x + self.backbone.absolute_pos_embed
            
        # Xử lý an toàn cho nhiều phiên bản timm khác nhau trên Kaggle
        if hasattr(self.backbone, 'pos_drop'):
            x = self.backbone.pos_drop(x)
        elif hasattr(self.backbone, 'drop'):
            x = self.backbone.drop(x)
        # 2. Xử lý qua từng Stage và áp dụng CBAM nối tiếp
        
        # Stage 0
        x = self.backbone.layers[0](x)
        x = self._apply_cbam(x, self.cbam_stage0, self.alpha0)
        
        # Stage 1
        x = self.backbone.layers[1](x)
        x = self._apply_cbam(x, self.cbam_stage1, self.alpha1)
        
        # Stage 2
        x = self.backbone.layers[2](x)
        x = self._apply_cbam(x, self.cbam_stage2, self.alpha2)
        
        # Stage 3
        x = self.backbone.layers[3](x)
        # Sửa lỗi vị trí: Đưa CBAM lên trước lớp LayerNorm cuối cùng
        x = self._apply_cbam(x, self.cbam_stage3, self.alpha3)
        x = self.backbone.norm(x) # Lớp norm chuẩn của Swin sau các layer
        
        # 3. Chuyển đổi định dạng cuối cùng cho avgpool: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # 4. Pooling về vector đặc trưng: (B, 768, 1, 1)
        x = self.avgpool(x)
        
        # 5. Flatten: (B, 768)
        x = torch.flatten(x, 1)
        return x
