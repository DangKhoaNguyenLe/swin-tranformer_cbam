import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import importlib.util
from collections import Counter

# --- Cấu hình huấn luyện (Tối ưu cho RMFRD) ---
MODEL_SAVE_PATH = "database/baseline_swin.pth"
METRICS_SAVE_PATH = "baseline_training_metrics.png"
BATCH_SIZE = 32
ACCUMULATION_STEPS = 1 
NUM_EPOCHS = 50
LEARNING_RATE = 5e-4 
BACKBONE_LR = 5e-5   

PATIENCE = 10        

def main(dataset_path):
    print("=" * 60)
    print(" BẮT ĐẦU HUẤN LUYỆN SWIN THUẦN (BASELINE) VỚI RMFRD")
    print("=" * 60)
    
    if not os.path.exists(dataset_path):
        print(f" LỖI: Không tìm thấy thư mục dataset tại {dataset_path}")
        return

    # 1. Thiết lập Data Augmentation (Cân bằng giữa chống Overfitting và cho phép mô hình học)
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.RandomRotation(15), # Tăng xoay nhẹ lên 15 độ
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)), # Mô phỏng camera xê dịch hoặc xa gần
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)), # Che khuất ngẫu nhiên (ép mô hình học các phần khác của mặt)
    ])

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Đang tải dữ liệu từ {dataset_path}...")
    dataset_train = ImageFolder(root=dataset_path, transform=transform_train)
    dataset_val = ImageFolder(root=dataset_path, transform=transform_val)
    num_classes = len(dataset_train.classes)

    print(f"Đã tìm thấy {len(dataset_train)} ảnh thuộc {num_classes} người.")

    # Chia dữ liệu 70% Train, 15% Val, 15% Test (Chia riêng cho từng nhóm Mask và Normal)
    targets = dataset_train.targets
    samples = dataset_train.samples # list of (path, label)
    
    train_indices = []
    val_indices = []
    test_indices = []
    
    def split_group(indices, ratio_train=0.70, ratio_val=0.15):
        np.random.shuffle(indices)
        n_total = len(indices)
        if n_total == 0:
            return [], [], []
        if n_total == 1:
            return indices, [], [] # 1 ảnh thì cho vào Train
        if n_total == 2:
            return [indices[0]], [indices[1]], [] # 2 ảnh thì 1 Train, 1 Val
            
        n_train = max(1, int(n_total * ratio_train))
        n_val = max(1, int(n_total * ratio_val))
        n_test = n_total - n_train - n_val
        
        # Đảm bảo có ít nhất 1 ảnh cho Test nếu tổng số ảnh >= 3
        if n_test <= 0 and n_total >= 3:
            n_test = 1
            n_val = max(1, int(n_total * ratio_val))
            n_train = n_total - n_val - n_test
            
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train+n_val]
        test_idx = indices[n_train+n_val:]
        return train_idx, val_idx, test_idx

    for cls_idx in range(num_classes):
        cls_indices = [i for i, t in enumerate(targets) if t == cls_idx]
        masked_indices = []
        normal_indices = []
        
        for i in cls_indices:
            path, _ = samples[i]
            path_lower = path.lower()
            is_masked = True if 'mask' in path_lower and 'unmask' not in path_lower and 'nomask' not in path_lower and 'normal' not in path_lower else False 
            if is_masked:
                masked_indices.append(i)
            else:
                normal_indices.append(i)
                
        # Split Masked
        t_m, v_m, test_m = split_group(masked_indices)
        train_indices.extend(t_m)
        val_indices.extend(v_m)
        test_indices.extend(test_m)
        
        # Split Normal
        t_n, v_n, test_n = split_group(normal_indices)
        train_indices.extend(t_n)
        val_indices.extend(v_n)
        test_indices.extend(test_n)
            
    train_dataset = Subset(dataset_train, train_indices)
    val_dataset = Subset(dataset_val, val_indices)
    test_dataset = Subset(dataset_val, test_indices) # Tập Test dùng riêng cho đánh giá cuối cùng

    print(f"Chia tập dữ liệu: {len(train_dataset)} Train - {len(val_dataset)} Val - {len(test_dataset)} Test.")

    # Bỏ persistent_workers=True vì nó có thể gây rò rỉ RAM (OOM) dần qua các epoch trên Kaggle
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False) # Không cần persistent cho test

    # 2. Khởi tạo mô hình
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nThiết bị huấn luyện: {device}")
    
    import timm
    
    class FaceClassifier(nn.Module):
        def __init__(self, num_classes):
            super().__init__()
            # Thay vì SwinWithCBAM, gọi trực tiếp Swin-T thuần từ timm
            self.feature_extractor = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=0)
            # Tăng Dropout lên 0.5 để cắt đứt mạnh mẽ các liên kết gây học vẹt
            self.dropout = nn.Dropout(p=0.5)
            
            # Thêm Bottleneck nén đặc trưng (Chuẩn Face Recognition)
            self.bottleneck = nn.Sequential(
                nn.Linear(768, 512),
                nn.BatchNorm1d(512),
                nn.PReLU()
            )
            self.head = nn.Linear(512, num_classes)
            
        def forward(self, x):
            features = self.feature_extractor(x)
            features = self.dropout(features)
            features = self.bottleneck(features)
            return self.head(features)
            
    model = FaceClassifier(num_classes=num_classes)
    
    model = model.to(device)

    # 3. Khởi tạo Optimizer (Tốc độ đa cấp - Differential Learning Rates)
    from sklearn.utils.class_weight import compute_class_weight
    targets = dataset_train.targets
    class_weights = compute_class_weight('balanced', classes=np.unique(targets), y=targets)
    class_weights = np.clip(class_weights, a_min=None, a_max=10.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Label Smoothing = 0.2 chống Overfitting quá đà khi cởi bỏ Dropout
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.2)
    
    # Không đóng băng Swin Backbone nữa vì đã dùng tạ LFW có sẵn đặc trưng khuôn mặt tốt
        
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'feature_extractor' in name:
            backbone_params.append(param)
        elif 'head' in name or 'bottleneck' in name:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': BACKBONE_LR}, 
        {'params': head_params, 'lr': LEARNING_RATE}    
    ], weight_decay=5e-2)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # Khởi tạo bộ tăng tốc AMP (Automatic Mixed Precision)
    scaler = torch.cuda.amp.GradScaler()

    # Kích hoạt Multi-GPU (DataParallel) ngay trước khi train nếu Kaggle có 2x T4
    if torch.cuda.device_count() > 1:
        print(f"\n[!] Phát hiện {torch.cuda.device_count()} GPUs. Đang kích hoạt Multi-GPU (DataParallel)...")
        model = nn.DataParallel(model)

    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_epoch = 1
    epochs_no_improve = 0
    start_epoch = 0

    # 3.5. Kế thừa trọng số LFW hoặc Resume thành quả đã học
    import glob
    custom_weights_path = MODEL_SAVE_PATH
    lfw_weights_path = "database/lfw_swin.pth"

    # Tự động tìm checkpoint mới nhất (VD: custom_swin_epoch_25.pth) nếu file chính không tồn tại
    if not os.path.exists(custom_weights_path):
        checkpoints = glob.glob(custom_weights_path.replace(".pth", "_epoch_*.pth"))
        if len(checkpoints) > 0:
            checkpoints.sort(key=lambda x: int(x.split('_epoch_')[-1].split('.pth')[0]))
            custom_weights_path = checkpoints[-1]

    if os.path.exists(custom_weights_path):
        print(f"\n[!] Tìm thấy {custom_weights_path}! Đang RESUME thành quả huấn luyện...")
        checkpoint = torch.load(custom_weights_path, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                if 'loss' in checkpoint:
                    best_val_loss = checkpoint['loss']
                if 'train_loss_history' in checkpoint:
                    train_loss_history = checkpoint['train_loss_history']
                    train_acc_history = checkpoint['train_acc_history']
                    val_loss_history = checkpoint['val_loss_history']
                    val_acc_history = checkpoint['val_acc_history']
                    print(f" -> Resume thành công tại Epoch {start_epoch}! Đã khôi phục Optimizer, Scheduler và Lịch sử.")
                else:
                    train_loss_history = [float('nan')] * start_epoch
                    train_acc_history = [float('nan')] * start_epoch
                    val_loss_history = [float('nan')] * start_epoch
                    val_acc_history = [float('nan')] * start_epoch
                    print(f" -> Resume thành công tại Epoch {start_epoch}! Đã khôi phục Optimizer và Scheduler (chưa có Lịch sử cũ).")
            else:
                print(" -> Resume thành công (chỉ có model state).")
        else:
            if isinstance(model, nn.DataParallel):
                model.module.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)
            print(" -> Resume trọng số mô hình thành công!")
            
        del checkpoint
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    elif os.path.exists(lfw_weights_path):
        print(f"\n[!] Tải trí nhớ LFW từ {lfw_weights_path}...")
        state_dict = torch.load(lfw_weights_path, map_location=device)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('head')}
        
        backbone_module = model.module.feature_extractor if isinstance(model, nn.DataParallel) else model.feature_extractor
        backbone_module.load_state_dict(state_dict, strict=False)
        print(" -> Thành công! Sẵn sàng chinh phục RMFRD.")



    print("\n Bắt đầu quá trình Huấn Luyện...")
    for epoch in range(start_epoch, NUM_EPOCHS):


        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_samples = 0
        
        optimizer.zero_grad() # Khởi tạo zero_grad ở đầu epoch

        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to(device), labels.to(device=device, dtype=torch.long)

            # Sử dụng AMP để tăng tốc x1.5 -> x2 lần
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # Gradient Accumulation: chia loss cho số bước tích lũy
                loss = loss / ACCUMULATION_STEPS
                
            scaler.scale(loss).backward()
            
            # Cập nhật trọng số sau mỗi ACCUMULATION_STEPS mini-batches
            if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(train_dataloader)):
                # Unscale trước khi clip grad norm để đảm bảo tính toán gradient chính xác
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += (loss.item() * ACCUMULATION_STEPS) * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels).item()
            total_samples += labels.size(0)

            current_lr = optimizer.param_groups[-1]['lr']
            # if (batch_idx + 1) % 400 == 0:
            #     print(f"   -> Epoch [{epoch+1}/{NUM_EPOCHS}] Batch [{batch_idx+1}/{len(train_dataloader)}] | Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
            
            # --- Giải phóng Tensor ngay lập tức để tránh tích tụ VRAM ---
            del inputs, labels, outputs, loss, preds
            
        scheduler.step()

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_preds / total_samples
        
        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_dataloader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device=device, dtype=torch.long)
                
                with torch.amp.autocast('cuda'):
                    val_outputs = model(val_inputs)
                    v_loss = criterion(val_outputs, val_labels)
                
                val_loss += v_loss.item() * val_inputs.size(0)
                _, val_preds = torch.max(val_outputs, 1)
                val_correct += torch.sum(val_preds == val_labels).item()
                val_total += val_labels.size(0)
                
                # --- Giải phóng Tensor ---
                del val_inputs, val_labels, val_outputs, v_loss, val_preds
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f} - LR: {current_lr:.6f}")

        # 4. Lưu mô hình tốt nhất (TẮT lưu định kỳ để tiết kiệm bộ nhớ)
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

        if epoch_val_acc >= best_val_acc and epoch_val_loss < best_val_loss:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            
            # Chỉ tạo state dict và lưu khi mô hình đạt điểm tốt nhất (tiết kiệm RAM)
            model_state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            state_to_save = {
                'epoch': epoch + 1,
                'model_state_dict': model_state_to_save,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_val_loss,
                'train_loss_history': train_loss_history,
                'train_acc_history': train_acc_history,
                'val_loss_history': val_loss_history,
                'val_acc_history': val_acc_history,
            }
            torch.save(state_to_save, MODEL_SAVE_PATH)
            print(f" --> Best Model Saved! (Val Acc: {best_val_acc:.4f} tại Epoch {best_epoch})")
            
            # Xóa ngay sau khi lưu
            del state_to_save, model_state_to_save
        else:
            epochs_no_improve += 1
            print(f" --> Không cải thiện trong {epochs_no_improve} epochs (Patience: {PATIENCE}).")
            if epochs_no_improve >= PATIENCE:
                print(f"\n[!] EARLY STOPPING KÍCH HOẠT: Mô hình đã hội tụ và ổn định. Dừng huấn luyện sớm tại epoch {epoch+1}.")
                break
        
        # Dọn dẹp bộ nhớ RAM và VRAM
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # 5. Vẽ biểu đồ Loss và Accuracy
    actual_epochs = len(train_loss_history)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, actual_epochs+1), train_loss_history, marker='o', color='purple', label='Train Loss')
    plt.plot(range(1, actual_epochs+1), val_loss_history, marker='o', color='red', label='Val Loss')
    plt.axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Loss Chart (Train vs Val)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, actual_epochs+1), train_acc_history, marker='o', color='green', label='Train Acc')
    plt.plot(range(1, actual_epochs+1), val_acc_history, marker='o', color='blue', label='Val Acc')
    plt.axvline(x=best_epoch, color='gray', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Accuracy Chart (Train vs Val)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.savefig(METRICS_SAVE_PATH, dpi=300)
    plt.close()
    
    # 6. Đánh giá chuyên sâu trên tập Test (Tập hoàn toàn mới)
    
    # --- Dọn dẹp RAM triệt để trước khi đánh giá ---
    print("\n[!] Dọn dẹp RAM trước quá trình Test...")
    del train_dataloader
    del val_dataloader
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    res_file = open("evaluation_results.txt", "w", encoding="utf-8")
    def log_result(text):
        print(text)
        res_file.write(text + "\n")
        res_file.flush()

    log_result("\n" + "="*60)
    log_result(" ĐÁNH GIÁ CHUYÊN SÂU MÔ HÌNH TỐT NHẤT TRÊN TẬP TEST")
    log_result("="*60)
    
    checkpoint = torch.load(MODEL_SAVE_PATH)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    model.eval()
    
    all_preds, all_labels, all_paths = [], [], []
    top1_correct = 0
    top5_correct = 0
    total_test = 0

    for idx in test_dataset.indices:
        path, _ = dataset_val.samples[idx]
        all_paths.append(path)
    
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Tính Top-1 và Top-5 Accuracy
            k = min(5, outputs.size(1))
            _, topk_preds = outputs.topk(k, 1, True, True)
            topk_preds = topk_preds.t()
            correct = topk_preds.eq(labels.view(1, -1).expand_as(topk_preds))
            
            top1_correct += correct[:1].reshape(-1).float().sum(0, keepdim=True).item()
            top5_correct += correct[:k].reshape(-1).float().sum(0, keepdim=True).item()
            total_test += labels.size(0)

    log_result(f"\n[KẾT QUẢ TỔNG QUÁT TRÊN TẬP TEST]")
    log_result(f" - Tổng số ảnh test: {total_test}")
    log_result(f" - Top-1 Accuracy:   {top1_correct / total_test:.4f}")
    log_result(f" - Top-5 Accuracy:   {top5_correct / total_test:.4f}")
            
    masked_preds, masked_labels = [], []
    unmasked_preds, unmasked_labels = [], []
    
    for i, path in enumerate(all_paths):
        path_lower = path.lower()
        is_masked = True if 'mask' in path_lower and 'unmask' not in path_lower and 'nomask' not in path_lower and 'normal' not in path_lower else False 
            
        if is_masked:
            masked_preds.append(all_preds[i])
            masked_labels.append(all_labels[i])
        else:
            unmasked_preds.append(all_preds[i])
            unmasked_labels.append(all_labels[i])

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    
    def print_metrics(name, labels, preds):
        log_result(f"\n[{name}]")
        if len(labels) > 0:
            acc = accuracy_score(labels, preds)
            p, r, f, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
            log_result(f" - Support:   {len(labels)}")
            log_result(f" - Accuracy:  {acc:.4f}")
            log_result(f" - Precision: {p:.4f}")
            log_result(f" - Recall:    {r:.4f}")
            log_result(f" - F1-Score:  {f:.4f}")
        else:
            log_result(" - Không có ảnh nào trong nhóm này.")

    print_metrics("ẢNH ĐEO KHẨU TRANG (MASKED)", masked_labels, masked_preds)
    print_metrics("ẢNH KHÔNG ĐEO (UNMASKED)", unmasked_labels, unmasked_preds)
            
    log_result("\nBáo cáo phân loại tổng thể đã được lưu. Chi tiết xem tại ma trận nhầm lẫn.")
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    # Tắt số liệu trong heatmap vì 380 classes số lượng quá dày đặc
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=False, yticklabels=False)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix) trên tập Test')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('custom_confusion_matrix.png', dpi=300)
    plt.close()
    
    log_result("\n HOÀN TẤT HUẤN LUYỆN RMFRD!")
    res_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện Swin Transformer trên RMFRD")
    parser.add_argument("--dataset_path", type=str, default="dataset/MaskedFace-Net", help="Đường dẫn dataset")
    args = parser.parse_args()
    main(args.dataset_path)
