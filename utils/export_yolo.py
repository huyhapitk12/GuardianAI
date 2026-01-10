import os
import shutil
from pathlib import Path
from ultralytics import YOLO
import torch

def export_models():
    # Định nghĩa các model cần xử lý
    models_config = {
        'Small': 'yolov8s.pt',
        'Medium': 'yolov8m.pt'
    }
    
    # Đường dẫn base: Data/Model
    base_dir = Path(__file__).parent.parent / 'Data' / 'Model'
    
    for size, model_name in models_config.items():
        print(f"\n{'='*50}")
        print(f"Đang xử lý model {size} ({model_name})...")
        print(f"{'='*50}")
        
        # Tạo thư mục đích: Data/Model/{Size}/Person
        target_dir = base_dir / size / 'Person'
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # Đường dẫn file model đích
        pt_path = target_dir / f"{size}.pt"
        onnx_path = target_dir / f"{size}.onnx"
        engine_path = target_dir / f"{size}.engine"
        
        # 1. Tải và lưu model PyTorch (.pt)
        if not pt_path.exists():
            print(f"[INFO] Đang tải {model_name}...")
            # Load model (sẽ tự tải về cache nếu chưa có)
            model = YOLO(model_name)
            
            print(f"[INFO] Lưu model vào {pt_path}...")
            
            # YOLO() download file về current dir.
            if Path(model_name).exists():
                shutil.move(model_name, pt_path)
            else:
                pass
            
            # Kiểm tra lại
            if not pt_path.exists() and Path(model_name).exists():
                shutil.move(model_name, pt_path)
                
            # Reload from path
            model = YOLO(str(pt_path))
        else:
            print(f"[OK] Đã có model PyTorch: {pt_path}")
            model = YOLO(str(pt_path))

        # 2. Export sang TensorRT
        if engine_path.exists():
            print(f"[OK] Đã có model TensorRT: {engine_path}")
        else:
            print(f"[INFO] Đang export sang TensorRT (cần GPU)...")
        if torch.cuda.is_available():
            # format='engine' : TensorRT
            # device=0 : GPU 0
            model.export(format='engine', device=0, verbose=True)
            print(f"[OK] Export thành công!")
        else:
            print(f"[WARN] Không tìm thấy GPU, bỏ qua export TensorRT.")
            print(f"[INFO] Bạn cần chạy script này trên máy có GPU NVIDIA.")

if __name__ == "__main__":
    export_models()
