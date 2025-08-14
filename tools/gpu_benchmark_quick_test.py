#!/usr/bin/env python3
"""
GPU 快速測試腳本 - 用於驗證程式碼是否正常運作
"""

import torch
import sys
from pathlib import Path

# 添加專案路徑
sys.path.append(str(Path(__file__).parent.parent))

def quick_test():
    """快速測試主要功能"""
    print("🔍 開始快速測試...")
    
    # 1. 檢查 CUDA
    print(f"✅ CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    
    # 2. 檢查必要套件
    packages = []
    try:
        import GPUtil
        packages.append("✅ GPUtil")
    except ImportError:
        packages.append("❌ GPUtil (需要: pip install gputil)")
    
    try:
        from tqdm import tqdm
        packages.append("✅ tqdm")
    except ImportError:
        packages.append("❌ tqdm (需要: pip install tqdm)")
    
    try:
        from models.yolo import Model
        packages.append("✅ YOLOv7 Model")
    except ImportError as e:
        packages.append(f"❌ YOLOv7 Model: {e}")
    
    try:
        from utils.loss import ComputeLoss
        packages.append("✅ ComputeLoss")
    except ImportError as e:
        packages.append(f"❌ ComputeLoss: {e}")
    
    print("\n📦 套件檢查:")
    for p in packages:
        print(f"   {p}")
    
    # 3. 測試模型建立
    if all("✅" in p for p in packages[:4]):
        print("\n🏗️  測試模型建立...")
        try:
            from models.yolo import Model
            from utils.loss import ComputeLoss
            import yaml
            
            model = Model("cfg/training/yolov7-tiny.yaml", ch=3, nc=80).cuda()
            
            # 載入超參數
            with open("data/hyp.scratch.tiny.yaml", 'r') as f:
                hyp = yaml.safe_load(f)
            model.hyp = hyp
            
            compute_loss = ComputeLoss(model)
            
            # 測試前向傳播
            dummy_input = torch.randn(1, 3, 320, 320).cuda()
            with torch.cuda.amp.autocast():
                outputs = model(dummy_input)
            
            print("✅ 模型前向傳播成功")
            
            # 測試 loss 計算
            dummy_targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.3]], dtype=torch.float32).cuda()
            loss, _ = compute_loss(outputs, dummy_targets)
            print(f"✅ Loss 計算成功: {loss.item():.4f}")
            
        except Exception as e:
            print(f"❌ 模型測試失敗: {e}")
    
    print("\n✨ 快速測試完成！")
    
    # 檢查是否有任何失敗
    if any("❌" in p for p in packages):
        print("\n⚠️  請先安裝缺少的套件再執行完整測試")
        return False
    return True

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n💡 可以執行完整測試:")
        print("   python tools/gpu_benchmark.py --quick")