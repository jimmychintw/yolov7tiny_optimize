#!/usr/bin/env python3
"""
GPU/CUDA 環境檢查工具 - 確保真的在使用 GPU
"""

import torch
import sys
import subprocess

def check_cuda_environment():
    """嚴格檢查 CUDA 環境"""
    print("=" * 60)
    print("🔍 GPU/CUDA 環境嚴格檢查")
    print("=" * 60)
    
    # 1. 檢查 PyTorch CUDA
    print("\n1️⃣ PyTorch CUDA 檢查:")
    cuda_available = torch.cuda.is_available()
    print(f"   CUDA 可用: {cuda_available}")
    
    if not cuda_available:
        print("   ❌ 錯誤: PyTorch 無法使用 CUDA!")
        print("   可能原因:")
        print("   - PyTorch 安裝版本不支援 CUDA")
        print("   - CUDA 驅動版本不相容")
        print("   - 沒有 NVIDIA GPU")
        return False
    
    print(f"   PyTorch 版本: {torch.__version__}")
    print(f"   PyTorch CUDA 版本: {torch.version.cuda}")
    print(f"   CUDNN 版本: {torch.backends.cudnn.version()}")
    
    # 2. 檢查 GPU 資訊
    print("\n2️⃣ GPU 硬體檢查:")
    device_count = torch.cuda.device_count()
    print(f"   GPU 數量: {device_count}")
    
    if device_count == 0:
        print("   ❌ 錯誤: 找不到任何 GPU!")
        return False
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"   GPU {i}: {props.name}")
        print(f"      記憶體: {props.total_memory / 1024**3:.1f} GB")
        print(f"      計算能力: {props.major}.{props.minor}")
    
    # 3. nvidia-smi 檢查
    print("\n3️⃣ nvidia-smi 檢查:")
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                               '--format=csv,noheader'], 
                              capture_output=True, text=True, check=True)
        print("   系統 GPU 資訊:")
        for line in result.stdout.strip().split('\n'):
            print(f"   - {line}")
    except Exception as e:
        print(f"   ⚠️ nvidia-smi 執行失敗: {e}")
    
    # 4. 實際 GPU 測試
    print("\n4️⃣ 實際 GPU 運算測試:")
    try:
        # 建立測試張量
        device = torch.device('cuda:0')
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        
        # 執行矩陣運算
        torch.cuda.synchronize()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        print(f"   ✅ GPU 運算成功")
        print(f"   測試裝置: {device}")
        print(f"   記憶體已使用: {torch.cuda.memory_allocated(device) / 1024**2:.1f} MB")
        
    except Exception as e:
        print(f"   ❌ GPU 運算失敗: {e}")
        return False
    
    # 5. 版本相容性檢查
    print("\n5️⃣ 版本相容性:")
    try:
        # 取得系統 CUDA 版本
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    print(f"   系統 CUDA: {line.strip()}")
        
        print(f"   PyTorch CUDA: {torch.version.cuda}")
        
        # 檢查版本匹配
        if torch.version.cuda:
            pytorch_cuda = float(torch.version.cuda.split('.')[0] + '.' + torch.version.cuda.split('.')[1])
            print(f"   版本檢查: PyTorch CUDA {pytorch_cuda}")
            
            if pytorch_cuda < 11.8:
                print("   ⚠️ 警告: H100 建議使用 CUDA 11.8 或更高版本")
        
    except Exception as e:
        print(f"   ⚠️ 無法檢查 nvcc: {e}")
    
    print("\n" + "=" * 60)
    print("✅ GPU/CUDA 環境檢查完成 - 可以使用 GPU!")
    print("=" * 60)
    return True

def main():
    """主程式"""
    success = check_cuda_environment()
    
    if not success:
        print("\n❌ GPU 環境有問題，請修復後再執行測試!")
        print("\n建議解決方案:")
        print("1. 重新安裝正確版本的 PyTorch:")
        print("   pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118")
        print("2. 確認 NVIDIA 驅動正確安裝:")
        print("   nvidia-smi")
        print("3. 確認 CUDA 工具包安裝:")
        print("   nvcc --version")
        sys.exit(1)
    else:
        print("\n✅ 可以執行 GPU 基準測試!")
        print("   python tools/gpu_benchmark.py")

if __name__ == "__main__":
    main()