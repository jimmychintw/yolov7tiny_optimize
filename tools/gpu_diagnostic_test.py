#!/usr/bin/env python3
"""
GPU 診斷測試 - 確認 GPU 是否真正在工作
"""

import torch
import time
import subprocess
import numpy as np

def run_command(cmd):
    """執行命令並返回輸出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout
    except:
        return "Command failed"

def gpu_stress_test():
    """GPU 壓力測試 - 確保 GPU 真的在工作"""
    print("=" * 60)
    print("🔬 GPU 診斷測試開始")
    print("=" * 60)
    
    # 1. 基本 CUDA 檢查
    print("\n1️⃣ 基本 CUDA 檢查:")
    print(f"   CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU 數量: {torch.cuda.device_count()}")
        print(f"   當前 GPU: {torch.cuda.current_device()}")
        print(f"   GPU 名稱: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA 版本: {torch.version.cuda}")
        
        # 檢查 GPU 屬性
        props = torch.cuda.get_device_properties(0)
        print(f"   計算能力: {props.major}.{props.minor}")
        print(f"   記憶體: {props.total_memory / 1024**3:.1f} GB")
    else:
        print("   ❌ CUDA 不可用！停止測試")
        return
    
    # 2. 檢查環境變數
    print("\n2️⃣ 環境變數檢查:")
    import os
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '未設定')
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # 3. 執行前的 nvidia-smi
    print("\n3️⃣ 測試前 nvidia-smi:")
    before_smi = run_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader")
    print(f"   {before_smi.strip()}")
    
    # 4. 簡單的 GPU 運算測試
    print("\n4️⃣ 執行簡單 GPU 運算:")
    device = torch.device('cuda:0')
    
    # 小矩陣測試
    print("   測試 1: 小矩陣乘法 (1000x1000)")
    a = torch.randn(1000, 1000).to(device)
    b = torch.randn(1000, 1000).to(device)
    
    torch.cuda.synchronize()
    start = time.time()
    for i in range(100):
        c = torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"   ✅ 完成 100 次運算，耗時: {elapsed:.3f}秒")
    
    # 檢查記憶體
    print(f"   記憶體使用: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # 5. 大矩陣壓力測試
    print("\n5️⃣ GPU 壓力測試 (10秒):")
    print("   創建大矩陣 (8192x8192)...")
    
    try:
        # 創建大矩陣
        size = 8192
        x = torch.randn(size, size, dtype=torch.float32).cuda()
        y = torch.randn(size, size, dtype=torch.float32).cuda()
        
        print(f"   矩陣大小: {x.element_size() * x.nelement() / 1024**3:.2f} GB 每個")
        print("   開始壓力測試...")
        
        # 壓力測試 10 秒
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < 10:
            # 執行矩陣運算
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            iteration += 1
            
            # 每秒檢查一次
            if iteration % 1 == 0:
                # 檢查 GPU 狀態
                gpu_stats = run_command("nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader")
                print(f"   迭代 {iteration}: {gpu_stats.strip()}")
        
        elapsed = time.time() - start_time
        print(f"\n   ✅ 完成 {iteration} 次迭代，總時間: {elapsed:.2f}秒")
        print(f"   平均速度: {iteration/elapsed:.2f} 次/秒")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("   ⚠️ GPU 記憶體不足，嘗試較小的矩陣")
            size = 4096
            x = torch.randn(size, size, dtype=torch.float32).cuda()
            y = torch.randn(size, size, dtype=torch.float32).cuda()
            # 重試較小的測試
        else:
            print(f"   ❌ 錯誤: {e}")
    
    # 6. 測試後檢查
    print("\n6️⃣ 測試後狀態:")
    after_smi = run_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw --format=csv,noheader")
    print(f"   {after_smi.strip()}")
    
    # 7. 檢查進程
    print("\n7️⃣ GPU 進程檢查:")
    processes = run_command("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader")
    if processes.strip():
        print(f"   找到進程:\n{processes}")
    else:
        print("   ❌ 沒有找到 GPU 進程！")
    
    # 8. 驗證 CUDA 設備
    print("\n8️⃣ CUDA 設備驗證:")
    print(f"   torch.cuda.current_device(): {torch.cuda.current_device()}")
    print(f"   torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")
    
    # 測試張量是否真的在 GPU 上
    test_tensor = torch.randn(100, 100).cuda()
    print(f"   測試張量設備: {test_tensor.device}")
    print(f"   is_cuda: {test_tensor.is_cuda}")
    
    # 9. 最終診斷
    print("\n" + "=" * 60)
    print("🔍 診斷結果:")
    
    # 解析 GPU 使用率
    try:
        gpu_util = float(after_smi.split(',')[0].strip().replace('%', ''))
        if gpu_util < 10:
            print("   ❌ GPU 使用率過低！可能沒有真正使用 GPU")
            print("   建議：")
            print("   1. 檢查 CUDA 安裝")
            print("   2. 確認 PyTorch 是 GPU 版本")
            print("   3. 檢查驅動程序")
        else:
            print(f"   ✅ GPU 正在工作 (使用率: {gpu_util}%)")
    except:
        print("   ⚠️ 無法解析 GPU 使用率")
    
    print("=" * 60)

if __name__ == "__main__":
    gpu_stress_test()
