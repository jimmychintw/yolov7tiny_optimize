#!/usr/bin/env python3
"""
GPU 真相探針 - 確認環境能真正使用 GPU 且被 nvidia-smi 檢測到
設計理念：不能靜默退回 CPU，會讓 GPU 忙 10-15 秒的最小程式
"""

import torch
import time
import argparse
import sys

def gpu_truth_probe(seconds=15, size=8192, dtype='bf16'):
    """GPU 真相探針 - 確保 GPU 真的在工作"""
    print("=" * 60)
    print("🔬 GPU 真相探針啟動")
    print("=" * 60)
    
    # 1. 嚴格檢查 CUDA
    if not torch.cuda.is_available():
        print("❌ 致命錯誤: CUDA 不可用!")
        sys.exit(1)
    
    device = torch.device('cuda:0')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"🎯 目標 GPU: {gpu_name}")
    print(f"🔧 矩陣大小: {size}x{size}")
    print(f"⏱️  測試時長: {seconds} 秒")
    print(f"🔢 數據類型: {dtype}")
    
    # 2. 設定數據類型
    if dtype == 'bf16':
        torch_dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif dtype == 'fp16':
        torch_dtype = torch.float16  
        autocast_dtype = torch.float16
    else:
        torch_dtype = torch.float32
        autocast_dtype = torch.float32
    
    # 3. 啟用最佳化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    # 4. 創建大矩陣確保 GPU 忙碌
    print(f"\n📦 創建 {size}x{size} 矩陣...")
    try:
        # 使用指定的數據類型
        x = torch.randn(size, size, dtype=torch_dtype, device=device)
        y = torch.randn(size, size, dtype=torch_dtype, device=device)
        
        memory_gb = (x.element_size() * x.nelement() + y.element_size() * y.nelement()) / 1024**3
        print(f"📊 GPU 記憶體使用: {memory_gb:.2f} GB")
        
        # 強制驗證在 CUDA 上
        assert x.is_cuda and y.is_cuda, "❌ 矩陣不在 CUDA 上"
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"⚠️  記憶體不足，嘗試較小矩陣 ({size//2}x{size//2})")
            size = size // 2
            x = torch.randn(size, size, dtype=torch_dtype, device=device)
            y = torch.randn(size, size, dtype=torch_dtype, device=device)
        else:
            raise e
    
    # 5. 開始壓力測試
    print(f"\n🔥 開始 {seconds} 秒壓力測試...")
    print("💡 請同時執行: watch -n 0.5 nvidia-smi")
    print("-" * 60)
    
    start_time = time.time()
    iteration = 0
    total_iterations = 0
    
    while time.time() - start_time < seconds:
        # 使用 CUDA Events 精確計時
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        # 密集矩陣運算
        with torch.cuda.amp.autocast(dtype=autocast_dtype):
            z = torch.matmul(x, y)
            # 額外運算確保 GPU 真的忙碌
            z = torch.matmul(z, x)
            z = z + torch.randn_like(z) * 0.1
            
        # 讓 GPU 忙得夠久，nvidia-smi 才看得到
        torch.cuda._sleep(100_000_000)  # 約 100ms
        
        end_event.record()
        torch.cuda.synchronize()
        
        iteration += 1
        total_iterations += 1
        
        # 每秒報告一次
        if iteration >= 3:  # 約 1 秒
            elapsed_ms = start_event.elapsed_time(end_event)
            gpu_memory = torch.cuda.memory_allocated() / 1024**2
            print(f"迭代 {total_iterations:3d}: {elapsed_ms:.1f}ms/iter | GPU記憶體: {gpu_memory:.0f}MB | 已運行: {time.time() - start_time:.1f}s")
            iteration = 0
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"✅ 測試完成!")
    print(f"📊 總迭代: {total_iterations}")
    print(f"⏱️  總時間: {total_time:.2f} 秒")
    print(f"🚀 平均速度: {total_iterations/total_time:.2f} 次/秒")
    
    # 6. 最終驗證
    final_memory = torch.cuda.memory_allocated() / 1024**2
    print(f"💾 最終 GPU 記憶體: {final_memory:.1f} MB")
    
    if final_memory < 100:
        print("⚠️  警告: GPU 記憶體使用過低，可能未真正使用 GPU")
        return False
    
    print("\n🎯 診斷結論:")
    print("   如果 nvidia-smi 顯示:")
    print("   ✅ GPU 使用率 > 80% → 環境正常")
    print("   ✅ 看到 python 進程 → 進程可見")
    print("   ❌ GPU 使用率 < 10% → 可能有問題")
    print("   ❌ 沒有進程 → 容器權限限制")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="GPU 真相探針")
    parser.add_argument("--seconds", type=int, default=15, help="測試時長（秒）")
    parser.add_argument("--size", type=int, default=8192, help="矩陣大小")
    parser.add_argument("--dtype", choices=['fp32', 'fp16', 'bf16'], default='bf16', help="數據類型")
    
    args = parser.parse_args()
    
    success = gpu_truth_probe(args.seconds, args.size, args.dtype)
    
    if success:
        print("\n✅ GPU 環境檢查通過，可以執行基準測試")
        print("   python tools/gpu_benchmark.py")
    else:
        print("\n❌ GPU 環境可能有問題，請檢查")
        sys.exit(1)

if __name__ == "__main__":
    main()