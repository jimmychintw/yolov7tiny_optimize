#!/usr/bin/env python3
"""
YOLOv7 訓練監控器 - 自動解析訓練參數
使用方法：使用與 train.py 相同的參數，只是把 train.py 改成 monitor.py

範例：
python monitor.py --project runs/feasibility --name baseline_bs512_optimized
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np

def parse_args():
    """解析與 train.py 相同的參數"""
    parser = argparse.ArgumentParser(description='YOLOv7 Training Monitor')
    
    # 只需要這幾個參數來定位實驗
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    
    # 監控特定參數
    parser.add_argument('--refresh', type=int, default=5, help='refresh interval in seconds')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    
    # 忽略其他 train.py 參數（如果不小心傳入）
    parser.add_argument('--weights', default='')
    parser.add_argument('--cfg', default='')
    parser.add_argument('--data', default='')
    parser.add_argument('--hyp', default='')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640])
    parser.add_argument('--device', default='')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cache-images', action='store_true')
    # ... 可以加更多，但不影響監控
    
    return parser.parse_args()

def find_experiment_path(project, name, exist_ok):
    """根據參數找到實驗路徑"""
    base_path = Path(project)
    
    if exist_ok:
        # 如果 exist_ok，直接使用指定路徑
        exp_path = base_path / name
    else:
        # 否則找最新的帶數字後綴的版本
        exp_path = base_path / name
        if not exp_path.exists():
            # 嘗試找 exp2, exp3 等
            for i in range(2, 100):
                test_path = base_path / f"{name}{i}"
                if test_path.exists():
                    exp_path = test_path
                else:
                    break
    
    return exp_path

def get_gpu_stats():
    """獲取 GPU 狀態"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        values = result.stdout.strip().split(',')
        return {
            'util': float(values[0]),
            'mem_used': float(values[1]) / 1024,
            'mem_total': float(values[2]) / 1024,
            'temp': float(values[3]),
            'power': float(values[4])
        }
    except:
        return None

def get_training_stats(exp_path, verbose=False):
    """獲取訓練統計"""
    results_file = exp_path / "results.txt"
    
    if not results_file.exists():
        return None
    
    try:
        # 讀取結果，處理可能的格式問題
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 2:  # 至少需要 header + 1 行數據
            return None
            
        # 解析 header
        header = lines[0].strip().split()
        
        # 解析最後一行
        last_line = lines[-1].strip().split()
        
        if len(last_line) < 10:  # 確保有足夠的欄位
            return None
        
        # 創建 DataFrame 來找最佳結果
        try:
            df = pd.read_csv(results_file, sep='\s+', skipinitialspace=True)
        except Exception:
            # 如果 pandas 解析失敗，使用簡單解析
            df = None
        
        # 找出關鍵指標的索引
        epoch_idx = 0  # 通常第一欄
        map50_idx = None
        map_idx = None
        
        for i, col in enumerate(header):
            if 'mAP@.5' in col or 'mAP_0.5' in col:
                map50_idx = i
            if 'mAP@.5:.95' in col or 'mAP_0.5:0.95' in col:
                map_idx = i
        
        # 提取數據
        current_epoch = len(lines) - 1  # 減去 header
        
        stats = {
            'current_epoch': current_epoch,
            'train_loss': float(last_line[3]) if len(last_line) > 3 else 0,
            'val_loss': float(last_line[7]) if len(last_line) > 7 else 0,
        }
        
        # 嘗試獲取 mAP
        if map50_idx and len(last_line) > map50_idx:
            stats['mAP50'] = float(last_line[map50_idx])
        else:
            stats['mAP50'] = 0
            
        if map_idx and len(last_line) > map_idx:
            stats['mAP'] = float(last_line[map_idx])
        else:
            stats['mAP'] = 0
        
        # 找最佳結果
        if df is not None and 'metrics/mAP_0.5:0.95' in df.columns:
            best_idx = df['metrics/mAP_0.5:0.95'].idxmax()
            stats['best_mAP'] = df.loc[best_idx, 'metrics/mAP_0.5:0.95']
            stats['best_epoch'] = best_idx + 1
        else:
            stats['best_mAP'] = stats['mAP']
            stats['best_epoch'] = current_epoch
            
        return stats
        
    except Exception as e:
        if verbose:
            print(f"解析錯誤: {e}")
        return None

def format_time(seconds):
    """格式化時間"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def monitor(args):
    """主監控循環"""
    # 找到實驗路徑
    exp_path = find_experiment_path(args.project, args.name, args.exist_ok)
    
    print("\033[2J\033[H")  # 清屏
    print("=" * 80)
    print(f" YOLOv7 Training Monitor".center(80))
    print("=" * 80)
    print(f"📂 Monitoring: {exp_path}")
    print(f"🔄 Refresh: {args.refresh}s | Press Ctrl+C to exit")
    print("=" * 80)
    
    if not exp_path.exists():
        print(f"\n❌ 實驗路徑不存在: {exp_path}")
        print("\n可能的原因：")
        print("1. 訓練還沒開始")
        print("2. 參數不正確")
        print(f"3. 檢查是否需要 --exist-ok")
        return
    
    start_time = time.time()
    last_epoch = 0
    epoch_times = []
    
    while True:
        try:
            # 移動游標到數據區域
            print("\033[10;0H")  # 移到第10行
            
            # GPU 狀態
            gpu = get_gpu_stats()
            if gpu:
                # 進度條視覺化
                util_bar = "█" * int(gpu['util']/5) + "░" * (20-int(gpu['util']/5))
                mem_percent = gpu['mem_used'] / gpu['mem_total'] * 100
                mem_bar = "█" * int(mem_percent/5) + "░" * (20-int(mem_percent/5))
                
                print(f"\n{'─'*35} GPU Status {'─'*34}")
                print(f"Utilization: {gpu['util']:5.1f}% [{util_bar}]")
                print(f"Memory:      {gpu['mem_used']:5.1f}/{gpu['mem_total']:.0f}GB [{mem_bar}]")
                print(f"Temperature: {gpu['temp']:5.1f}°C | Power: {gpu['power']:5.1f}W")
                
                # 警告
                warnings = []
                if gpu['util'] < 70:
                    warnings.append("⚠️ Low GPU utilization")
                if gpu['temp'] > 80:
                    warnings.append("⚠️ High temperature")
                if mem_percent > 95:
                    warnings.append("⚠️ Memory nearly full")
                    
                if warnings:
                    print(f"Warnings: {', '.join(warnings)}")
            
            # 訓練統計
            stats = get_training_stats(exp_path, args.verbose)
            if stats:
                print(f"\n{'─'*35} Training {'─'*35}")
                
                # 計算 epoch 時間
                if stats['current_epoch'] > last_epoch:
                    if last_epoch > 0:
                        epoch_time = time.time() - start_time - sum(epoch_times)
                        epoch_times.append(epoch_time)
                    last_epoch = stats['current_epoch']
                
                # 進度條
                if args.epochs > 0:
                    progress = stats['current_epoch'] / args.epochs * 100
                    prog_bar = "█" * int(progress/2.5) + "░" * (40-int(progress/2.5))
                    print(f"Progress: [{prog_bar}] {stats['current_epoch']}/{args.epochs} ({progress:.1f}%)")
                else:
                    print(f"Epoch: {stats['current_epoch']}")
                
                # 時間估算
                elapsed = time.time() - start_time
                if epoch_times:
                    avg_epoch_time = np.mean(epoch_times[-10:])  # 最近10個epoch的平均
                    if args.epochs > 0:
                        eta = avg_epoch_time * (args.epochs - stats['current_epoch'])
                        print(f"Time: Elapsed {format_time(elapsed)} | ETA {format_time(eta)} | {avg_epoch_time:.1f}s/epoch")
                    else:
                        print(f"Time: Elapsed {format_time(elapsed)} | {avg_epoch_time:.1f}s/epoch")
                else:
                    print(f"Time: Elapsed {format_time(elapsed)}")
                
                # 損失和指標
                print(f"\nLosses:  Train {stats['train_loss']:.4f} | Val {stats['val_loss']:.4f}")
                print(f"Metrics: mAP@0.5 {stats['mAP50']:.4f} | mAP@0.5:0.95 {stats['mAP']:.4f}")
                print(f"Best:    mAP@0.5:0.95 {stats['best_mAP']:.4f} @ Epoch {stats['best_epoch']}")
                
                # 收斂判斷
                if stats['current_epoch'] > 20 and epoch_times:
                    recent_improvement = abs(stats['mAP'] - stats['best_mAP'])
                    if recent_improvement < 0.001 and stats['current_epoch'] - stats['best_epoch'] > 10:
                        print("\n📌 Model appears to be converging (no improvement for 10+ epochs)")
                        
            else:
                print(f"\n⏳ Waiting for training data...")
                print(f"   Checking: {exp_path / 'results.txt'}")
            
            # 底部信息
            print(f"\n{'─'*80}")
            print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            time.sleep(args.refresh)
            
        except KeyboardInterrupt:
            print("\n\n✋ Monitoring stopped by user")
            break
        except Exception as e:
            if args.verbose:
                print(f"\n❌ Error: {e}")
            time.sleep(args.refresh)

if __name__ == "__main__":
    args = parse_args()
    monitor(args)