#!/usr/bin/env python3
"""
GPU 效能測試腳本
測試不同 batch size 下的訓練速度和記憶體使用
"""

import torch
import torch.nn as nn
import time
import psutil
import yaml
import argparse
from pathlib import Path
import sys
import os

# 添加專案路徑
sys.path.append(str(Path(__file__).parent.parent))

from models.yolo import Model
from utils.general import check_img_size
from utils.torch_utils import select_device

class GPUBenchmark:
    def __init__(self, config_file="configs/gpu_configs.yaml"):
        self.config_file = Path(config_file)
        self.load_config()
        self.device = select_device('0')
        self.results = {}
        
    def load_config(self):
        """載入 GPU 配置"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def detect_gpu(self):
        """自動偵測 GPU 型號"""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"偵測到 GPU: {gpu_name}")
            
            # 簡化的 GPU 類型判斷
            if "4090" in gpu_name:
                return "RTX4090"
            elif "5090" in gpu_name:
                return "RTX5090" 
            elif "H100" in gpu_name:
                return "H100"
            elif "B200" in gpu_name:
                return "B200"
            else:
                return "Unknown"
        return None
    
    def create_model(self):
        """建立測試用模型"""
        cfg = "cfg/training/yolov7-tiny.yaml"
        model = Model(cfg, ch=3, nc=80, anchors=None).to(self.device)
        return model
    
    def benchmark_batch_size(self, model, batch_sizes, img_size=320):
        """測試不同 batch size 的效能"""
        results = {}
        
        for batch_size in batch_sizes:
            try:
                print(f"\n測試 batch_size: {batch_size}")
                
                # 清空 GPU 記憶體
                torch.cuda.empty_cache()
                
                # 建立測試資料
                dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
                dummy_targets = self.create_dummy_targets(batch_size)
                
                # 記憶體使用測試
                torch.cuda.reset_peak_memory_stats()
                
                # 前向傳播測試
                model.train()
                start_time = time.time()
                
                for _ in range(10):  # 測試 10 次取平均
                    with torch.cuda.amp.autocast():
                        outputs = model(dummy_input)
                        # 簡單的 loss 計算
                        loss = sum(output.mean() for output in outputs) if isinstance(outputs, (list, tuple)) else outputs.mean()
                        loss.backward()
                        model.zero_grad()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 10
                
                # 記錄結果
                peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                
                results[batch_size] = {
                    'avg_time_per_batch': avg_time,
                    'peak_memory_gb': peak_memory,
                    'fps': batch_size / avg_time,
                    'successful': True
                }
                
                print(f"  平均時間: {avg_time:.3f}s")
                print(f"  峰值記憶體: {peak_memory:.2f}GB") 
                print(f"  FPS: {batch_size/avg_time:.1f}")
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    results[batch_size] = {
                        'error': 'Out of Memory',
                        'successful': False
                    }
                    print(f"  記憶體不足!")
                    torch.cuda.empty_cache()
                else:
                    results[batch_size] = {
                        'error': str(e),
                        'successful': False
                    }
                    print(f"  錯誤: {e}")
        
        return results
    
    def create_dummy_targets(self, batch_size):
        """建立測試用標籤（簡化版本）"""
        # 這裡建立簡化的假目標，實際訓練時會有真實的標籤格式
        targets = []
        for i in range(batch_size):
            # 假設每張圖片有 2 個目標物件
            targets.extend([
                [i, 0, 0.5, 0.5, 0.2, 0.3],  # [batch_idx, class, x, y, w, h]
                [i, 1, 0.3, 0.7, 0.15, 0.25]
            ])
        return torch.tensor(targets).to(self.device)
    
    def run_benchmark(self, gpu_type=None):
        """執行完整效能測試"""
        if gpu_type is None:
            gpu_type = self.detect_gpu()
        
        if gpu_type not in self.config['gpu_configs']:
            print(f"未知的 GPU 類型: {gpu_type}")
            gpu_type = "RTX4090"  # 預設值
            
        gpu_config = self.config['gpu_configs'][gpu_type]
        print(f"使用 {gpu_config['name']} 設定進行測試")
        
        # 建立模型
        model = self.create_model()
        
        # 測試不同 batch size
        batch_sizes = gpu_config['optimal_batch_sizes']
        results = self.benchmark_batch_size(model, batch_sizes)
        
        # 儲存結果
        self.results[gpu_type] = {
            'gpu_info': gpu_config,
            'benchmark_results': results,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda
            }
        }
        
        return results
    
    def save_results(self, output_file="benchmark_results.yaml"):
        """儲存測試結果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.results, f, default_flow_style=False, allow_unicode=True)
        print(f"結果已儲存至: {output_file}")
    
    def print_summary(self, gpu_type):
        """列印測試摘要"""
        if gpu_type not in self.results:
            return
            
        results = self.results[gpu_type]['benchmark_results']
        
        print(f"\n=== {gpu_type} 效能測試摘要 ===")
        print(f"{'Batch Size':<12} {'Status':<12} {'Time/Batch':<12} {'Memory(GB)':<12} {'FPS':<8}")
        print("-" * 60)
        
        for batch_size, result in results.items():
            if result['successful']:
                print(f"{batch_size:<12} {'成功':<12} {result['avg_time_per_batch']:.3f}s{'':<6} {result['peak_memory_gb']:.2f}{'':<8} {result['fps']:.1f}")
            else:
                print(f"{batch_size:<12} {'失敗':<12} {result.get('error', '未知錯誤')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU 效能測試")
    parser.add_argument("--gpu-type", type=str, help="指定 GPU 類型 (RTX4090, RTX5090, H100, B200)")
    parser.add_argument("--config", type=str, default="configs/gpu_configs.yaml", help="配置檔案路徑")
    parser.add_argument("--output", type=str, default="benchmark_results.yaml", help="輸出檔案")
    
    args = parser.parse_args()
    
    benchmark = GPUBenchmark(args.config)
    results = benchmark.run_benchmark(args.gpu_type)
    benchmark.print_summary(args.gpu_type or benchmark.detect_gpu())
    benchmark.save_results(args.output)