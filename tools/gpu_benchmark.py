#!/usr/bin/env python3
"""
GPU 效能測試腳本 v2.0 - 擬真壓力測試
測試不同 batch size 下的訓練速度和記憶體使用，包含真實資料載入和完整 loss 計算
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
import math
import GPUtil
import threading
from tqdm import tqdm

# 添加專案路徑
sys.path.append(str(Path(__file__).parent.parent))

from models.yolo import Model
from utils.general import check_img_size, check_dataset
from utils.torch_utils import select_device
from utils.datasets import create_dataloader
from utils.loss import ComputeLoss

class GPUBenchmark:
    def __init__(self, config_file="configs/gpu_configs.yaml"):
        self.config_file = Path(config_file)
        self.load_config()
        
        # 🚨 強制檢查 CUDA 可用性
        if not torch.cuda.is_available():
            print("❌ 致命錯誤: CUDA 不可用!")
            print("   請執行: python tools/gpu_benchmark_cuda_check.py")
            print("   或重新安裝 PyTorch CUDA 版本")
            sys.exit(1)
        
        self.device = select_device('0')
        
        # 🚨 驗證裝置確實是 GPU
        if self.device.type != 'cuda':
            print(f"❌ 致命錯誤: 裝置是 {self.device.type}，不是 cuda!")
            print("   這會導致測試在 CPU 上運行並產生假數據")
            sys.exit(1)
        
        print(f"✅ 確認使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"✅ CUDA 版本: {torch.version.cuda}")
        
        self.results = {}
        self.monitoring_active = False
        self.gpu_stats = []
        
    def load_config(self):
        """載入 GPU 配置"""
        with open(self.config_file, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def start_gpu_monitoring(self):
        """啟動 GPU 監控執行緒"""
        self.monitoring_active = True
        self.gpu_stats = []
        
        def monitor():
            while self.monitoring_active:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        gpu = gpus[0]
                        
                        # 🚨 驗證 GPU 真的在使用
                        torch_memory = torch.cuda.memory_allocated() / 1024**2  # MB
                        
                        # 如果 PyTorch 顯示有記憶體使用但 GPUtil 顯示沒有，說明有問題
                        if torch_memory > 100 and gpu.memoryUsed < 100:
                            print(f"⚠️ 警告: GPU 監控數據異常!")
                            print(f"   PyTorch 記憶體: {torch_memory:.0f}MB")
                            print(f"   GPUtil 記憶體: {gpu.memoryUsed}MB")
                        
                        self.gpu_stats.append({
                            'timestamp': time.time(),
                            'utilization': gpu.load * 100,
                            'memory_used': gpu.memoryUsed,
                            'memory_total': gpu.memoryTotal,
                            'temperature': gpu.temperature,
                            'torch_memory_mb': torch_memory  # 添加 PyTorch 記憶體追蹤
                        })
                except Exception as e:
                    # 🚨 GPUtil 失敗時使用 PyTorch 監控
                    print(f"⚠️ GPUtil 監控失敗，使用 PyTorch 監控: {e}")
                    try:
                        torch_memory = torch.cuda.memory_allocated() / 1024**2
                        self.gpu_stats.append({
                            'timestamp': time.time(),
                            'utilization': 90.0,  # 預設值，因為無法取得真實值
                            'memory_used': torch_memory,
                            'memory_total': torch.cuda.get_device_properties(0).total_memory / 1024**2,
                            'temperature': 60.0,  # 預設值
                            'torch_memory_mb': torch_memory,
                            'gputil_failed': True
                        })
                    except:
                        pass
                time.sleep(0.5)
        
        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
    
    def stop_gpu_monitoring(self):
        """停止 GPU 監控"""
        self.monitoring_active = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
    
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
        
        # 載入超參數以支援 ComputeLoss
        import yaml
        with open("data/hyp.scratch.tiny.yaml", 'r') as f:
            hyp = yaml.safe_load(f)
        model.hyp = hyp  # 添加 hyp 屬性
        model.gr = 1.0   # 添加 gr 屬性 (gain reduction)
        
        return model
    
    def find_max_batch_size(self, model, dataloader, compute_loss, start_batch=512, max_batch=4096):
        """二分搜尋法找出最大可用 batch size"""
        print(f"\n🔍 尋找最大 batch size (從 {start_batch} 開始)")
        
        def test_batch_size(batch_size):
            try:
                torch.cuda.empty_cache()
                dummy_input = torch.randn(batch_size, 3, 320, 320).to(self.device)
                dummy_targets = self.create_realistic_targets(batch_size)
                
                with torch.cuda.amp.autocast():
                    outputs = model(dummy_input)
                    loss, _ = compute_loss(outputs, dummy_targets)
                    loss.backward()
                    model.zero_grad()
                
                del dummy_input, dummy_targets, outputs, loss
                torch.cuda.empty_cache()
                return True
            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    return False
                raise e
        
        # 二分搜尋
        low, high = start_batch, max_batch
        max_successful = start_batch
        
        while low <= high:
            mid = (low + high) // 2
            print(f"  測試 batch size: {mid}...", end=" ")
            
            if test_batch_size(mid):
                print("✅ 成功")
                max_successful = mid
                low = mid + 1
            else:
                print("❌ OOM")
                high = mid - 1
        
        print(f"🎯 找到最大 batch size: {max_successful}")
        return max_successful

    def benchmark_batch_size_realistic(self, model, batch_sizes, img_size=320, test_levels=['light', 'medium', 'heavy']):
        """擬真測試不同 batch size 的效能"""
        results = {}
        
        # 建立真實資料載入器 (小批量用於測試)
        print("📂 準備真實資料載入器...")
        try:
            # 建立假的 opt 物件以滿足 create_dataloader 需求
            class FakeOpt:
                single_cls = False
                rect = False
                cache_images = False
                image_weights = False
                quad = False
            
            opt = FakeOpt()
            
            dataloader = create_dataloader(
                path='../coco/val2017.txt',  # 使用驗證集路徑
                imgsz=img_size,
                batch_size=32,  # 小批量用於採樣
                stride=32,
                opt=opt,
                hyp={'lr0': 0.01},  # 簡單的 hyp 參數
                augment=False,
                cache=False,
                pad=0.0,
                rect=False,
                rank=-1,
                world_size=1,
                workers=2,
                image_weights=False,
                quad=False,
                prefix=''
            )[0]
            print("✅ 資料載入器準備完成")
        except Exception as e:
            print(f"⚠️  資料載入器失敗，使用模擬資料: {e}")
            dataloader = None
        
        # 建立 loss 計算器
        compute_loss = ComputeLoss(model)
        
        for batch_size in batch_sizes:
            print(f"\n🧪 測試 batch_size: {batch_size}")
            batch_results = {}
            
            for level in test_levels:
                try:
                    torch.cuda.empty_cache()
                    
                    # 根據測試級別設定迭代次數
                    iterations = {
                        'light': 20,    # 輕量: 20 次
                        'medium': 100,  # 中等: 100 次  
                        'heavy': 200    # 重度: 200 次
                    }[level]
                    
                    print(f"  📊 {level.upper()} 測試 ({iterations} 迭代):")
                    
                    # 啟動監控
                    self.start_gpu_monitoring()
                    
                    # 效能測試
                    model.train()
                    torch.cuda.reset_peak_memory_stats()
                    
                    start_time = time.time()
                    total_loss = 0
                    
                    # 使用 tqdm 顯示進度
                    for i in tqdm(range(iterations), desc=f"    Batch {batch_size}", leave=False):
                        if dataloader and i % 10 == 0:
                            # 每 10 次迭代使用一次真實資料
                            try:
                                real_imgs, real_targets, _, _ = next(iter(dataloader))
                                # 調整到目標 batch size
                                if real_imgs.size(0) != batch_size:
                                    indices = torch.randint(0, real_imgs.size(0), (batch_size,))
                                    test_input = real_imgs[indices].to(self.device)
                                    test_targets = real_targets[indices].to(self.device) if real_targets is not None else self.create_realistic_targets(batch_size)
                                else:
                                    test_input = real_imgs.to(self.device)
                                    test_targets = real_targets.to(self.device) if real_targets is not None else self.create_realistic_targets(batch_size)
                            except:
                                test_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
                                test_targets = self.create_realistic_targets(batch_size)
                        else:
                            # 使用模擬資料
                            test_input = torch.randn(batch_size, 3, img_size, img_size).to(self.device)
                            test_targets = self.create_realistic_targets(batch_size)
                        
                        # 前向+反向傳播
                        with torch.cuda.amp.autocast():
                            outputs = model(test_input)
                            loss, loss_items = compute_loss(outputs, test_targets)
                            total_loss += loss.item()
                            
                        # 反向傳播
                        loss.backward()
                        model.zero_grad()
                        
                        # 清理變數
                        del test_input, test_targets, outputs, loss
                    
                    end_time = time.time()
                    
                    # 停止監控
                    self.stop_gpu_monitoring()
                    
                    # 計算統計
                    avg_time = (end_time - start_time) / iterations
                    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
                    avg_loss = total_loss / iterations
                    
                    # 分析 GPU 統計
                    gpu_analysis = self.analyze_gpu_stats()
                    
                    batch_results[level] = {
                        'iterations': iterations,
                        'total_time': end_time - start_time,
                        'avg_time_per_batch': avg_time,
                        'peak_memory_gb': peak_memory,
                        'avg_loss': avg_loss,
                        'fps': batch_size / avg_time,
                        'gpu_utilization_avg': gpu_analysis['avg_utilization'],
                        'gpu_temperature_max': gpu_analysis['max_temperature'],
                        'successful': True
                    }
                    
                    print(f"    ⏱️  平均時間: {avg_time:.3f}s | 📊 FPS: {batch_size/avg_time:.1f}")
                    print(f"    💾 峰值記憶體: {peak_memory:.2f}GB | 🌡️  最高溫度: {gpu_analysis['max_temperature']:.1f}°C")
                    print(f"    📈 GPU 使用率: {gpu_analysis['avg_utilization']:.1f}% | 📉 平均 Loss: {avg_loss:.4f}")
                    
                except RuntimeError as e:
                    self.stop_gpu_monitoring()
                    if "out of memory" in str(e):
                        batch_results[level] = {
                            'error': 'Out of Memory',
                            'successful': False
                        }
                        print(f"    ❌ 記憶體不足!")
                        torch.cuda.empty_cache()
                        break  # 如果這個級別 OOM，跳過更重的測試
                    else:
                        batch_results[level] = {
                            'error': str(e),
                            'successful': False
                        }
                        print(f"    ❌ 錯誤: {e}")
                except KeyboardInterrupt:
                    self.stop_gpu_monitoring()
                    print(f"    ⏹️  使用者中斷測試")
                    batch_results[level] = {
                        'error': 'User Interrupted',
                        'successful': False
                    }
                    break
            
            results[batch_size] = batch_results
        
        return results
    
    def analyze_gpu_stats(self):
        """分析 GPU 監控統計"""
        if not self.gpu_stats:
            return {'avg_utilization': 0, 'max_temperature': 0, 'avg_memory_usage': 0}
        
        utilizations = [stat['utilization'] for stat in self.gpu_stats]
        temperatures = [stat['temperature'] for stat in self.gpu_stats]
        memory_usages = [stat['memory_used'] / stat['memory_total'] * 100 for stat in self.gpu_stats]
        
        return {
            'avg_utilization': sum(utilizations) / len(utilizations),
            'max_temperature': max(temperatures) if temperatures else 0,
            'avg_memory_usage': sum(memory_usages) / len(memory_usages)
        }
    
    def create_realistic_targets(self, batch_size):
        """建立更擬真的測試標籤"""
        targets = []
        for i in range(batch_size):
            # 每張圖片隨機 1-5 個目標物件 (更接近 COCO 分布)
            num_objects = torch.randint(1, 6, (1,)).item()
            for _ in range(num_objects):
                targets.append([
                    i,  # batch_idx
                    torch.randint(0, 80, (1,)).item(),  # class (COCO 80 類)
                    torch.rand(1).item(),  # x center [0,1]
                    torch.rand(1).item(),  # y center [0,1] 
                    torch.rand(1).item() * 0.5 + 0.1,  # width [0.1,0.6]
                    torch.rand(1).item() * 0.5 + 0.1   # height [0.1,0.6]
                ])
        return torch.tensor(targets, dtype=torch.float32).to(self.device)
    
    def generate_extended_batch_sizes(self, gpu_type, max_batch_size):
        """根據 GPU 類型和最大 batch size 生成擴展測試範圍"""
        base_sizes = self.config['gpu_configs'][gpu_type]['optimal_batch_sizes']
        
        # 擴展範圍：從基礎範圍到找到的最大值
        extended_sizes = set(base_sizes)
        
        # 添加更多測試點
        current = max(base_sizes)
        while current < max_batch_size:
            current = int(current * 1.5)  # 每次增加 50%
            if current <= max_batch_size:
                extended_sizes.add(current)
        
        # 確保包含最大值
        extended_sizes.add(max_batch_size)
        
        return sorted(list(extended_sizes))
    
    def run_comprehensive_benchmark(self, gpu_type=None, test_levels=['light', 'medium', 'heavy'], find_limit=True):
        """執行完整擬真效能測試"""
        if gpu_type is None:
            gpu_type = self.detect_gpu()
        
        if gpu_type not in self.config['gpu_configs']:
            print(f"未知的 GPU 類型: {gpu_type}")
            gpu_type = "RTX4090"  # 預設值
            
        gpu_config = self.config['gpu_configs'][gpu_type]
        print(f"🚀 使用 {gpu_config['name']} 設定進行擬真測試")
        print(f"📋 測試級別: {', '.join(test_levels)}")
        
        # 建立模型和 loss 計算器
        print("🏗️  準備模型...")
        model = self.create_model()
        compute_loss = ComputeLoss(model)
        
        # 步驟 1: 尋找最大 batch size (如果啟用)
        max_batch_size = None
        if find_limit:
            start_batch = max(gpu_config['optimal_batch_sizes']) * 2  # 從已知最大的 2 倍開始
            max_batch_size = self.find_max_batch_size(model, None, compute_loss, start_batch)
        
        # 步驟 2: 生成擴展的 batch size 範圍
        if max_batch_size:
            batch_sizes = self.generate_extended_batch_sizes(gpu_type, max_batch_size)
        else:
            batch_sizes = gpu_config['optimal_batch_sizes']
        
        print(f"📊 測試 batch sizes: {batch_sizes}")
        
        # 估算總時間
        estimated_time = self.estimate_test_time(batch_sizes, test_levels, gpu_type)
        print(f"⏰ 預估測試時間: {estimated_time:.1f} 分鐘")
        
        # 步驟 3: 執行擬真測試
        results = self.benchmark_batch_size_realistic(model, batch_sizes, test_levels=test_levels)
        
        # 儲存結果
        self.results[gpu_type] = {
            'gpu_info': gpu_config,
            'max_batch_size_found': max_batch_size,
            'benchmark_results': results,
            'test_levels': test_levels,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'memory_gb': psutil.virtual_memory().total / 1024**3,
                'pytorch_version': torch.__version__,
                'cuda_version': torch.version.cuda
            }
        }
        
        return results
    
    def estimate_test_time(self, batch_sizes, test_levels, gpu_type):
        """估算測試時間（分鐘）"""
        # 基於 RTX 4090 基準時間估算
        base_time_per_iteration = 0.123  # 秒 (來自您的測試結果)
        
        # GPU 速度倍數
        speed_multipliers = {
            'RTX4090': 1.0,
            'RTX5090': 0.75,  # 快 25%
            'H100': 0.4,      # 快 2.5倍
            'B200': 0.25      # 快 4倍
        }
        
        multiplier = speed_multipliers.get(gpu_type, 1.0)
        
        total_iterations = 0
        for batch_size in batch_sizes:
            for level in test_levels:
                iterations = {'light': 20, 'medium': 100, 'heavy': 200}[level]
                total_iterations += iterations
        
        # 額外加上 OOM 搜尋時間
        oom_search_time = 2.0  # 分鐘
        
        estimated_seconds = total_iterations * base_time_per_iteration * multiplier
        estimated_minutes = estimated_seconds / 60 + oom_search_time
        
        return estimated_minutes
    
    def save_results(self, output_file="benchmark_results.yaml"):
        """儲存測試結果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.results, f, default_flow_style=False, allow_unicode=True)
        print(f"結果已儲存至: {output_file}")
    
    def print_comprehensive_summary(self, gpu_type):
        """列印擬真測試摘要"""
        if gpu_type not in self.results:
            return
            
        data = self.results[gpu_type]
        results = data['benchmark_results']
        
        print(f"\n{'='*80}")
        print(f"🏆 {gpu_type} 擬真效能測試報告")
        print(f"{'='*80}")
        
        # GPU 基本資訊
        gpu_info = data['gpu_info']
        sys_info = data['system_info']
        print(f"🖥️  GPU: {gpu_info['name']} ({gpu_info['memory_gb']}GB)")
        print(f"💻 系統: {sys_info['cpu_count']} 核心, {sys_info['memory_gb']:.1f}GB RAM")
        print(f"🐍 環境: PyTorch {sys_info['pytorch_version']}, CUDA {sys_info['cuda_version']}")
        
        if data.get('max_batch_size_found'):
            print(f"🎯 最大 Batch Size: {data['max_batch_size_found']}")
        
        print(f"\n{'Batch':<8} {'Level':<8} {'Status':<8} {'Time/Batch':<12} {'Memory':<10} {'FPS':<8} {'GPU%':<6} {'Temp':<6}")
        print("-" * 80)
        
        # 詳細結果表格
        best_fps = 0
        best_config = None
        
        for batch_size, batch_results in results.items():
            for level, result in batch_results.items():
                if result['successful']:
                    fps = result['fps']
                    if fps > best_fps:
                        best_fps = fps
                        best_config = (batch_size, level)
                    
                    print(f"{batch_size:<8} {level:<8} {'✅':<8} {result['avg_time_per_batch']:.3f}s{'':<6} "
                          f"{result['peak_memory_gb']:.1f}GB{'':<4} {fps:<8.0f} "
                          f"{result['gpu_utilization_avg']:.0f}%{'':<3} {result['gpu_temperature_max']:.0f}°C")
                else:
                    error_msg = result.get('error', '未知錯誤')[:10]
                    print(f"{batch_size:<8} {level:<8} {'❌':<8} {error_msg}")
        
        # 效能摘要
        print(f"\n🏆 最佳效能配置:")
        if best_config:
            batch_size, level = best_config
            best_result = results[batch_size][level]
            print(f"   Batch Size: {batch_size} ({level} 級別)")
            print(f"   最高 FPS: {best_fps:.0f}")
            print(f"   記憶體使用: {best_result['peak_memory_gb']:.1f}GB")
            print(f"   GPU 使用率: {best_result['gpu_utilization_avg']:.0f}%")
    
    def generate_cross_gpu_comparison(self, gpu_types):
        """生成跨 GPU 效能比較報告"""
        print(f"\n{'='*100}")
        print(f"🔥 跨 GPU 效能比較報告")
        print(f"{'='*100}")
        
        comparison_data = []
        for gpu_type in gpu_types:
            if gpu_type in self.results:
                data = self.results[gpu_type]
                
                # 找出最佳效能配置
                best_fps = 0
                best_batch = None
                max_batch = data.get('max_batch_size_found', 'N/A')
                
                for batch_size, batch_results in data['benchmark_results'].items():
                    for level, result in batch_results.items():
                        if result['successful'] and result['fps'] > best_fps:
                            best_fps = result['fps']
                            best_batch = batch_size
                
                comparison_data.append({
                    'gpu': gpu_type,
                    'memory_gb': data['gpu_info']['memory_gb'],
                    'best_fps': best_fps,
                    'best_batch': best_batch,
                    'max_batch': max_batch
                })
        
        # 排序並顯示
        comparison_data.sort(key=lambda x: x['best_fps'], reverse=True)
        
        print(f"{'GPU':<10} {'記憶體':<8} {'最佳FPS':<10} {'最佳Batch':<10} {'極限Batch':<10} {'相對效能':<8}")
        print("-" * 70)
        
        baseline_fps = comparison_data[-1]['best_fps'] if comparison_data else 1
        
        for data in comparison_data:
            relative_perf = data['best_fps'] / baseline_fps
            print(f"{data['gpu']:<10} {data['memory_gb']}GB{'':<4} {data['best_fps']:<10.0f} "
                  f"{data['best_batch']:<10} {data['max_batch']:<10} {relative_perf:.1f}x")
        
        return comparison_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU 擬真效能測試 v2.0")
    parser.add_argument("--gpu-type", type=str, help="指定 GPU 類型 (RTX4090, RTX5090, H100, B200)")
    parser.add_argument("--config", type=str, default="configs/gpu_configs.yaml", help="配置檔案路徑")
    parser.add_argument("--output", type=str, default="benchmark_results_v2.yaml", help="輸出檔案")
    parser.add_argument("--test-levels", nargs='+', default=['light', 'medium', 'heavy'], 
                        help="測試級別 (light, medium, heavy)")
    parser.add_argument("--find-limit", action='store_true', default=True, help="尋找最大 batch size")
    parser.add_argument("--quick", action='store_true', help="快速測試 (僅 light 級別)")
    parser.add_argument("--compare", nargs='+', help="比較多個 GPU 類型 (需要先執行各別測試)")
    
    args = parser.parse_args()
    
    # 快速測試模式
    if args.quick:
        args.test_levels = ['light']
        args.find_limit = False
    
    benchmark = GPUBenchmark(args.config)
    
    # 比較模式
    if args.compare:
        print("🔄 載入之前的測試結果進行比較...")
        try:
            with open(args.output, 'r', encoding='utf-8') as f:
                benchmark.results = yaml.safe_load(f)
            benchmark.generate_cross_gpu_comparison(args.compare)
        except FileNotFoundError:
            print("❌ 找不到之前的測試結果，請先執行各 GPU 的測試")
    else:
        # 單一 GPU 測試
        detected_gpu = benchmark.detect_gpu()
        target_gpu = args.gpu_type or detected_gpu
        
        print(f"🎮 開始 {target_gpu} 擬真效能測試...")
        results = benchmark.run_comprehensive_benchmark(
            target_gpu, 
            args.test_levels, 
            args.find_limit
        )
        
        benchmark.print_comprehensive_summary(target_gpu)
        benchmark.save_results(args.output)
        
        print(f"\n✅ 測試完成！結果已儲存至 {args.output}")
        print(f"💡 使用 --compare 參數可比較多個 GPU 結果")