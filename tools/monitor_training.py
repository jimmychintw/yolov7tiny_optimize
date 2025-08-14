#!/usr/bin/env python3
"""
訓練監控與效能分析工具
即時監控 GPU 使用率、記憶體、訓練速度等指標
"""

import time
import psutil
import GPUtil
import json
import csv
import argparse
from datetime import datetime
from pathlib import Path
import threading
import signal
import sys

class TrainingMonitor:
    def __init__(self, exp_name, log_interval=5):
        self.exp_name = exp_name
        self.log_interval = log_interval
        self.running = False
        self.metrics = []
        
        # 建立日誌目錄
        self.log_dir = Path("experiments") / exp_name / "monitoring"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 日誌檔案
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = self.log_dir / f"metrics_{timestamp}.csv"
        self.summary_file = self.log_dir / f"summary_{timestamp}.json"
        
        # 初始化 CSV 檔案
        self.init_csv()
        
    def init_csv(self):
        """初始化 CSV 檔案標題"""
        headers = [
            'timestamp', 'elapsed_time', 
            'gpu_util', 'gpu_memory_used', 'gpu_memory_total', 'gpu_temp',
            'cpu_percent', 'ram_used_gb', 'ram_total_gb',
            'disk_io_read_mb', 'disk_io_write_mb'
        ]
        
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def get_gpu_metrics(self):
        """取得 GPU 指標"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # 使用第一張 GPU
                return {
                    'util': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                }
            else:
                return {
                    'util': 0,
                    'memory_used': 0,
                    'memory_total': 0,
                    'temperature': 0
                }
        except Exception as e:
            print(f"GPU 指標取得錯誤: {e}")
            return {
                'util': 0,
                'memory_used': 0,
                'memory_total': 0,
                'temperature': 0
            }
    
    def get_system_metrics(self):
        """取得系統指標"""
        # CPU 使用率
        cpu_percent = psutil.cpu_percent()
        
        # 記憶體使用
        memory = psutil.virtual_memory()
        ram_used = memory.used / (1024**3)  # GB
        ram_total = memory.total / (1024**3)  # GB
        
        # 磁碟 I/O
        disk_io = psutil.disk_io_counters()
        disk_read = disk_io.read_bytes / (1024**2) if disk_io else 0  # MB
        disk_write = disk_io.write_bytes / (1024**2) if disk_io else 0  # MB
        
        return {
            'cpu_percent': cpu_percent,
            'ram_used': ram_used,
            'ram_total': ram_total,
            'disk_read': disk_read,
            'disk_write': disk_write
        }
    
    def collect_metrics(self):
        """收集所有指標"""
        gpu_metrics = self.get_gpu_metrics()
        system_metrics = self.get_system_metrics()
        
        current_time = datetime.now()
        elapsed_time = (current_time - self.start_time).total_seconds()
        
        metrics = {
            'timestamp': current_time.isoformat(),
            'elapsed_time': elapsed_time,
            'gpu_util': gpu_metrics['util'],
            'gpu_memory_used': gpu_metrics['memory_used'],
            'gpu_memory_total': gpu_metrics['memory_total'],
            'gpu_temp': gpu_metrics['temperature'],
            'cpu_percent': system_metrics['cpu_percent'],
            'ram_used_gb': system_metrics['ram_used'],
            'ram_total_gb': system_metrics['ram_total'],
            'disk_io_read_mb': system_metrics['disk_read'],
            'disk_io_write_mb': system_metrics['disk_write']
        }
        
        return metrics
    
    def log_metrics(self, metrics):
        """記錄指標到檔案"""
        self.metrics.append(metrics)
        
        # 寫入 CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            row = [
                metrics['timestamp'], metrics['elapsed_time'],
                metrics['gpu_util'], metrics['gpu_memory_used'], 
                metrics['gpu_memory_total'], metrics['gpu_temp'],
                metrics['cpu_percent'], metrics['ram_used_gb'], 
                metrics['ram_total_gb'], metrics['disk_io_read_mb'], 
                metrics['disk_io_write_mb']
            ]
            writer.writerow(row)
    
    def print_metrics(self, metrics):
        """列印當前指標"""
        elapsed_min = metrics['elapsed_time'] / 60
        gpu_mem_percent = (metrics['gpu_memory_used'] / metrics['gpu_memory_total'] * 100) if metrics['gpu_memory_total'] > 0 else 0
        
        print(f"\\r[{elapsed_min:6.1f}min] GPU: {metrics['gpu_util']:5.1f}% "
              f"VRAM: {gpu_mem_percent:5.1f}% ({metrics['gpu_memory_used']:.0f}MB) "
              f"Temp: {metrics['gpu_temp']:4.1f}°C CPU: {metrics['cpu_percent']:5.1f}% "
              f"RAM: {metrics['ram_used_gb']:.1f}GB", end='', flush=True)
    
    def monitoring_loop(self):
        """監控主迴圈"""
        self.start_time = datetime.now()
        
        while self.running:
            try:
                metrics = self.collect_metrics()
                self.log_metrics(metrics)
                self.print_metrics(metrics)
                time.sleep(self.log_interval)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\\n監控錯誤: {e}")
                time.sleep(self.log_interval)
    
    def start(self):
        """開始監控"""
        self.running = True
        print(f"開始監控實驗: {self.exp_name}")
        print(f"日誌檔案: {self.metrics_file}")
        print("按 Ctrl+C 停止監控\\n")
        
        try:
            self.monitoring_loop()
        finally:
            self.stop()
    
    def stop(self):
        """停止監控"""
        self.running = False
        self.generate_summary()
        print(f"\\n監控已停止，摘要已儲存至: {self.summary_file}")
    
    def generate_summary(self):
        """生成摘要統計"""
        if not self.metrics:
            return
        
        gpu_utils = [m['gpu_util'] for m in self.metrics]
        gpu_memories = [m['gpu_memory_used'] for m in self.metrics]
        cpu_utils = [m['cpu_percent'] for m in self.metrics]
        temperatures = [m['gpu_temp'] for m in self.metrics if m['gpu_temp'] > 0]
        
        summary = {
            'experiment': self.exp_name,
            'monitoring_duration_minutes': self.metrics[-1]['elapsed_time'] / 60,
            'total_samples': len(self.metrics),
            'gpu_utilization': {
                'avg': sum(gpu_utils) / len(gpu_utils),
                'max': max(gpu_utils),
                'min': min(gpu_utils)
            },
            'gpu_memory_mb': {
                'avg': sum(gpu_memories) / len(gpu_memories),
                'max': max(gpu_memories),
                'min': min(gpu_memories),
                'total': self.metrics[0]['gpu_memory_total'] if self.metrics else 0
            },
            'cpu_utilization': {
                'avg': sum(cpu_utils) / len(cpu_utils),
                'max': max(cpu_utils),
                'min': min(cpu_utils)
            },
            'temperature_celsius': {
                'avg': sum(temperatures) / len(temperatures) if temperatures else 0,
                'max': max(temperatures) if temperatures else 0,
                'min': min(temperatures) if temperatures else 0
            },
            'files': {
                'metrics_csv': str(self.metrics_file),
                'summary_json': str(self.summary_file)
            }
        }
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # 列印摘要
        print(f"\\n=== 監控摘要 ===")
        print(f"實驗: {summary['experiment']}")
        print(f"監控時間: {summary['monitoring_duration_minutes']:.1f} 分鐘")
        print(f"GPU 使用率: 平均 {summary['gpu_utilization']['avg']:.1f}% (最高 {summary['gpu_utilization']['max']:.1f}%)")
        print(f"GPU 記憶體: 平均 {summary['gpu_memory_mb']['avg']:.0f}MB (最高 {summary['gpu_memory_mb']['max']:.0f}MB)")
        print(f"CPU 使用率: 平均 {summary['cpu_utilization']['avg']:.1f}%")
        if temperatures:
            print(f"GPU 溫度: 平均 {summary['temperature_celsius']['avg']:.1f}°C (最高 {summary['temperature_celsius']['max']:.1f}°C)")

def signal_handler(signum, frame):
    """處理中斷信號"""
    print("\\n收到停止信號...")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="訓練監控工具")
    parser.add_argument('--exp-name', required=True, help='實驗名稱')
    parser.add_argument('--interval', type=int, default=5, help='監控間隔（秒）')
    
    args = parser.parse_args()
    
    # 設定信號處理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    monitor = TrainingMonitor(args.exp_name, args.interval)
    monitor.start()

if __name__ == "__main__":
    main()