#!/usr/bin/env python3
"""
YOLOv7 訓練監控器 - 增強版
提供詳細的 GPU、CPU、記憶體、I/O 和訓練狀態監控
"""

import argparse
import subprocess
import time
import sys
import os
import re
import json
import psutil
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import threading
import signal

class ColorCode:
    """終端顏色代碼"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def parse_args():
    """解析參數"""
    parser = argparse.ArgumentParser(description='YOLOv7 Training Monitor - Enhanced')
    
    # 實驗定位參數
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    
    # 監控參數
    parser.add_argument('--refresh', type=float, default=1.0, help='refresh interval in seconds')
    parser.add_argument('--verbose', action='store_true', help='verbose output')
    parser.add_argument('--log-file', type=str, help='training log file path')
    parser.add_argument('--history', type=int, default=60, help='history points to keep')
    
    # 忽略的 train.py 參數（完整對齊）
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    
    return parser.parse_args()

class SystemMonitor:
    """系統資源監控器"""
    
    def __init__(self, history_size=60):
        self.history_size = history_size
        self.gpu_history = deque(maxlen=history_size)
        self.cpu_history = deque(maxlen=history_size)
        self.mem_history = deque(maxlen=history_size)
        self.io_history = deque(maxlen=history_size)
        
    def get_gpu_detailed(self):
        """獲取詳細 GPU 資訊"""
        try:
            # 基本 GPU 資訊
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,memory.reserved,temperature.gpu,power.draw,power.limit,clocks.current.sm,clocks.max.sm,clocks.current.memory,clocks.max.memory,pcie.link.gen.current,pcie.link.width.current,fan.speed',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode != 0:
                return None
                
            values = result.stdout.strip().split(',')
            
            # 進程資訊
            proc_result = subprocess.run([
                'nvidia-smi', 
                'pmon', '-c', '1'
            ], capture_output=True, text=True, timeout=2)
            
            processes = []
            if proc_result.returncode == 0:
                for line in proc_result.stdout.strip().split('\n')[2:]:  # 跳過標題
                    parts = line.split()
                    if len(parts) >= 8:
                        processes.append({
                            'pid': parts[1],
                            'type': parts[2],
                            'sm': parts[3],
                            'mem': parts[4],
                            'enc': parts[5],
                            'dec': parts[6],
                            'command': parts[7] if len(parts) > 7 else 'N/A'
                        })
            
            gpu_info = {
                'index': int(values[0]),
                'name': values[1].strip(),
                'gpu_util': float(values[2]),
                'mem_util': float(values[3]),
                'mem_used': float(values[4]) / 1024,  # GB
                'mem_total': float(values[5]) / 1024,
                'mem_reserved': float(values[6]) / 1024 if values[6] != '-' else 0,
                'temp': float(values[7]),
                'power': float(values[8]),
                'power_limit': float(values[9]),
                'sm_clock': int(values[10]),
                'sm_clock_max': int(values[11]),
                'mem_clock': int(values[12]),
                'mem_clock_max': int(values[13]),
                'pcie_gen': values[14].strip(),
                'pcie_width': values[15].strip(),
                'fan_speed': float(values[16]) if values[16] != '[N/A]' else 0,
                'processes': processes
            }
            
            # 計算效率指標
            gpu_info['clock_efficiency'] = (gpu_info['sm_clock'] / gpu_info['sm_clock_max']) * 100 if gpu_info['sm_clock_max'] > 0 else 0
            gpu_info['power_efficiency'] = (gpu_info['power'] / gpu_info['power_limit']) * 100 if gpu_info['power_limit'] > 0 else 0
            
            return gpu_info
            
        except Exception as e:
            return None
    
    def get_cpu_detailed(self):
        """獲取詳細 CPU 資訊"""
        try:
            # CPU 使用率（每個核心）
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # CPU 頻率
            cpu_freq = psutil.cpu_freq(percpu=True)
            
            # CPU 統計
            cpu_stats = psutil.cpu_stats()
            
            # 載入平均值
            load_avg = os.getloadavg()
            
            # 進程和線程數
            process_count = len(psutil.pids())
            
            # Python 相關進程
            python_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'num_threads']):
                if 'python' in proc.info['name'].lower():
                    python_procs.append(proc.info)
            
            return {
                'percent_per_core': cpu_percent,
                'percent_avg': np.mean(cpu_percent),
                'percent_max': max(cpu_percent),
                'freq_current': [f.current for f in cpu_freq] if cpu_freq else [],
                'freq_max': [f.max for f in cpu_freq] if cpu_freq else [],
                'ctx_switches': cpu_stats.ctx_switches,
                'interrupts': cpu_stats.interrupts,
                'load_1min': load_avg[0],
                'load_5min': load_avg[1],
                'load_15min': load_avg[2],
                'process_count': process_count,
                'python_processes': python_procs
            }
        except Exception as e:
            return None
    
    def get_memory_detailed(self):
        """獲取詳細記憶體資訊"""
        try:
            # 虛擬記憶體
            vm = psutil.virtual_memory()
            
            # 交換記憶體
            swap = psutil.swap_memory()
            
            # 共享記憶體（如果有）
            try:
                shm_result = subprocess.run(['df', '-h', '/dev/shm'], 
                                          capture_output=True, text=True, timeout=1)
                shm_lines = shm_result.stdout.strip().split('\n')
                if len(shm_lines) > 1:
                    shm_parts = shm_lines[1].split()
                    shm_used = shm_parts[2] if len(shm_parts) > 2 else 'N/A'
                    shm_total = shm_parts[1] if len(shm_parts) > 1 else 'N/A'
                else:
                    shm_used = shm_total = 'N/A'
            except:
                shm_used = shm_total = 'N/A'
            
            return {
                'total': vm.total / (1024**3),  # GB
                'available': vm.available / (1024**3),
                'used': vm.used / (1024**3),
                'free': vm.free / (1024**3),
                'percent': vm.percent,
                'cached': vm.cached / (1024**3) if hasattr(vm, 'cached') else 0,
                'buffers': vm.buffers / (1024**3) if hasattr(vm, 'buffers') else 0,
                'swap_total': swap.total / (1024**3),
                'swap_used': swap.used / (1024**3),
                'swap_percent': swap.percent,
                'shm_used': shm_used,
                'shm_total': shm_total
            }
        except Exception as e:
            return None
    
    def get_io_detailed(self):
        """獲取詳細 I/O 資訊"""
        try:
            # 磁碟 I/O
            disk_io = psutil.disk_io_counters()
            
            # 網路 I/O
            net_io = psutil.net_io_counters()
            
            # 磁碟使用率
            disk_usage = {}
            for partition in psutil.disk_partitions():
                if partition.mountpoint in ['/', '/home', '/data']:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = {
                        'total': usage.total / (1024**3),
                        'used': usage.used / (1024**3),
                        'free': usage.free / (1024**3),
                        'percent': usage.percent
                    }
            
            return {
                'disk_read_mb': disk_io.read_bytes / (1024**2),
                'disk_write_mb': disk_io.write_bytes / (1024**2),
                'disk_read_count': disk_io.read_count,
                'disk_write_count': disk_io.write_count,
                'disk_read_time': disk_io.read_time,
                'disk_write_time': disk_io.write_time,
                'net_sent_mb': net_io.bytes_sent / (1024**2),
                'net_recv_mb': net_io.bytes_recv / (1024**2),
                'net_packets_sent': net_io.packets_sent,
                'net_packets_recv': net_io.packets_recv,
                'disk_usage': disk_usage
            }
        except Exception as e:
            return None

class TrainingMonitor:
    """訓練狀態監控器"""
    
    def __init__(self, exp_path, log_file=None):
        self.exp_path = Path(exp_path)
        self.log_file = log_file
        self.iteration_times = deque(maxlen=100)
        self.last_iteration = 0
        self.training_start = None
        
    def parse_results_file(self):
        """解析 results.txt"""
        results_file = self.exp_path / "results.txt"
        
        if not results_file.exists():
            return None
            
        try:
            # 讀取最後幾行以獲取最新狀態
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 2:
                return None
            
            # 解析標題行
            header = lines[0].strip().split()
            
            # 解析最後一行
            last_line = lines[-1].strip().split()
            
            if len(last_line) < len(header):
                return None
            
            # 創建字典
            data = {}
            for i, col in enumerate(header):
                try:
                    data[col] = float(last_line[i])
                except:
                    data[col] = last_line[i]
            
            # 嘗試用 pandas 找最佳結果
            try:
                df = pd.read_csv(results_file, sep=r'\s+', skipinitialspace=True)
                if 'metrics/mAP_0.5:0.95' in df.columns:
                    best_idx = df['metrics/mAP_0.5:0.95'].idxmax()
                    data['best_mAP'] = df.loc[best_idx, 'metrics/mAP_0.5:0.95']
                    data['best_epoch'] = best_idx + 1
            except:
                pass
            
            return data
            
        except Exception as e:
            return None
    
    def parse_training_log(self):
        """解析訓練日誌檔"""
        if not self.log_file:
            # 嘗試自動找日誌
            possible_logs = list(self.exp_path.parent.glob(f"*{self.exp_path.name}*.log"))
            if not possible_logs:
                possible_logs = list(self.exp_path.parent.glob("*.log"))
            
            if possible_logs:
                self.log_file = str(possible_logs[-1])  # 最新的
        
        if not self.log_file or not Path(self.log_file).exists():
            return None
        
        try:
            # 讀取最後 1000 行
            with open(self.log_file, 'r') as f:
                lines = f.readlines()[-1000:]
            
            info = {
                'mixed_precision': False,
                'cache_status': 'unknown',
                'current_lr': None,
                'batch_time': None,
                'data_time': None,
                'img_size': None,
                'augmentation': [],
                'warnings': [],
                'recent_iterations': []
            }
            
            for line in lines:
                # 檢查混合精度
                if 'AMP' in line or 'autocast' in line or 'fp16' in line or 'half' in line:
                    info['mixed_precision'] = True
                
                # 檢查快取狀態
                if 'Caching images' in line:
                    info['cache_status'] = 'caching'
                elif 'Cached' in line or 'cached' in line:
                    info['cache_status'] = 'cached'
                
                # 提取學習率
                lr_match = re.search(r'lr[:\s]+([0-9.e-]+)', line)
                if lr_match:
                    info['current_lr'] = float(lr_match.group(1))
                
                # 提取批次時間
                time_match = re.search(r'(\d+\.?\d*)\s*s/it', line)
                if not time_match:
                    time_match = re.search(r'(\d+\.?\d*)\s*it/s', line)
                    if time_match:
                        info['batch_time'] = 1.0 / float(time_match.group(1))
                else:
                    info['batch_time'] = float(time_match.group(1))
                
                # 提取圖片大小
                size_match = re.search(r'imgsz[:\s]+(\d+)[x,\s]+(\d+)', line)
                if size_match:
                    info['img_size'] = (int(size_match.group(1)), int(size_match.group(2)))
                
                # 檢查增強
                if 'mosaic' in line.lower():
                    info['augmentation'].append('mosaic')
                if 'mixup' in line.lower():
                    info['augmentation'].append('mixup')
                if 'copy_paste' in line.lower():
                    info['augmentation'].append('copy_paste')
                
                # 收集警告
                if 'WARNING' in line or 'Error' in line:
                    info['warnings'].append(line.strip())
                
                # 收集最近的迭代時間
                if 's/it' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if 's/it' in part and i > 0:
                            try:
                                time_val = float(parts[i-1])
                                info['recent_iterations'].append(time_val)
                            except:
                                pass
            
            return info
            
        except Exception as e:
            return None
    
    def check_convergence(self, stats):
        """檢查收斂狀態"""
        if not stats or 'best_epoch' not in stats:
            return None
            
        current_epoch = stats.get('epoch', 0)
        best_epoch = stats.get('best_epoch', 0)
        
        # 判斷標準
        status = {
            'converged': False,
            'overfitting': False,
            'improving': False,
            'stagnant': False
        }
        
        if current_epoch > 20:
            epochs_since_best = current_epoch - best_epoch
            
            if epochs_since_best > 15:
                status['stagnant'] = True
                if epochs_since_best > 25:
                    status['converged'] = True
            elif epochs_since_best < 3:
                status['improving'] = True
            
            # 檢查過擬合
            if 'train/loss' in stats and 'val/loss' in stats:
                train_loss = stats['train/loss']
                val_loss = stats['val/loss']
                if val_loss > train_loss * 1.5:
                    status['overfitting'] = True
        
        return status

class EnhancedMonitor:
    """增強版監控器主類"""
    
    def __init__(self, args):
        self.args = args
        self.exp_path = self.find_experiment_path()
        self.system_monitor = SystemMonitor(args.history)
        self.training_monitor = TrainingMonitor(self.exp_path, args.log_file)
        self.running = True
        self.last_update = time.time()
        
    def find_experiment_path(self):
        """找到實驗路徑"""
        base_path = Path(self.args.project)
        
        if self.args.exist_ok:
            exp_path = base_path / self.args.name
        else:
            exp_path = base_path / self.args.name
            if not exp_path.exists():
                for i in range(2, 100):
                    test_path = base_path / f"{self.args.name}{i}"
                    if test_path.exists():
                        exp_path = test_path
                    else:
                        break
        
        return exp_path
    
    def format_bar(self, value, max_value, width=20, filled='█', empty='░'):
        """格式化進度條"""
        filled_width = int(value / max_value * width) if max_value > 0 else 0
        return filled * filled_width + empty * (width - filled_width)
    
    def format_time(self, seconds):
        """格式化時間"""
        if seconds is None:
            return "N/A"
        return str(timedelta(seconds=int(seconds)))
    
    def format_size(self, bytes_val):
        """格式化大小"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.2f}{unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.2f}PB"
    
    def print_header(self):
        """打印標題"""
        print("\033[2J\033[H")  # 清屏
        print(f"{ColorCode.CYAN}{'='*100}{ColorCode.RESET}")
        print(f"{ColorCode.BOLD} YOLOv7 Training Monitor - Enhanced Edition{ColorCode.RESET}".center(110))
        print(f"{ColorCode.CYAN}{'='*100}{ColorCode.RESET}")
        print(f"📂 Experiment: {ColorCode.GREEN}{self.exp_path}{ColorCode.RESET}")
        print(f"🔄 Refresh: {self.args.refresh}s | ⌛ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{ColorCode.CYAN}{'='*100}{ColorCode.RESET}")
    
    def display_gpu_section(self, gpu_info):
        """顯示 GPU 區塊"""
        if not gpu_info:
            print(f"\n{ColorCode.RED}❌ GPU information unavailable{ColorCode.RESET}")
            return
        
        print(f"\n{ColorCode.YELLOW}{'─'*45} GPU Status {'─'*44}{ColorCode.RESET}")
        
        # GPU 名稱和狀態
        print(f"Device: {ColorCode.BOLD}{gpu_info['name']}{ColorCode.RESET} (GPU {gpu_info['index']})")
        
        # 使用率
        gpu_color = ColorCode.GREEN if gpu_info['gpu_util'] > 70 else ColorCode.YELLOW if gpu_info['gpu_util'] > 30 else ColorCode.RED
        mem_color = ColorCode.GREEN if gpu_info['mem_util'] < 90 else ColorCode.YELLOW if gpu_info['mem_util'] < 95 else ColorCode.RED
        
        gpu_bar = self.format_bar(gpu_info['gpu_util'], 100)
        mem_bar = self.format_bar(gpu_info['mem_util'], 100)
        
        print(f"GPU Util:  {gpu_color}{gpu_info['gpu_util']:5.1f}%{ColorCode.RESET} [{gpu_bar}]")
        print(f"Mem Util:  {mem_color}{gpu_info['mem_util']:5.1f}%{ColorCode.RESET} [{mem_bar}] "
              f"({gpu_info['mem_used']:.1f}/{gpu_info['mem_total']:.1f} GB, Reserved: {gpu_info['mem_reserved']:.1f} GB)")
        
        # 時脈和效率
        clock_color = ColorCode.GREEN if gpu_info['clock_efficiency'] > 80 else ColorCode.YELLOW
        print(f"SM Clock:  {clock_color}{gpu_info['sm_clock']:4d}/{gpu_info['sm_clock_max']:4d} MHz{ColorCode.RESET} "
              f"({gpu_info['clock_efficiency']:.1f}% efficiency)")
        print(f"Mem Clock: {gpu_info['mem_clock']:4d}/{gpu_info['mem_clock_max']:4d} MHz")
        
        # 溫度和功耗
        temp_color = ColorCode.GREEN if gpu_info['temp'] < 70 else ColorCode.YELLOW if gpu_info['temp'] < 80 else ColorCode.RED
        power_color = ColorCode.GREEN if gpu_info['power_efficiency'] < 90 else ColorCode.YELLOW
        
        print(f"Temp:      {temp_color}{gpu_info['temp']:.0f}°C{ColorCode.RESET} | "
              f"Fan: {gpu_info['fan_speed']:.0f}% | "
              f"Power: {power_color}{gpu_info['power']:.1f}/{gpu_info['power_limit']:.0f}W{ColorCode.RESET} "
              f"({gpu_info['power_efficiency']:.1f}%)")
        
        # PCIe 狀態
        print(f"PCIe:      Gen {gpu_info['pcie_gen']} x{gpu_info['pcie_width']}")
        
        # GPU 進程
        if gpu_info['processes']:
            print(f"\nGPU Processes:")
            for proc in gpu_info['processes'][:3]:  # 最多顯示3個
                print(f"  PID {proc['pid']}: SM {proc['sm']}% MEM {proc['mem']}% - {proc['command']}")
    
    def display_cpu_section(self, cpu_info):
        """顯示 CPU 區塊"""
        if not cpu_info:
            return
        
        print(f"\n{ColorCode.YELLOW}{'─'*45} CPU Status {'─'*44}{ColorCode.RESET}")
        
        # CPU 使用率
        avg_color = ColorCode.GREEN if cpu_info['percent_avg'] < 70 else ColorCode.YELLOW if cpu_info['percent_avg'] < 90 else ColorCode.RED
        cpu_bar = self.format_bar(cpu_info['percent_avg'], 100)
        
        print(f"CPU Usage: {avg_color}{cpu_info['percent_avg']:5.1f}%{ColorCode.RESET} [{cpu_bar}] "
              f"(Max: {cpu_info['percent_max']:.1f}%)")
        
        # 核心使用率熱圖
        print("Cores:     ", end='')
        for i, percent in enumerate(cpu_info['percent_per_core']):
            if i % 16 == 0 and i > 0:
                print("\n           ", end='')
            color = ColorCode.RED if percent > 90 else ColorCode.YELLOW if percent > 70 else ColorCode.GREEN if percent > 30 else ColorCode.DIM
            print(f"{color}■{ColorCode.RESET}", end='')
        print()
        
        # 載入平均值
        load_color = ColorCode.GREEN if cpu_info['load_1min'] < psutil.cpu_count() else ColorCode.YELLOW
        print(f"Load Avg:  {load_color}{cpu_info['load_1min']:.2f}, {cpu_info['load_5min']:.2f}, {cpu_info['load_15min']:.2f}{ColorCode.RESET}")
        
        # Python 進程
        if cpu_info['python_processes']:
            total_threads = sum(p.get('num_threads', 0) for p in cpu_info['python_processes'])
            print(f"Python:    {len(cpu_info['python_processes'])} processes, {total_threads} threads total")
    
    def display_memory_section(self, mem_info):
        """顯示記憶體區塊"""
        if not mem_info:
            return
        
        print(f"\n{ColorCode.YELLOW}{'─'*44} Memory Status {'─'*43}{ColorCode.RESET}")
        
        # RAM
        mem_color = ColorCode.GREEN if mem_info['percent'] < 80 else ColorCode.YELLOW if mem_info['percent'] < 90 else ColorCode.RED
        mem_bar = self.format_bar(mem_info['percent'], 100)
        
        print(f"RAM:       {mem_color}{mem_info['percent']:5.1f}%{ColorCode.RESET} [{mem_bar}] "
              f"({mem_info['used']:.1f}/{mem_info['total']:.1f} GB)")
        print(f"Available: {mem_info['available']:.1f} GB | Cached: {mem_info['cached']:.1f} GB")
        
        # Swap
        if mem_info['swap_total'] > 0:
            swap_color = ColorCode.GREEN if mem_info['swap_percent'] < 50 else ColorCode.YELLOW
            print(f"Swap:      {swap_color}{mem_info['swap_percent']:5.1f}%{ColorCode.RESET} "
                  f"({mem_info['swap_used']:.1f}/{mem_info['swap_total']:.1f} GB)")
        
        # 共享記憶體
        if mem_info['shm_used'] != 'N/A':
            print(f"Shared:    {mem_info['shm_used']}/{mem_info['shm_total']}")
    
    def display_io_section(self, io_info):
        """顯示 I/O 區塊"""
        if not io_info:
            return
        
        print(f"\n{ColorCode.YELLOW}{'─'*45} I/O Status {'─'*44}{ColorCode.RESET}")
        
        # 磁碟 I/O
        print(f"Disk Read:  {self.format_size(io_info['disk_read_mb'] * 1024 * 1024)} "
              f"({io_info['disk_read_count']:,} ops)")
        print(f"Disk Write: {self.format_size(io_info['disk_write_mb'] * 1024 * 1024)} "
              f"({io_info['disk_write_count']:,} ops)")
        
        # 網路 I/O
        print(f"Net Sent:   {self.format_size(io_info['net_sent_mb'] * 1024 * 1024)} "
              f"({io_info['net_packets_sent']:,} packets)")
        print(f"Net Recv:   {self.format_size(io_info['net_recv_mb'] * 1024 * 1024)} "
              f"({io_info['net_packets_recv']:,} packets)")
        
        # 磁碟使用率
        if io_info['disk_usage']:
            print("Disk Usage:")
            for mount, usage in io_info['disk_usage'].items():
                color = ColorCode.GREEN if usage['percent'] < 80 else ColorCode.YELLOW if usage['percent'] < 90 else ColorCode.RED
                print(f"  {mount:10s} {color}{usage['percent']:5.1f}%{ColorCode.RESET} "
                      f"({usage['used']:.1f}/{usage['total']:.1f} GB)")
    
    def display_training_section(self, results, log_info):
        """顯示訓練區塊"""
        print(f"\n{ColorCode.YELLOW}{'─'*43} Training Status {'─'*42}{ColorCode.RESET}")
        
        if not results:
            print(f"{ColorCode.DIM}⏳ Waiting for training data...{ColorCode.RESET}")
            print(f"   Checking: {self.exp_path / 'results.txt'}")
            return
        
        # 基本訓練資訊
        epoch = int(results.get('epoch', 0))
        
        # 進度條
        if self.args.epochs > 0:
            progress = epoch / self.args.epochs * 100
            prog_bar = self.format_bar(progress, 100, width=40)
            print(f"Progress:  [{prog_bar}] {epoch}/{self.args.epochs} ({progress:.1f}%)")
        else:
            print(f"Epoch:     {epoch}")
        
        # 訓練配置（從日誌）
        if log_info:
            config_items = []
            if log_info['mixed_precision']:
                config_items.append(f"{ColorCode.GREEN}FP16✓{ColorCode.RESET}")
            else:
                config_items.append(f"{ColorCode.YELLOW}FP32{ColorCode.RESET}")
            
            if log_info['cache_status'] == 'cached':
                config_items.append(f"{ColorCode.GREEN}Cached✓{ColorCode.RESET}")
            elif log_info['cache_status'] == 'caching':
                config_items.append(f"{ColorCode.YELLOW}Caching...{ColorCode.RESET}")
            
            if log_info['img_size']:
                config_items.append(f"Size: {log_info['img_size'][0]}x{log_info['img_size'][1]}")
            
            if log_info['augmentation']:
                config_items.append(f"Aug: {','.join(log_info['augmentation'][:2])}")
            
            if config_items:
                print(f"Config:    {' | '.join(config_items)}")
            
            # 學習率
            if log_info['current_lr']:
                print(f"LR:        {log_info['current_lr']:.2e}")
            
            # 批次時間
            if log_info['batch_time']:
                iter_per_sec = 1.0 / log_info['batch_time'] if log_info['batch_time'] > 0 else 0
                color = ColorCode.GREEN if log_info['batch_time'] < 1 else ColorCode.YELLOW if log_info['batch_time'] < 3 else ColorCode.RED
                print(f"Speed:     {color}{log_info['batch_time']:.2f}s/it{ColorCode.RESET} ({iter_per_sec:.2f} it/s)")
                
                # 預估時間
                if self.args.epochs > 0 and log_info['batch_time']:
                    # 假設每個 epoch 約 5000 iterations (COCO)
                    total_iters = (self.args.epochs - epoch) * 5000
                    eta_seconds = total_iters * log_info['batch_time']
                    print(f"ETA:       {self.format_time(eta_seconds)}")
        
        # 損失
        train_loss = results.get('train/box_loss', 0) + results.get('train/obj_loss', 0) + results.get('train/cls_loss', 0)
        val_loss = results.get('val/box_loss', 0) + results.get('val/obj_loss', 0) + results.get('val/cls_loss', 0)
        
        if train_loss > 0 or val_loss > 0:
            print(f"\nLosses:")
            print(f"  Train:   {train_loss:.4f} (Box: {results.get('train/box_loss', 0):.4f}, "
                  f"Obj: {results.get('train/obj_loss', 0):.4f}, Cls: {results.get('train/cls_loss', 0):.4f})")
            print(f"  Val:     {val_loss:.4f} (Box: {results.get('val/box_loss', 0):.4f}, "
                  f"Obj: {results.get('val/obj_loss', 0):.4f}, Cls: {results.get('val/cls_loss', 0):.4f})")
        
        # 指標
        map50 = results.get('metrics/mAP_0.5', 0)
        map95 = results.get('metrics/mAP_0.5:0.95', 0)
        precision = results.get('metrics/precision', 0)
        recall = results.get('metrics/recall', 0)
        
        print(f"\nMetrics:")
        print(f"  mAP@0.5:     {map50:.4f}")
        print(f"  mAP@0.5:0.95: {map95:.4f}")
        if precision > 0 or recall > 0:
            print(f"  Precision:   {precision:.4f}")
            print(f"  Recall:      {recall:.4f}")
        
        # 最佳結果
        if 'best_mAP' in results:
            improvement = map95 - results['best_mAP']
            if improvement >= 0:
                color = ColorCode.GREEN
                symbol = "↑"
            else:
                color = ColorCode.YELLOW
                symbol = "↓"
            
            print(f"\nBest mAP:  {results['best_mAP']:.4f} @ Epoch {results['best_epoch']} "
                  f"{color}({symbol}{abs(improvement):.4f}){ColorCode.RESET}")
        
        # 收斂狀態
        conv_status = self.training_monitor.check_convergence(results)
        if conv_status:
            status_msgs = []
            if conv_status['improving']:
                status_msgs.append(f"{ColorCode.GREEN}✓ Improving{ColorCode.RESET}")
            if conv_status['stagnant']:
                status_msgs.append(f"{ColorCode.YELLOW}⚠ Stagnant{ColorCode.RESET}")
            if conv_status['converged']:
                status_msgs.append(f"{ColorCode.CYAN}✓ Converged{ColorCode.RESET}")
            if conv_status['overfitting']:
                status_msgs.append(f"{ColorCode.RED}⚠ Overfitting{ColorCode.RESET}")
            
            if status_msgs:
                print(f"Status:    {' | '.join(status_msgs)}")
        
        # 警告
        if log_info and log_info['warnings']:
            print(f"\n{ColorCode.RED}Warnings:{ColorCode.RESET}")
            for warning in log_info['warnings'][-3:]:  # 最多顯示3個
                print(f"  • {warning[:100]}")
    
    def display_bottleneck_analysis(self, gpu_info, cpu_info, mem_info, log_info):
        """顯示瓶頸分析"""
        bottlenecks = []
        suggestions = []
        
        # GPU 瓶頸
        if gpu_info:
            if gpu_info['gpu_util'] < 70:
                bottlenecks.append(f"Low GPU utilization ({gpu_info['gpu_util']:.1f}%)")
                if cpu_info and cpu_info['percent_max'] > 95:
                    suggestions.append("CPU bottleneck - increase --workers")
                else:
                    suggestions.append("Consider larger batch size or image size")
            
            if gpu_info['mem_util'] > 95:
                bottlenecks.append(f"GPU memory nearly full ({gpu_info['mem_util']:.1f}%)")
                suggestions.append("Reduce batch size or enable gradient checkpointing")
            
            if gpu_info['temp'] > 83:
                bottlenecks.append(f"High GPU temperature ({gpu_info['temp']:.0f}°C)")
                suggestions.append("Check cooling or reduce power limit")
        
        # CPU 瓶頸
        if cpu_info:
            if cpu_info['percent_avg'] > 90:
                bottlenecks.append(f"High CPU usage ({cpu_info['percent_avg']:.1f}%)")
                suggestions.append("Data preprocessing bottleneck")
            
            if cpu_info['load_1min'] > psutil.cpu_count():
                bottlenecks.append(f"System overloaded (load: {cpu_info['load_1min']:.2f})")
                suggestions.append("Reduce --workers or other processes")
        
        # 記憶體瓶頸
        if mem_info:
            if mem_info['percent'] > 90:
                bottlenecks.append(f"Low RAM ({mem_info['available']:.1f} GB available)")
                suggestions.append("Close other applications or add swap")
            
            if mem_info['swap_percent'] > 50:
                bottlenecks.append(f"Heavy swap usage ({mem_info['swap_percent']:.1f}%)")
                suggestions.append("Performance degradation - need more RAM")
        
        # 訓練瓶頸
        if log_info:
            if log_info['batch_time'] and log_info['batch_time'] > 3:
                bottlenecks.append(f"Slow iteration ({log_info['batch_time']:.2f}s/it)")
                if not log_info['mixed_precision']:
                    suggestions.append("Enable mixed precision with --fp16")
                if log_info['cache_status'] != 'cached':
                    suggestions.append("Enable --cache-images")
        
        if bottlenecks or suggestions:
            print(f"\n{ColorCode.YELLOW}{'─'*42} Bottleneck Analysis {'─'*39}{ColorCode.RESET}")
            
            if bottlenecks:
                print(f"{ColorCode.RED}Issues:{ColorCode.RESET}")
                for issue in bottlenecks[:5]:
                    print(f"  ⚠ {issue}")
            
            if suggestions:
                print(f"\n{ColorCode.GREEN}Suggestions:{ColorCode.RESET}")
                for suggestion in suggestions[:5]:
                    print(f"  💡 {suggestion}")
    
    def run(self):
        """主執行循環"""
        if not self.exp_path.exists():
            print(f"\n{ColorCode.RED}❌ Experiment path not found: {self.exp_path}{ColorCode.RESET}")
            print("\nPossible reasons:")
            print("1. Training hasn't started yet")
            print("2. Wrong parameters")
            print("3. Check if --exist-ok is needed")
            return
        
        print(f"{ColorCode.CYAN}Starting enhanced monitoring...{ColorCode.RESET}")
        print(f"Press Ctrl+C to stop\n")
        
        try:
            while self.running:
                # 收集所有資料
                gpu_info = self.system_monitor.get_gpu_detailed()
                cpu_info = self.system_monitor.get_cpu_detailed()
                mem_info = self.system_monitor.get_memory_detailed()
                io_info = self.system_monitor.get_io_detailed()
                
                results = self.training_monitor.parse_results_file()
                log_info = self.training_monitor.parse_training_log()
                
                # 顯示
                self.print_header()
                self.display_gpu_section(gpu_info)
                self.display_cpu_section(cpu_info)
                self.display_memory_section(mem_info)
                self.display_io_section(io_info)
                self.display_training_section(results, log_info)
                self.display_bottleneck_analysis(gpu_info, cpu_info, mem_info, log_info)
                
                # 底部
                print(f"\n{ColorCode.CYAN}{'='*100}{ColorCode.RESET}")
                print(f"Update rate: {1000/self.args.refresh:.1f} Hz | "
                      f"History: {self.args.history} points | "
                      f"Log: {self.log_file if self.log_file else 'Auto-detect'}")
                
                # 等待
                time.sleep(self.args.refresh)
                
        except KeyboardInterrupt:
            print(f"\n\n{ColorCode.YELLOW}✋ Monitoring stopped by user{ColorCode.RESET}")
        except Exception as e:
            print(f"\n{ColorCode.RED}❌ Error: {e}{ColorCode.RESET}")
            if self.args.verbose:
                import traceback
                traceback.print_exc()

def main():
    args = parse_args()
    monitor = EnhancedMonitor(args)
    monitor.run()

if __name__ == "__main__":
    main()