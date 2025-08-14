#!/usr/bin/env python3
"""
實驗管理系統 - 支援超參數實驗與結果追蹤
遵循 PRD 規範，只調整允許的參數
"""

import os
import yaml
import json
import shutil
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys

class ExperimentManager:
    def __init__(self, base_dir="experiments"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.load_base_config()
        
    def load_base_config(self):
        """載入基礎配置"""
        try:
            with open("configs/gpu_configs.yaml", 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print("警告: 找不到 configs/gpu_configs.yaml，使用預設設定")
            self.config = self.get_default_config()
    
    def get_default_config(self):
        """預設配置"""
        return {
            'base_training': {
                'img_size': 320,
                'epochs': 300,
                'data': 'data/coco.yaml',
                'weights': '',
                'hyp': 'data/hyp.scratch.tiny.yaml',
                'device': 0,
                'amp': True,
                'save_period': 25
            }
        }
    
    def create_experiment(self, exp_name, gpu_type, custom_params=None):
        """建立新實驗"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_id = f"{exp_name}_{gpu_type}_{timestamp}"
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(exist_ok=True)
        
        # 建立實驗配置
        exp_config = self.generate_experiment_config(gpu_type, custom_params)
        
        # 儲存實驗配置
        config_file = exp_dir / "experiment_config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(exp_config, f, default_flow_style=False, allow_unicode=True)
        
        # 建立超參數檔案（如果有自定義）
        if custom_params and 'hyperparameters' in custom_params:
            self.create_custom_hyperparameters(exp_dir, custom_params['hyperparameters'])
            exp_config['training_args']['hyp'] = f"experiments/{exp_id}/hyp_custom.yaml"
        
        # 建立執行腳本
        self.create_training_script(exp_dir, exp_config)
        
        # 記錄實驗資訊
        self.log_experiment(exp_id, exp_config)
        
        return exp_id, exp_dir
    
    def generate_experiment_config(self, gpu_type, custom_params=None):
        """生成實驗配置"""
        base_config = self.config['base_training'].copy()
        
        # 根據 GPU 類型調整
        if gpu_type in self.config.get('gpu_configs', {}):
            gpu_config = self.config['gpu_configs'][gpu_type]
            recommended_batch = gpu_config['optimal_batch_sizes'][1]  # 選擇中等 batch size
            base_config['batch'] = recommended_batch
            base_config['workers'] = gpu_config['recommended_workers']
        else:
            base_config['batch'] = 128  # 預設值
            base_config['workers'] = 4
        
        # 應用自定義參數
        if custom_params:
            # 只允許調整 PRD 中允許的參數
            allowed_params = [
                'batch', 'workers', 'optimizer', 'lr_multiplier', 
                'warmup_epochs_override', 'augmentation_strength'
            ]
            
            for param, value in custom_params.items():
                if param in allowed_params:
                    base_config[param] = value
        
        return {
            'experiment_info': {
                'gpu_type': gpu_type,
                'created_at': datetime.now().isoformat(),
                'prd_compliance': True,
                'base_spec': 'PRD v1.4'
            },
            'training_args': base_config
        }
    
    def create_custom_hyperparameters(self, exp_dir, hyp_overrides):
        """建立自定義超參數檔案（基於原始檔案）"""
        # 讀取原始超參數
        with open("data/hyp.scratch.tiny.yaml", 'r') as f:
            base_hyp = yaml.safe_load(f)
        
        # 應用覆寫（只允許特定參數）
        allowed_hyp_params = [
            'lr0', 'lrf', 'momentum', 'weight_decay', 'warmup_epochs',
            'box', 'cls', 'obj', 'hsv_h', 'hsv_s', 'hsv_v',
            'translate', 'scale', 'fliplr', 'mosaic', 'mixup'
        ]
        
        for param, value in hyp_overrides.items():
            if param in allowed_hyp_params:
                base_hyp[param] = value
        
        # 儲存自定義超參數檔案
        hyp_file = exp_dir / "hyp_custom.yaml"
        with open(hyp_file, 'w', encoding='utf-8') as f:
            yaml.dump(base_hyp, f, default_flow_style=False)
        
        return str(hyp_file)
    
    def create_training_script(self, exp_dir, config):
        """建立訓練執行腳本"""
        script_content = f"""#!/bin/bash
# 實驗訓練腳本 - 自動生成
# GPU 類型: {config['experiment_info']['gpu_type']}
# 建立時間: {config['experiment_info']['created_at']}

cd "$(dirname "$0")/../.."

# 設定實驗目錄
EXPERIMENT_DIR="{exp_dir.name}"
PROJECT_DIR="runs/train/$EXPERIMENT_DIR"

# 執行訓練
python train.py \\
  --img {config['training_args']['img_size']} \\
  --batch {config['training_args']['batch']} \\
  --epochs {config['training_args']['epochs']} \\
  --data {config['training_args']['data']} \\
  --weights "{config['training_args']['weights']}" \\
  --hyp {config['training_args']['hyp']} \\
  --device {config['training_args']['device']} \\
  --workers {config['training_args']['workers']} \\
  --project runs/train \\
  --name $EXPERIMENT_DIR \\
  --save-period {config['training_args']['save_period']}"""

        if config['training_args']['amp']:
            script_content += " \\\n  --amp"
        
        script_content += f"""

# 訓練完成後的處理
echo "實驗完成: $EXPERIMENT_DIR"
echo "結果路徑: $PROJECT_DIR"
echo "TensorBoard: tensorboard --logdir runs/train"

# 複製結果到實驗目錄
if [ -d "$PROJECT_DIR" ]; then
    cp -r "$PROJECT_DIR"/* "experiments/$EXPERIMENT_DIR/"
    echo "結果已複製到實驗目錄"
fi
"""
        
        script_file = exp_dir / "run_experiment.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # 設定執行權限
        os.chmod(script_file, 0o755)
        
        return script_file
    
    def log_experiment(self, exp_id, config):
        """記錄實驗到總日誌"""
        log_file = self.base_dir / "experiments_log.json"
        
        # 讀取現有日誌
        if log_file.exists():
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        else:
            log_data = {"experiments": []}
        
        # 添加新實驗
        experiment_entry = {
            "exp_id": exp_id,
            "gpu_type": config['experiment_info']['gpu_type'],
            "created_at": config['experiment_info']['created_at'],
            "batch_size": config['training_args']['batch'],
            "status": "created",
            "config_path": f"experiments/{exp_id}/experiment_config.yaml"
        }
        
        log_data["experiments"].append(experiment_entry)
        
        # 儲存日誌
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    
    def list_experiments(self):
        """列出所有實驗"""
        log_file = self.base_dir / "experiments_log.json"
        if not log_file.exists():
            print("尚無實驗記錄")
            return
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        print("實驗列表:")
        print(f"{'實驗ID':<30} {'GPU類型':<10} {'Batch Size':<12} {'狀態':<10} {'建立時間':<20}")
        print("-" * 85)
        
        for exp in log_data["experiments"]:
            print(f"{exp['exp_id']:<30} {exp['gpu_type']:<10} {exp['batch_size']:<12} {exp['status']:<10} {exp['created_at'][:16]:<20}")
    
    def run_experiment(self, exp_id):
        """執行特定實驗"""
        exp_dir = self.base_dir / exp_id
        script_file = exp_dir / "run_experiment.sh"
        
        if not script_file.exists():
            print(f"找不到實驗腳本: {script_file}")
            return False
        
        print(f"開始執行實驗: {exp_id}")
        try:
            subprocess.run(["bash", str(script_file)], check=True)
            print(f"實驗完成: {exp_id}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"實驗執行失敗: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="YOLOv7-tiny 實驗管理系統")
    subparsers = parser.add_subparsers(dest='command', help='可用指令')
    
    # 建立實驗
    create_parser = subparsers.add_parser('create', help='建立新實驗')
    create_parser.add_argument('--name', required=True, help='實驗名稱')
    create_parser.add_argument('--gpu', required=True, choices=['RTX4090', 'RTX5090', 'H100', 'B200'], help='GPU 類型')
    create_parser.add_argument('--batch', type=int, help='批次大小')
    create_parser.add_argument('--lr-mult', type=float, help='學習率倍數')
    create_parser.add_argument('--warmup', type=int, help='預熱週期')
    
    # 列出實驗
    subparsers.add_parser('list', help='列出所有實驗')
    
    # 執行實驗
    run_parser = subparsers.add_parser('run', help='執行實驗')
    run_parser.add_argument('exp_id', help='實驗ID')
    
    args = parser.parse_args()
    
    manager = ExperimentManager()
    
    if args.command == 'create':
        custom_params = {}
        if args.batch:
            custom_params['batch'] = args.batch
        if args.lr_mult:
            custom_params['hyperparameters'] = {'lr0': 0.01 * args.lr_mult}
        if args.warmup:
            custom_params['hyperparameters'] = custom_params.get('hyperparameters', {})
            custom_params['hyperparameters']['warmup_epochs'] = args.warmup
        
        exp_id, exp_dir = manager.create_experiment(args.name, args.gpu, custom_params)
        print(f"實驗已建立: {exp_id}")
        print(f"實驗目錄: {exp_dir}")
        print(f"執行指令: python tools/experiment_manager.py run {exp_id}")
    
    elif args.command == 'list':
        manager.list_experiments()
    
    elif args.command == 'run':
        manager.run_experiment(args.exp_id)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()