#!/usr/bin/env python3
"""
實驗結果比較工具
比較不同實驗的效能與準確度結果
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import yaml
import re
from datetime import datetime

class ResultComparator:
    def __init__(self, experiments_dir="experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.results = {}
        plt.style.use('seaborn-v0_8')
        
    def load_experiment_results(self, exp_id):
        """載入單個實驗結果"""
        exp_dir = self.experiments_dir / exp_id
        
        if not exp_dir.exists():
            print(f"找不到實驗目錄: {exp_dir}")
            return None
        
        result = {
            'exp_id': exp_id,
            'config': self.load_experiment_config(exp_dir),
            'training_metrics': self.load_training_metrics(exp_dir),
            'monitoring_summary': self.load_monitoring_summary(exp_dir),
            'final_results': self.load_final_results(exp_dir)
        }
        
        return result
    
    def load_experiment_config(self, exp_dir):
        """載入實驗配置"""
        config_file = exp_dir / "experiment_config.yaml"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return None
    
    def load_training_metrics(self, exp_dir):
        """載入訓練指標（從 results.txt 或類似檔案）"""
        # 尋找可能的結果檔案
        result_files = [
            exp_dir / "results.txt",
            exp_dir / "results.csv",
        ]
        
        for file_path in result_files:
            if file_path.exists():
                return self.parse_training_results(file_path)
        
        return None
    
    def parse_training_results(self, file_path):
        """解析訓練結果檔案"""
        metrics = {}
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # 使用正則表達式提取關鍵指標
            patterns = {
                'final_map50': r'mAP@0.5:\\s*(\\d+\\.\\d+)',
                'final_map50_95': r'mAP@0.5:0.95:\\s*(\\d+\\.\\d+)',
                'best_fitness': r'Best fitness:\\s*(\\d+\\.\\d+)',
                'training_time': r'Training time:\\s*([\\d.]+)\\s*hours?',
                'epochs_completed': r'Epoch\\s+(\\d+)/'
            }
            
            for metric, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    if metric == 'epochs_completed':
                        metrics[metric] = int(match.group(1))
                    else:
                        metrics[metric] = float(match.group(1))
            
        except Exception as e:
            print(f"解析結果檔案錯誤: {e}")
        
        return metrics
    
    def load_monitoring_summary(self, exp_dir):
        """載入監控摘要"""
        monitoring_dir = exp_dir / "monitoring"
        if not monitoring_dir.exists():
            return None
        
        # 尋找最新的摘要檔案
        summary_files = list(monitoring_dir.glob("summary_*.json"))
        if not summary_files:
            return None
        
        latest_summary = max(summary_files, key=lambda x: x.stat().st_mtime)
        
        with open(latest_summary, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_final_results(self, exp_dir):
        """載入最終評測結果"""
        # 尋找 best.pt 相關的評測結果
        weight_files = list(exp_dir.glob("**/best.pt"))
        if weight_files:
            # 假設有對應的評測結果檔案
            return {"model_path": str(weight_files[0])}
        return None
    
    def load_all_experiments(self):
        """載入所有實驗結果"""
        log_file = self.experiments_dir / "experiments_log.json"
        if not log_file.exists():
            print("找不到實驗日誌檔案")
            return
        
        with open(log_file, 'r') as f:
            log_data = json.load(f)
        
        for exp in log_data["experiments"]:
            exp_id = exp["exp_id"]
            result = self.load_experiment_results(exp_id)
            if result:
                self.results[exp_id] = result
    
    def create_comparison_table(self):
        """建立比較表格"""
        if not self.results:
            print("沒有實驗結果可比較")
            return None
        
        comparison_data = []
        
        for exp_id, result in self.results.items():
            config = result.get('config', {})
            training_args = config.get('training_args', {})
            training_metrics = result.get('training_metrics', {})
            monitoring = result.get('monitoring_summary', {})
            
            row = {
                '實驗ID': exp_id[:20] + '...' if len(exp_id) > 20 else exp_id,
                'GPU類型': config.get('experiment_info', {}).get('gpu_type', 'Unknown'),
                'Batch Size': training_args.get('batch', 'N/A'),
                'Workers': training_args.get('workers', 'N/A'),
                'mAP@0.5': training_metrics.get('final_map50', 'N/A'),
                'mAP@0.5:0.95': training_metrics.get('final_map50_95', 'N/A'),
                '訓練時間(h)': training_metrics.get('training_time', 'N/A'),
                'GPU使用率(%)': monitoring.get('gpu_utilization', {}).get('avg', 'N/A'),
                'GPU記憶體(MB)': monitoring.get('gpu_memory_mb', {}).get('avg', 'N/A'),
                '狀態': '完成' if training_metrics else '進行中'
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def plot_performance_comparison(self, save_path=None):
        """繪製效能比較圖"""
        df = self.create_comparison_table()
        if df is None or df.empty:
            return
        
        # 過濾出有完整資料的實驗
        complete_df = df[df['狀態'] == '完成'].copy()
        
        if complete_df.empty:
            print("沒有完成的實驗可繪圖")
            return
        
        # 轉換數值欄位
        numeric_cols = ['mAP@0.5', 'mAP@0.5:0.95', '訓練時間(h)', 'GPU使用率(%)', 'GPU記憶體(MB)']
        for col in numeric_cols:
            if col in complete_df.columns:
                complete_df[col] = pd.to_numeric(complete_df[col], errors='coerce')
        
        # 建立子圖
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('實驗效能比較', fontsize=16)
        
        # mAP 比較
        if 'mAP@0.5:0.95' in complete_df.columns and complete_df['mAP@0.5:0.95'].notna().any():
            axes[0,0].bar(range(len(complete_df)), complete_df['mAP@0.5:0.95'])
            axes[0,0].set_title('mAP@0.5:0.95 比較')
            axes[0,0].set_ylabel('mAP')
            axes[0,0].set_xticks(range(len(complete_df)))
            axes[0,0].set_xticklabels(complete_df['GPU類型'], rotation=45)
        
        # 訓練時間比較
        if '訓練時間(h)' in complete_df.columns and complete_df['訓練時間(h)'].notna().any():
            axes[0,1].bar(range(len(complete_df)), complete_df['訓練時間(h)'])
            axes[0,1].set_title('訓練時間比較')
            axes[0,1].set_ylabel('小時')
            axes[0,1].set_xticks(range(len(complete_df)))
            axes[0,1].set_xticklabels(complete_df['GPU類型'], rotation=45)
        
        # GPU 使用率比較
        if 'GPU使用率(%)' in complete_df.columns and complete_df['GPU使用率(%)'].notna().any():
            axes[1,0].bar(range(len(complete_df)), complete_df['GPU使用率(%)'])
            axes[1,0].set_title('GPU 使用率比較')
            axes[1,0].set_ylabel('使用率 (%)')
            axes[1,0].set_xticks(range(len(complete_df)))
            axes[1,0].set_xticklabels(complete_df['GPU類型'], rotation=45)
        
        # Batch Size vs mAP 散點圖
        if all(col in complete_df.columns for col in ['Batch Size', 'mAP@0.5:0.95']):
            batch_sizes = pd.to_numeric(complete_df['Batch Size'], errors='coerce')
            map_values = complete_df['mAP@0.5:0.95']
            
            axes[1,1].scatter(batch_sizes, map_values, c=range(len(complete_df)), cmap='viridis')
            axes[1,1].set_title('Batch Size vs mAP')
            axes[1,1].set_xlabel('Batch Size')
            axes[1,1].set_ylabel('mAP@0.5:0.95')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"圖表已儲存至: {save_path}")
        else:
            plt.show()
    
    def export_comparison_report(self, output_file="comparison_report.xlsx"):
        """匯出比較報告到 Excel"""
        df = self.create_comparison_table()
        if df is None:
            return
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 主要比較表
            df.to_excel(writer, sheet_name='實驗比較', index=False)
            
            # 詳細統計
            if not df.empty:
                numeric_cols = ['Batch Size', 'mAP@0.5', 'mAP@0.5:0.95', '訓練時間(h)', 'GPU使用率(%)', 'GPU記憶體(MB)']
                stats_data = []
                
                for col in numeric_cols:
                    if col in df.columns:
                        numeric_data = pd.to_numeric(df[col], errors='coerce').dropna()
                        if not numeric_data.empty:
                            stats_data.append({
                                '指標': col,
                                '平均值': numeric_data.mean(),
                                '最大值': numeric_data.max(),
                                '最小值': numeric_data.min(),
                                '標準差': numeric_data.std()
                            })
                
                if stats_data:
                    stats_df = pd.DataFrame(stats_data)
                    stats_df.to_excel(writer, sheet_name='統計摘要', index=False)
        
        print(f"比較報告已匯出至: {output_file}")
    
    def print_summary(self):
        """列印摘要資訊"""
        df = self.create_comparison_table()
        if df is None:
            return
        
        print("\\n=== 實驗比較摘要 ===")
        print(f"總實驗數: {len(df)}")
        print(f"完成實驗數: {len(df[df['狀態'] == '完成'])}")
        
        if len(df) > 0:
            print("\\n實驗列表:")
            print(df.to_string(index=False, max_colwidth=20))
            
            # 最佳結果
            complete_df = df[df['狀態'] == '完成']
            if not complete_df.empty and 'mAP@0.5:0.95' in complete_df.columns:
                map_values = pd.to_numeric(complete_df['mAP@0.5:0.95'], errors='coerce')
                if not map_values.isna().all():
                    best_idx = map_values.idxmax()
                    print(f"\\n最佳 mAP@0.5:0.95: {map_values.iloc[best_idx]:.3f} ({complete_df.iloc[best_idx]['實驗ID']})")

def main():
    parser = argparse.ArgumentParser(description="實驗結果比較工具")
    parser.add_argument('--experiments-dir', default='experiments', help='實驗目錄路徑')
    parser.add_argument('--export-excel', action='store_true', help='匯出 Excel 報告')
    parser.add_argument('--plot', action='store_true', help='繪製比較圖表')
    parser.add_argument('--output', default='comparison_report', help='輸出檔案前綴')
    
    args = parser.parse_args()
    
    comparator = ResultComparator(args.experiments_dir)
    comparator.load_all_experiments()
    
    # 列印摘要
    comparator.print_summary()
    
    # 匯出 Excel 報告
    if args.export_excel:
        excel_file = f"{args.output}.xlsx"
        comparator.export_comparison_report(excel_file)
    
    # 繪製圖表
    if args.plot:
        plot_file = f"{args.output}_plots.png"
        comparator.plot_performance_comparison(plot_file)

if __name__ == "__main__":
    main()