# YOLOv7-tiny 實驗系統使用指南

本系統提供完整的多 GPU 實驗管理、效能測試和結果比較功能，遵循 PRD v1.4 規範。

## 🚀 快速開始

### 1. GPU 效能測試
首先測試您的 GPU 效能，找出最佳 batch size：

```bash
# 自動偵測 GPU 並測試
python tools/gpu_benchmark.py

# 指定 GPU 類型測試
python tools/gpu_benchmark.py --gpu-type H100
```

### 2. 建立實驗
根據測試結果建立實驗：

```bash
# 基本實驗（使用預設參數）
python tools/experiment_manager.py create --name baseline_test --gpu H100

# 自定義 batch size
python tools/experiment_manager.py create --name large_batch --gpu H100 --batch 512

# 調整學習率
python tools/experiment_manager.py create --name high_lr --gpu H100 --lr-mult 1.5

# 調整預熱週期
python tools/experiment_manager.py create --name long_warmup --gpu H100 --warmup 10
```

### 3. 執行實驗
```bash
# 列出所有實驗
python tools/experiment_manager.py list

# 執行特定實驗
python tools/experiment_manager.py run baseline_test_H100_20250814_120000
```

### 4. 監控訓練（另開終端）
```bash
# 監控特定實驗
python tools/monitor_training.py --exp-name baseline_test_H100_20250814_120000

# 自定義監控間隔
python tools/monitor_training.py --exp-name your_exp_id --interval 10
```

### 5. 比較結果
```bash
# 列印比較摘要
python tools/compare_results.py

# 匯出詳細報告
python tools/compare_results.py --export-excel --plot
```

## 📁 專案結構

```
yolov7tiny_baseline/
├── configs/
│   └── gpu_configs.yaml          # GPU 配置檔案
├── tools/
│   ├── gpu_benchmark.py          # GPU 效能測試
│   ├── experiment_manager.py     # 實驗管理系統
│   ├── monitor_training.py       # 訓練監控工具
│   └── compare_results.py        # 結果比較工具
├── experiments/                  # 實驗結果目錄
│   ├── experiments_log.json      # 實驗日誌
│   └── [exp_id]/                 # 個別實驗目錄
│       ├── experiment_config.yaml
│       ├── run_experiment.sh
│       └── monitoring/
└── runs/train/                   # 訓練輸出目錄
```

## 🔧 支援的 GPU 配置

| GPU | 記憶體 | 建議 Batch Size | 建議 Workers |
|-----|--------|-----------------|--------------|
| RTX 4090 | 24GB | 64, 128, 192, 256 | 8 |
| RTX 5090 | 32GB | 128, 192, 256, 320 | 12 |
| H100 | 80GB | 256, 384, 512, 640 | 16 |
| B200 | 192GB | 512, 768, 1024, 1280 | 24 |

## ⚙️ 可調整的超參數

根據 PRD v1.4 規範，以下參數可以調整：

### 訓練參數
- `batch_size`: 批次大小
- `workers`: 資料載入程序數
- `optimizer`: 優化器類型（SGD, Adam, AdamW）

### 學習率相關
- `lr0`: 初始學習率（透過倍數調整）
- `warmup_epochs`: 預熱週期

### 資料增強（在允許範圍內）
- `hsv_h/s/v`: HSV 色彩增強
- `translate`: 平移增強
- `scale`: 縮放增強
- `fliplr`: 水平翻轉機率
- `mosaic`: Mosaic 增強機率
- `mixup`: MixUp 增強機率

## 📊 監控指標

### GPU 指標
- GPU 使用率
- GPU 記憶體使用量
- GPU 溫度

### 系統指標
- CPU 使用率
- RAM 使用量
- 磁碟 I/O

### 訓練指標
- mAP@0.5
- mAP@0.5:0.95
- 訓練時間
- 訓練速度（FPS）

## 📈 TensorBoard 整合

每個實驗都會自動記錄 TensorBoard 日誌：

```bash
# 檢視特定實驗的 TensorBoard
tensorboard --logdir runs/train/your_exp_id

# 檢視所有實驗
tensorboard --logdir runs/train
```

## 🎯 實驗最佳實踐

### 1. 效能基準測試
```bash
# 先測試 GPU 效能
python tools/gpu_benchmark.py --gpu-type H100

# 根據結果選擇最佳 batch size
python tools/experiment_manager.py create --name perf_test --gpu H100 --batch 512
```

### 2. 學習率調優
```bash
# 測試不同學習率
python tools/experiment_manager.py create --name lr_05x --gpu H100 --lr-mult 0.5
python tools/experiment_manager.py create --name lr_10x --gpu H100 --lr-mult 1.0
python tools/experiment_manager.py create --name lr_15x --gpu H100 --lr-mult 1.5
python tools/experiment_manager.py create --name lr_20x --gpu H100 --lr-mult 2.0
```

### 3. Batch Size 影響分析
```bash
# 測試不同 batch size 對準確度的影響
for batch in 256 384 512 640; do
    python tools/experiment_manager.py create --name batch_$batch --gpu H100 --batch $batch
done
```

### 4. 完整實驗流程
```bash
# 1. 效能測試
python tools/gpu_benchmark.py

# 2. 建立實驗
python tools/experiment_manager.py create --name production_run --gpu H100 --batch 384

# 3. 開始監控（背景執行）
python tools/monitor_training.py --exp-name production_run_H100_xxx &

# 4. 執行訓練
python tools/experiment_manager.py run production_run_H100_xxx

# 5. 比較結果
python tools/compare_results.py --export-excel --plot
```

## 🔍 故障排除

### 記憶體不足
```bash
# 重新測試 GPU 找出適合的 batch size
python tools/gpu_benchmark.py --gpu-type your_gpu
```

### 訓練速度慢
- 檢查 workers 數量是否適當
- 確認 GPU 使用率是否充分
- 檢查是否啟用 AMP

### 監控問題
確認安裝了必要套件：
```bash
pip install GPUtil psutil matplotlib seaborn
```

## 📝 注意事項

1. **PRD 合規性**: 所有實驗都遵循 PRD v1.4 規範
2. **資料完整性**: 確保 COCO 資料集正確放置
3. **資源管理**: 大 batch size 實驗需要充足的 GPU 記憶體
4. **結果備份**: 重要實驗建議備份整個 experiments 目錄

## 🤝 貢獻

如需新增功能或修正問題，請確保：
- 遵循 PRD v1.4 規範
- 保持程式碼結構清晰
- 更新相關文件