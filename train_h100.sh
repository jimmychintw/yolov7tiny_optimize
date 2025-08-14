#!/bin/bash
# H100 訓練腳本 - YOLOv7-tiny Baseline v1.0
# 針對 H100 雙卡系統優化的訓練配置

echo "🚀 啟動 YOLOv7-tiny H100 訓練..."
echo "📊 配置: 320×320, Batch 512, 300 epochs, 雙 GPU, AMP"
echo "=================================="

python train.py \
  --img 320 \
  --batch 512 \
  --epochs 300 \
  --data data/coco.yaml \
  --weights '' \
  --hyp data/hyp.scratch.tiny.yaml \
  --device 0,1 \
  --workers 16 \
  --amp \
  --save-period 25

echo "=================================="
echo "✅ 訓練完成！"