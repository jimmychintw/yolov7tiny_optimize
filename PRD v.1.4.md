### **Baseline Spec v1.4: YOLOv7-tiny From Scratch (AMP)**





#### **版本歷史**



- **v1.4 (2025-08-14)**:
  - **新增**: 第 2 章「專案與資料集結構指南」，標準化程式碼與資料的分離式存放方式。
  - **釐清**: 詳細說明 `coco.yaml` 中相對路徑的設定方法與注意事項。
- **v1.2 (Corrected)**:
  - 以 v1.1 為基礎，修正超參數定義，明確指定必須使用官方 `data/hyp.scratch.tiny.yaml`。
- **v1.1**:
  - 確立「從頭訓練 (From Scratch)」原則。
- **v1.0**:
  - 建立初始 baseline 所有技術規格。

------



### **1. 專案目標與範圍 (Objective & Scope)**



- **Model**: YOLOv7‑tiny
- **Dataset**: COCO 2017（官方 split），輸入統一 **320×320**（letterbox）
- **Training**: **從隨機權重開始**，**AMP（FP16 混合精度）**，**300 epochs**，不中斷。
- **Quantization**: **PTQ（ONNX Runtime 靜態量化，QDQ）**；各部署平台**一律**由該 INT8 ONNX 再轉。
- **Evaluation**: COCO mAP50‑95（含 S/M/L），固定前處理與 NMS。

------



### **2. 專案與資料集結構指南 (修訂)**





#### **2.1 核心原則**



為確保專案的整潔性、可移植性與可維護性，我們採用**「程式碼」與「資料」分離管理**的模式。所有團隊成員均需遵守此結構。



#### **2.2 標準目錄結構 (修訂)**



專案應在一個最上層的工作區 (Workspace) 中，以**平行方式**存放專案資料夾與資料集資料夾。

```
/workspace/              <-- 工作區根目錄 (註：此資料夾名稱不重要，可自訂)
│
├── yolov7/              <-- 核心專案資料夾 (直接為 yolov7 倉庫)
│   ├── data/
│   │   └── coco.yaml   <-- 關鍵設定檔
│   └── train.py
│
└── coco/                 <-- COCO 資料集，與專案資料夾平行
    ├── images/
    └── labels/
```



#### **2.3 `coco.yaml` 設定注意事項 (修訂)**



此檔案是連接程式碼與資料的橋樑，其設定至關重要。

- **檔案位置**: `yolov7/data/coco.yaml`
- **設定方法**: 必須使用**相對路徑**來指定 `train` 和 `val` 資料夾的位置。

YAML

```
# yolov7/data/coco.yaml
#
# 假設 train.py 是在 /yolov7/ 目錄下執行，
# 我們需要使用 ../ 從 data/ 目錄向上返回一層，再進入 coco/。

train: ../coco/images/train2017/
val: ../coco/images/val2017/

# --- 以下部分保持不變 ---

# number of classes
nc: 80

# class names
names: [ 'person', 'bicycle', 'car', ... ] # (此處省略完整列表)
```



#### **2.4 關於最上層目錄的說明**



最上層的 `workspace` 資料夾**其名稱本身完全不重要**。它只是一個容器，您可以任意命名（例如 `ML_Projects`, `MyWork`）。真正重要的是 `yolov7` 和 `coco` 這兩個資料夾之間的**平行相對關係**。

------



### **3. 訓練規格 (Training Specs)**



- **權重初始化**: **從隨機權重開始訓練 (From Scratch)** (`--weights ''`)。

- **超參數**: **必須使用官方檔案 `data/hyp.scratch.tiny.yaml`** (`--hyp` 參數)。

- **img**: 320；**epochs**: 300。

- **batch**: 依 GPU 能力動態調整。

- **AMP**: 開；**EMA**: 開。

- **模型保存/抽測**: 每 **25 epoch** 保存 `last.pt`，並啟動**背景驗證**。

- **訓練啟動（示例）**:

  Bash

  ```
  # 進入 yolov7 目錄下執行
  python train.py --img 320 --batch 128 --epochs 300 \
    --data data/coco.yaml --weights '' \
    --hyp data/hyp.scratch.tiny.yaml \
    --device 0 --workers 4 --amp --save-period 25
  ```

------



### **4. 導出（ONNX）**



- **opset**: 13

- **shape**: 靜態 1×3×320×320；**simplify**: 開

- **指令**:

  Bash

  ```
  python export.py --weights runs/train/exp/weights/best.pt \
    --img 320 320 --batch 1 --include onnx --opset 13
  python -m onnxsim model.onnx model-sim.onnx
  ```

------



### **5. PTQ（黃金 INT8 模型，ONNX Runtime）**



- **Quant 格式**: QDQ
- **Weights**: INT8，symmetric, per‑channel
- **Activations**: INT8，asymmetric, per‑tensor
- **校正集**: calib.txt（512 張，無增強）

------



### **6. 評測規格（統一）**



- **NMS/閾值**: `conf_thres=0.001`, `iou_thres=0.65`, `max_det=300`
- **指標**: COCO mAP50‑95（含 S/M/L）
- **黃金引擎**: ONNX Runtime **CPU** 的 mAP 作為數值基準

------



### **7. 驗收門檻與預期（Sanity 範圍）**



- **mAP50‑95（AMP/FP16）**: ~ **32–34**
  - (註：此為『從頭訓練』的預估範圍，首次 baseline 執行完成後，應以此結果為準。)
- **INT8 全量化掉點**: **‑1.5 ~ ‑3.0 mAP**
- **跨平台一致性**: 各引擎對 ORT‑CPU 的差 ≤ **0.5 mAP**

------



### **8. 變更管理與公平對照**



- **嚴禁改動**: 資料與 split、前處理、訓練 epoch、AMP 開關、PTQ 食譜、校正集、NMS 參數、**超參數檔案 (`hyp.scratch.tiny.yaml`)**。
- **允許改動**: 模型架構、loss、增強策略（但收尾關閉 Mosaic/MixUp 的節點不變）。