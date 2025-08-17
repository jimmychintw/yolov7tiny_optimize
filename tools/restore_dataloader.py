#!/usr/bin/env python3
"""恢復 DataLoader 設定的快速腳本"""

import sys
import os

# 讀取原始檔案
datasets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'utils', 'datasets.py')

with open(datasets_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 恢復原始設定
original_loader_line = "    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader"
original_comment = "    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()"
original_debug = '    loader_type = "InfiniteDataLoader" if not image_weights else "DataLoader"'

# 替換回原始設定
content = content.replace(
    "    # A/B 測試：暫時改用官方 DataLoader 觀察 worker 使用情況\n    loader = torch.utils.data.DataLoader\n    # 原始設定：loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader\n    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()",
    original_loader_line + "\n" + original_comment
)

content = content.replace(
    '    loader_type = "DataLoader (A/B測試)"  # 暫時固定為官方 DataLoader',
    original_debug
)

# 寫回檔案
with open(datasets_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("已恢復 InfiniteDataLoader 原始設定")