#!/usr/bin/env python3
"""測試 OpenCV 線程和 OpenCL 設定"""

import cv2

print("OpenCV 版本:", cv2.__version__)
print("OpenCV 線程數:", cv2.getNumThreads())
print("OpenCL 啟用狀態:", cv2.ocl.useOpenCL())

# 模擬 train.py 的設定
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

print("\n應用設定後:")
print("OpenCV 線程數:", cv2.getNumThreads())
print("OpenCL 啟用狀態:", cv2.ocl.useOpenCL())