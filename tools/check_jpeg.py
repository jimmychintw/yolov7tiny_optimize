# check_jpeg.py
import cv2

info = cv2.getBuildInformation()
for line in info.splitlines():
    if "JPEG" in line:
        print(line.strip())