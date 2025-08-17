import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
import yaml
from pathlib import Path
from utils.datasets import create_dataloader
from utils.general import increment_path


def detect_data_config():
    """自動偵測正確的資料集配置檔案"""
    configs = {
        '/workspace/coco/': 'data/coco_vast.ai.yaml',
        '/root/coco/': 'data/coco_runpod.yaml', 
        '/Volumes/': 'data/coco_mac.yaml',
    }
    
    for path_prefix, config_file in configs.items():
        if os.path.exists(path_prefix):
            print(f"偵測到資料路徑: {path_prefix}")
            print(f"使用配置檔案: {config_file}")
            return config_file
    
    # 預設值
    return 'data/coco.yaml'


def main(opt):
    # 建立 dataloader
    dataloader, dataset = create_dataloader(
        path=opt.data,
        imgsz=opt.img_size,
        batch_size=opt.batch_size,
        stride=32,
        opt=opt,
        hyp=None,
        augment=True,
        cache=opt.cache,
        rect=False,
        rank=-1,
        world_size=1,
        workers=opt.workers
    )

    print(f"Dataset loaded: {len(dataset)} images")
    print(
        f"Testing {opt.num_batches} batches with batch size {opt.batch_size}...")

    # 計時
    t0 = time.time()
    for i, (imgs, targets, paths, _) in enumerate(dataloader):
        if i >= opt.num_batches:
            break
    dt = time.time() - t0

    print(f"\n=== Benchmark Results ===")
    print(f"Total time: {dt:.2f}s for {opt.num_batches} batches")
    print(f"Avg per batch: {dt/opt.num_batches:.3f}s")
    print(f"Throughput: {opt.batch_size/(dt/opt.num_batches):.1f} images/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default=None, help="dataset yaml path (auto-detect if not specified)")
    parser.add_argument("--img-size", type=int, default=320,
                        help="inference size (pixels)")
    parser.add_argument("--batch-size", type=int,
                        default=256, help="batch size")
    parser.add_argument("--workers", type=int, default=8,
                        help="number of dataloader workers")
    parser.add_argument("--num-batches", type=int, default=50,
                        help="number of batches to test")
    parser.add_argument("--cache", action="store_true",
                        help="use --cache to test RAM cached dataset")
    parser.add_argument("--single-cls", action="store_true",
                        help="treat as single-class dataset")
    opt = parser.parse_args()
    
    # 自動偵測資料配置檔案
    if opt.data is None:
        opt.data = detect_data_config()
    
    main(opt)
