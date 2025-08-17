import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import argparse
from utils.datasets import create_dataloader
from utils.general import increment_path


def main(opt):
    # 建立 dataloader
    dataloader, dataset = create_dataloader(
        path=opt.data,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        stride=32,
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
        "--data", type=str, default="data/coco_vast.ai.yaml", help="dataset yaml path")
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
    opt = parser.parse_args()
    main(opt)
