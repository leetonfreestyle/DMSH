"""把 COCO 图像预先缩放裁剪成小尺寸 JPEG 缓存到本地盘。

原始图像在 F: 盘，实测读取带宽只有约 6 MB/s，是训练与评估的瓶颈。
一次性把图像缩到 img_size 见方（默认 128）写到本地盘后，
后续每个 epoch 的解码量下降一个数量级，且读取走本地 SSD。

支持断点续跑：目标文件已存在则跳过。
"""
import argparse
import os
import time
from multiprocessing import Pool

from PIL import Image


def _convert(job):
    src, dst, size, quality = job
    if os.path.exists(dst):
        return 1
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            scale = size / min(w, h)
            nw, nh = max(size, int(round(w * scale))), max(size, int(round(h * scale)))
            im = im.resize((nw, nh), Image.BICUBIC)
            left, top = (nw - size) // 2, (nh - size) // 2
            im = im.crop((left, top, left + size, top + size))
            im.save(dst, "JPEG", quality=quality, optimize=True)
        return 1
    except Exception as e:  # 单张失败不应中断整体构建
        print(f"[skip] {src}: {e}", flush=True)
        return 0


def build_split(src_dir, dst_dir, size, workers, quality, limit=None):
    os.makedirs(dst_dir, exist_ok=True)
    names = [n for n in sorted(os.listdir(src_dir)) if n.lower().endswith(".jpg")]
    if limit:
        names = names[:limit]
    jobs = [(os.path.join(src_dir, n), os.path.join(dst_dir, n), size, quality) for n in names]
    t0 = time.time()
    done = 0
    with Pool(workers) as pool:
        for i, ok in enumerate(pool.imap_unordered(_convert, jobs, chunksize=32), 1):
            done += ok
            if i % 2000 == 0 or i == len(jobs):
                rate = i / max(time.time() - t0, 1e-6)
                eta = (len(jobs) - i) / max(rate, 1e-6) / 60
                print(f"[{os.path.basename(dst_dir)}] {i}/{len(jobs)} "
                      f"{rate:.1f} img/s ETA {eta:.1f} min", flush=True)
    total_bytes = sum(os.path.getsize(os.path.join(dst_dir, n)) for n in os.listdir(dst_dir))
    print(f"[done] {dst_dir}: {done} imgs, {total_bytes / 1024 ** 2:.0f} MB, "
          f"{time.time() - t0:.1f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description="构建 COCO 小尺寸图像缓存")
    p.add_argument("--src-root", default=r"F:\数据集\mscoco2017")
    p.add_argument("--dst-root", default=r"d:\workspace\github\.cache\coco2017_128")
    p.add_argument("--splits", nargs="+", default=["val2017", "train2017"])
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--quality", type=int, default=92)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    for split in args.splits:
        src = os.path.join(args.src_root, split)
        dst = os.path.join(args.dst_root, split)
        print(f"[build] {src} -> {dst} size={args.size} workers={args.workers}", flush=True)
        build_split(src, dst, args.size, args.workers, args.quality, args.limit)


if __name__ == "__main__":
    main()
