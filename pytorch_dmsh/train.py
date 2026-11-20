"""DMSH PyTorch 版训练入口（CIFAR-10）。

示例：
    # 冒烟调试：每类 100 张训练图、少量查询/库样本，验证 GPU 管线
    python pytorch_dmsh/train.py --smoke-test

    # 常规训练（全量 CIFAR-10）
    python pytorch_dmsh/train.py --bits 48 --epochs 30 --batch-size 128

数据集下载全程禁用 HTTP 代理（见 data.disable_proxy）。
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from data import build_loaders
from evaluate import evaluate
from losses import DMSHLoss
from model import DMSHNet


def parse_args():
    p = argparse.ArgumentParser(description="DMSH (PyTorch) on CIFAR-10")
    p.add_argument("--data-root", default="pytorch_dmsh/data")
    p.add_argument("--output-dir",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))
    p.add_argument("--bits", type=int, default=48)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--margin", type=float, default=0.62, help="Caffe bi_margin")
    p.add_argument("--tradeoff", type=float, default=0.01, help="量化正则权重")
    p.add_argument("--nsim", type=int, default=1, help="自适应采样的相似对阈值")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--train-per-class", type=int, default=None)
    p.add_argument("--query-per-class", type=int, default=None)
    p.add_argument("--database-per-class", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=0, help="每 N 个 epoch 评估一次，0=仅训练后评估")
    p.add_argument("--smoke-test", action="store_true", help="小样本快速验证 GPU 训练与检索")
    return p.parse_args()


def main():
    args = parse_args()
    if args.smoke_test:
        args.epochs = 2
        args.train_per_class = 100
        args.query_per_class = 20
        args.database_per_class = 100
        args.eval_every = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
    if device.type == "cuda":
        print(f"[env] GPU={torch.cuda.get_device_name(0)} "
              f"CUDA(runtime)={torch.version.cuda} capability={torch.cuda.get_device_capability(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("[env] 警告：未检测到可用 CUDA，将在 CPU 上运行")

    os.makedirs(args.output_dir, exist_ok=True)
    train_loader, query_loader, db_loader = build_loaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
        train_per_class=args.train_per_class,
        query_per_class=args.query_per_class,
        database_per_class=args.database_per_class,
    )
    print(f"[data] train={len(train_loader.dataset)} query={len(query_loader.dataset)} "
          f"database={len(db_loader.dataset)} (CIFAR-10, 无代理下载)")

    model = DMSHNet(nbit=args.bits).to(device)
    criterion = DMSHLoss(nbit=args.bits, margin=args.margin,
                         tradeoff=args.tradeoff, nsim=args.nsim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {"loss": 0.0, "pair_loss": 0.0, "reg": 0.0, "sim_ratio": 0.0}
        n_batch = 0
        for imgs, onehot, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            onehot = onehot.to(device, non_blocking=True)
            codes = model(imgs)
            loss, stats = criterion(codes, onehot)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            agg["loss"] += float(loss.detach())
            for k in ("pair_loss", "reg", "sim_ratio"):
                agg[k] += stats[k]
            n_batch += 1
            if n_batch == 1 and epoch == 1:
                print(f"[debug] codes shape={tuple(codes.shape)} device={codes.device} "
                      f"kept_pairs={stats['kept_pairs']}")

        for k in agg:
            agg[k] /= max(n_batch, 1)
        msg = (f"[epoch {epoch}/{args.epochs}] loss={agg['loss']:.4f} "
               f"pair={agg['pair_loss']:.4f} reg={agg['reg']:.4f} "
               f"sim_ratio={agg['sim_ratio']:.3f} "
               f"time={time.time() - t0:.1f}s")
        if device.type == "cuda":
            msg += f" max_mem={torch.cuda.max_memory_allocated() / 1024 ** 2:.0f}MB"
        print(msg)
        history.append({"epoch": epoch, **agg})

        if args.eval_every and epoch % args.eval_every == 0 and epoch < args.epochs:
            result = evaluate(model, query_loader, db_loader, device, args.bits)
            print(f"    -> mAP@all = {result['mAP']:.4f}")

    result = evaluate(model, query_loader, db_loader, device, args.bits)
    print(f"[final] bits={args.bits} mAP@all={result['mAP']:.4f} "
          f"(query={len(result['query_codes'])}, database={len(result['db_codes'])})")

    ckpt = os.path.join(args.output_dir, f"dmsh_cifar10_{args.bits}bit.pt")
    torch.save({"model": model.state_dict(), "args": vars(args), "mAP": result["mAP"]}, ckpt)
    np.savez_compressed(os.path.join(args.output_dir, f"dmsh_cifar10_{args.bits}bit_codes.npz"),
                        query_codes=result["query_codes"], db_codes=result["db_codes"],
                        query_labels=result["query_labels"], db_labels=result["db_labels"])
    with open(os.path.join(args.output_dir, f"dmsh_cifar10_{args.bits}bit_history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[save] checkpoint={ckpt}")


if __name__ == "__main__":
    main()
