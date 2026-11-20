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
from data_coco import build_coco_loaders
from evaluate import evaluate
from losses import DMSHLoss
from model import DMSHNet


def parse_args():
    p = argparse.ArgumentParser(description="DMSH (PyTorch) on CIFAR-10")
    p.add_argument("--data-root",
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
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
    p.add_argument("--resume", default=None, help="从该 checkpoint 续训（恢复权重/epoch/best_mAP）")
    p.add_argument("--dataset", choices=["cifar10", "coco"], default="cifar10")
    p.add_argument("--backbone", choices=["cifar", "resnet18"], default="cifar")
    p.add_argument("--pretrained", action="store_true", help="resnet18 使用 ImageNet 预训练权重")
    p.add_argument("--img-size", type=int, default=32)
    p.add_argument("--coco-root", default=r"F:\数据集\mscoco2017")
    p.add_argument("--coco-split", default="val2017")
    p.add_argument("--min-class-count", type=int, default=0, help="COCO 标注中剔除频次低于该值的类别")
    p.add_argument("--coco-max-images", type=int, default=None, help="COCO 小规模验证时随机取图数量")
    p.add_argument("--coco-val-ratio", type=float, default=0.2, help="COCO 从同 split 切出的查询集比例")
    p.add_argument("--coco-balanced", action="store_true", help="COCO 训练启用类别均衡采样")
    p.add_argument("--map-k", type=int, default=None, help="评估 mAP@K；缺省为 mAP@all")
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
    metric_name = f"mAP@{args.map_k}" if args.map_k else "mAP@all"
    num_classes = 10
    if args.dataset == "coco":
        train_loader, query_loader, db_loader, coco_info = build_coco_loaders(
            args.coco_root, split=args.coco_split, img_size=args.img_size,
            batch_size=args.batch_size, num_workers=args.num_workers,
            min_class_count=args.min_class_count, max_images=args.coco_max_images,
            val_ratio=args.coco_val_ratio, balanced=args.coco_balanced,
        )
        num_classes = coco_info["num_classes"]
        print(f"[data] dataset=COCO split={args.coco_split} classes={num_classes} "
              f"train={len(train_loader.dataset)} query={len(query_loader.dataset)} "
              f"database={len(db_loader.dataset)} img_size={args.img_size}")
    else:
        train_loader, query_loader, db_loader = build_loaders(
            args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
            train_per_class=args.train_per_class,
            query_per_class=args.query_per_class,
            database_per_class=args.database_per_class,
        )
        print(f"[data] train={len(train_loader.dataset)} query={len(query_loader.dataset)} "
              f"database={len(db_loader.dataset)} (CIFAR-10, 无代理下载)")

    model = DMSHNet(nbit=args.bits, backbone=args.backbone,
                    img_size=args.img_size, pretrained=args.pretrained).to(device)
    print(f"[model] backbone={args.backbone} pretrained={args.pretrained} img_size={args.img_size} "
          f"bits={args.bits} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    criterion = DMSHLoss(nbit=args.bits, margin=args.margin,
                         tradeoff=args.tradeoff, nsim=args.nsim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_map = -1.0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        start_epoch = int(ck.get("epoch", 0))
        best_map = float(ck.get("best_mAP", ck.get("mAP", -1.0)))
        print(f"[resume] loaded {args.resume}: epoch={start_epoch} best_mAP={best_map:.4f}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, last_epoch=start_epoch - 1)

    history = []
    last_ckpt = os.path.join(args.output_dir, f"dmsh_{args.dataset}_{args.bits}bit_last.pt")
    best_ckpt = os.path.join(args.output_dir, f"dmsh_{args.dataset}_{args.bits}bit_best.pt")
    eval_done = False
    for epoch in range(start_epoch + 1, args.epochs + 1):
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

        scheduler.step()
        for k in agg:
            agg[k] /= max(n_batch, 1)
        msg = (f"[epoch {epoch}/{args.epochs}] loss={agg['loss']:.4f} "
               f"pair={agg['pair_loss']:.4f} reg={agg['reg']:.4f} "
               f"sim_ratio={agg['sim_ratio']:.3f} "
               f"lr={scheduler.get_last_lr()[0]:.2e} time={time.time() - t0:.1f}s")
        if device.type == "cuda":
            msg += f" max_mem={torch.cuda.max_memory_allocated() / 1024 ** 2:.0f}MB"
        print(msg, flush=True)

        do_eval = (epoch == args.epochs) or (bool(args.eval_every) and epoch % args.eval_every == 0)
        cur_map = None
        if do_eval:
            result = evaluate(model, query_loader, db_loader, device, args.bits,
                              top_k=args.map_k)
            cur_map = result["mAP"]
            eval_done = (epoch == args.epochs)
            tag = ""
            if cur_map > best_map:
                best_map = cur_map
                torch.save({"model": model.state_dict(), "args": vars(args),
                            "epoch": epoch, "best_mAP": best_map, "mAP": cur_map}, best_ckpt)
                tag = " *BEST*"
            print(f"    -> {metric_name}={cur_map:.4f} (best={best_map:.4f}){tag}", flush=True)

        history.append({"epoch": epoch, **agg, "mAP": cur_map, "best_mAP": best_map})
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "args": vars(args), "epoch": epoch, "best_mAP": best_map}, last_ckpt)

    if not eval_done:
        result = evaluate(model, query_loader, db_loader, device, args.bits,
                          top_k=args.map_k)
        cur_map = result["mAP"]
        if cur_map > best_map:
            best_map = cur_map
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": args.epochs, "best_mAP": best_map, "mAP": cur_map}, best_ckpt)
        history[-1]["mAP"] = cur_map
        history[-1]["best_mAP"] = best_map
        print(f"    -> {metric_name}={cur_map:.4f} (best={best_map:.4f})", flush=True)

    print(f"[final] bits={args.bits} best_{metric_name}={best_map:.4f} "
          f"(query={len(result['query_codes'])}, database={len(result['db_codes'])})")
    np.savez_compressed(os.path.join(args.output_dir, f"dmsh_{args.dataset}_{args.bits}bit_codes.npz"),
                        query_codes=result["query_codes"], db_codes=result["db_codes"],
                        query_labels=result["query_labels"], db_labels=result["db_labels"])
    with open(os.path.join(args.output_dir, f"dmsh_{args.dataset}_{args.bits}bit_history.json"), "w",
              encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"[save] last={last_ckpt} best={best_ckpt}", flush=True)


if __name__ == "__main__":
    main()
