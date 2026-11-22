"""COCO 2017 多标签数据加载。

把 instances_{split}2017.json 的实例标注聚合成每张图的 multi-hot 标签，
用于 DMSH 的多标签哈希训练。相比 CIFAR-10 的单标签，COCO 天然一图多类。

用法要点：
- 零标注图片会被丢弃（在损失里等价于"与所有样本都 dissimilar"的噪声）。
- min_class_count 用于剔除长尾类，避免少数类主导 mAP（COCO 类别极不平衡）。
- 小规模验证时用 val2017：按 val_ratio 切成训练/查询两部分，
  训练部分同时作为检索库，查询部分只做评估。
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_coco_multilabel(root, split="val2017", min_class_count=0, verbose=True,
                         keep_cat_ids=None):
    """返回 (labels N×C float32, file_names, class_ids, class_names)。

    keep_cat_ids 给定时按该类别列表取列（用于训练/查询集口径一致）。
    """
    ann_path = os.path.join(root, "annotations", f"instances_{split}.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"缺少标注文件: {ann_path}")
    if verbose:
        print(f"[coco] 读取标注 {ann_path} ({os.path.getsize(ann_path) / 1024 ** 2:.1f} MB) ...", flush=True)
    with open(ann_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    cat_ids = sorted(c["id"] for c in data["categories"])
    cat_name = {c["id"]: c["name"] for c in data["categories"]}
    cat_col = {cid: i for i, cid in enumerate(cat_ids)}

    img_ids = [im["id"] for im in data["images"]]
    img_row = {iid: r for r, iid in enumerate(img_ids)}
    file_names = [im["file_name"] for im in data["images"]]

    labels = np.zeros((len(img_ids), len(cat_ids)), dtype=np.float32)
    for a in data["annotations"]:
        labels[img_row[a["image_id"]], cat_col[a["category_id"]]] = 1.0

    # 统计类别频次并筛掉长尾类
    counts = labels.sum(axis=0)
    if keep_cat_ids is not None:
        keep = np.array([cat_col[c] for c in keep_cat_ids], dtype=np.int64)
    else:
        keep = np.where(counts >= min_class_count)[0]
    labels = labels[:, keep]
    class_ids = [cat_ids[i] for i in keep]

    # 过滤掉（过滤后）无标签的图片
    has_label = labels.sum(axis=1) > 0
    kept = int(has_label.sum())
    if verbose:
        print(f"[coco] images={len(img_ids)} 有效={kept} 丢弃零标注={len(img_ids) - kept} "
              f"classes={len(cat_ids)}->{len(keep)} (min_class_count={min_class_count})", flush=True)
    labels = labels[has_label]
    file_names = [n for n, ok in zip(file_names, has_label) if ok]

    if verbose:
        per_img = labels.sum(axis=1)
        print(f"[coco] 每图平均类别数={per_img.mean():.2f} 最大={int(per_img.max())} "
              f"类别频次 min/max={int(counts[keep].min())}/{int(counts[keep].max())}", flush=True)
    return labels, file_names, class_ids, [cat_name[c] for c in class_ids]


class COCOMultiLabelDataset(Dataset):
    def __init__(self, img_dir, file_names, labels):
        self.img_dir = img_dir
        self.file_names = file_names
        self.labels = labels

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.file_names[idx])
        with Image.open(path) as im:
            img = im.convert("RGB")
        return img, idx


class _Transform:
    """把 PIL 图转成归一化张量；(train=True 时加随机裁剪与翻转)。"""

    def __init__(self, img_size, train):
        from torchvision import transforms
        norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        if train:
            self.tf = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize(img_size),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                norm,
            ])

    def __call__(self, img):
        return self.tf(img)


class _Wrapped(Dataset):
    """把 PIL 数据集包成 (tensor, onehot, index) 三元组，与 CIFAR 版本接口一致。"""

    def __init__(self, base, labels, transform):
        self.base = base
        self.labels = torch.from_numpy(labels)
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, i = self.base[idx]
        return self.transform(img), self.labels[i], i


def build_coco_loaders(root, split="val2017", img_size=224, batch_size=32, num_workers=0,
                       min_class_count=0, max_images=None, val_ratio=0.2, seed=0,
                       balanced=False, cache_root=None, query_split=None):
    """返回 train / query / database 三个 DataLoader 与类别信息。

    query_split 为 None（小规模验证）：把 split 按 val_ratio 随机切成训练/查询，
    训练部分同时作为检索库。
    query_split 给定时（全量训练）：split 全部作为训练集与检索库，
    另一个 split 全部作为查询集，类别口径以训练集为准。
    """
    labels, file_names, class_ids, class_names = load_coco_multilabel(
        root, split=split, min_class_count=min_class_count)

    if max_images and max_images < len(file_names):
        rng = np.random.RandomState(seed)
        sel = rng.choice(len(file_names), size=max_images, replace=False)
        labels = labels[sel]
        file_names = [file_names[i] for i in sel]

    if query_split:
        query_labels, q_files, _, _ = load_coco_multilabel(
            root, split=query_split, keep_cat_ids=class_ids, verbose=False)
        train_labels = labels
        print(f"[coco] 训练/库 split={split}({len(file_names)}) "
              f"查询 split={query_split}({len(q_files)})", flush=True)
    else:
        n = len(file_names)
        perm = np.random.RandomState(seed).permutation(n)
        n_query = max(1, int(round(n * val_ratio)))
        query_labels = labels[np.sort(perm[:n_query])]
        q_files = [file_names[i] for i in np.sort(perm[:n_query])]
        keep_rows = np.sort(perm[n_query:])
        train_labels = labels[keep_rows]
        file_names = [file_names[i] for i in keep_rows]
        print(f"[coco] split={split} train={len(file_names)} query={len(q_files)}", flush=True)

    print(f"[coco] train={len(train_labels)} query={len(query_labels)} "
          f"每图标签数 train={train_labels.sum(1).mean():.2f} "
          f"query={query_labels.sum(1).mean():.2f}", flush=True)

    def _dir(s):
        d = os.path.join(cache_root, s) if cache_root else os.path.join(root, s)
        if cache_root and not os.path.isdir(d):
            raise FileNotFoundError(f"缓存目录不存在: {d}（先运行 build_coco_cache.py）")
        return d

    train_dir = _dir(split)
    query_dir = _dir(query_split or split)
    if cache_root:
        print(f"[coco] 图像来源: 本地缓存 {cache_root}", flush=True)

    train_base = COCOMultiLabelDataset(train_dir, file_names, train_labels)
    db_base = COCOMultiLabelDataset(train_dir, file_names, train_labels)
    query_base = COCOMultiLabelDataset(query_dir, q_files, query_labels)

    train_set = _Wrapped(train_base, train_labels, _Transform(img_size, train=True))
    db_set = _Wrapped(db_base, train_labels, _Transform(img_size, train=False))
    query_set = _Wrapped(query_base, query_labels, _Transform(img_size, train=False))

    common = dict(num_workers=num_workers, pin_memory=True)
    if balanced:
        # 多标签类别均衡：图片权重取其标签类别频次倒数平方根的均值，
        # 有放回采样，避免 person 等高频类支配每个 batch。
        freq = np.maximum(train_labels.sum(axis=0), 1.0)
        inv_sqrt = 1.0 / np.sqrt(freq)
        n_labels = np.maximum(train_labels.sum(axis=1), 1.0)
        weights = (train_labels * inv_sqrt).sum(axis=1) / n_labels
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights).double(), num_samples=len(train_set), replacement=True)
        train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler,
                                  drop_last=True, **common)
        print("[coco] 训练集启用类别均衡采样（类别频次倒数平方根）", flush=True)
    else:
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                  drop_last=True, **common)
    db_loader = DataLoader(db_set, batch_size=batch_size, shuffle=False, **common)
    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False, **common)
    return train_loader, query_loader, db_loader, {
        "num_classes": len(class_ids),
        "class_ids": class_ids,
        "class_names": class_names,
    }
