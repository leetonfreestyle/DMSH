"""CIFAR-10 数据加载：多标签(one-hot)化、按类均衡子集、禁用 HTTP 代理下载。"""
import os
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def disable_proxy():
    """彻底关闭下载时的 HTTP 代理。

    除了清空环境变量，还安装一个空 ProxyHandler 的 opener，
    防止 urllib 在 Windows 上读取注册表里的系统代理设置。
    """
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)
    os.environ["NO_PROXY"] = "*"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    urllib.request.install_opener(opener)


class OneHotCIFAR10(Dataset):
    """把 CIFAR-10 的单标签转成 10 维 one-hot，接口与 DMSH 的多标签输入一致。"""

    def __init__(self, base, num_classes=10):
        self.base = base
        self.num_classes = num_classes

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        onehot = torch.zeros(self.num_classes, dtype=torch.float32)
        onehot[label] = 1.0
        return img, onehot, label


def balanced_indices(targets, per_class, num_classes=10, seed=0):
    """每类最多取 per_class 个样本，保证子集类别均衡。"""
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    picked = []
    for c in range(num_classes):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        picked.extend(idx[:per_class].tolist())
    return sorted(picked)


def build_loaders(data_root, batch_size=128, num_workers=2,
                  train_per_class=None, query_per_class=None, database_per_class=None):
    """返回 train / query(测试集) / database(训练集) 三个 DataLoader。

    沿用哈希检索常用协议：CIFAR-10 训练集用于训练并作为检索库，
    测试集作为查询集；*_per_class 用于调试时缩小规模，None 表示全量。
    """
    disable_proxy()
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_base = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
    test_base = datasets.CIFAR10(data_root, train=False, download=True, transform=eval_tf)
    db_base = datasets.CIFAR10(data_root, train=True, download=False, transform=eval_tf)

    if train_per_class:
        train_base = Subset(train_base, balanced_indices(train_base.targets, train_per_class))
    if query_per_class:
        test_base = Subset(test_base, balanced_indices(test_base.targets, query_per_class))
    if database_per_class:
        db_base = Subset(db_base, balanced_indices(db_base.targets, database_per_class))

    train_set = OneHotCIFAR10(train_base)
    query_set = OneHotCIFAR10(test_base)
    db_set = OneHotCIFAR10(db_base)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    query_loader = DataLoader(query_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    db_loader = DataLoader(db_set, batch_size=batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
    return train_loader, query_loader, db_loader
