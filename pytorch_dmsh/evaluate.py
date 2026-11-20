"""哈希码提取与检索评估（Hamming 排序 mAP）。

与 leeton/eval_mAP.py 的口径一致：对 ±1 二值码按内积排序
（内积越大汉明距离越小），相关定义为标签重叠数 >= 1，
CIFAR-10 单标签下即"同类别"。查询集与检索库不重叠，无需排除自身。
"""
import numpy as np
import torch

from model import binarize


@torch.no_grad()
def extract_codes(model, loader, device):
    model.eval()
    continuous, binary, onehots, labels = [], [], [], []
    for imgs, onehot, label in loader:
        imgs = imgs.to(device, non_blocking=True)
        codes = model(imgs)
        continuous.append(codes.cpu())
        binary.append(binarize(codes).cpu())
        onehots.append(onehot)
        labels.append(label)
    return (torch.cat(continuous), torch.cat(binary),
            torch.cat(onehots), torch.cat(labels))


@torch.no_grad()
def hamming_map(query_codes, db_codes, query_onehot, db_onehot, device,
                chunk=256, top_k=None):
    """计算汉明排序 mAP。query_codes/db_codes 为 ±1 码，形状 (N, K)。

    top_k=None 时计算 mAP@all；否则只评估每个查询的前 top_k 个检索结果，
    AP 分母采用 min(库中相关总数, top_k)，即标准 mAP@K 口径。
    """
    q = query_codes.to(device).float()
    db = db_codes.to(device).float()
    qy = query_onehot.to(device).float()
    dby = db_onehot.to(device).float()

    # 相关矩阵：标签重叠 >= 1（单标签即同类）
    rel_all = (qy @ dby.t()) >= 1.0
    n_rel_total = rel_all.sum(dim=1).float()
    aps = []
    for start in range(0, q.size(0), chunk):
        end = min(start + chunk, q.size(0))
        scores = q[start:end] @ db.t()          # 内积越大 => 汉明距离越小
        if top_k is not None and top_k < db.size(0):
            order = scores.topk(top_k, dim=1, largest=True, sorted=True).indices
        else:
            order = scores.argsort(dim=1, descending=True)
        rel = rel_all[start:end].gather(1, order).float()
        cum = rel.cumsum(dim=1)
        ranks = torch.arange(1, order.size(1) + 1, device=device).float().unsqueeze(0)
        precision = cum / ranks
        n_rel = n_rel_total[start:end]
        if top_k is not None:
            n_rel = n_rel.clamp(max=float(top_k))
        valid = n_rel > 0
        ap = (precision * rel).sum(dim=1) / n_rel.clamp(min=1.0)
        aps.append(ap[valid].cpu())
    return float(torch.cat(aps).mean()) if aps else 0.0


def evaluate(model, query_loader, db_loader, device, nbit, top_k=None):
    _, q_codes, q_onehot, _ = extract_codes(model, query_loader, device)
    _, d_codes, d_onehot, _ = extract_codes(model, db_loader, device)
    mAP = hamming_map(q_codes, d_codes, q_onehot, d_onehot, device, top_k=top_k)
    map_all = mAP if top_k is None else hamming_map(
        q_codes, d_codes, q_onehot, d_onehot, device)
    return {
        "mAP": mAP,
        "mAP@all": map_all,
        "top_k": top_k,
        "query_codes": q_codes.numpy().astype(np.int8),
        "db_codes": d_codes.numpy().astype(np.int8),
        "query_labels": q_onehot.numpy().astype(np.int8),
        "db_labels": d_onehot.numpy().astype(np.int8),
        "nbit": nbit,
    }
