"""DMSH 多相似度哈希损失的 PyTorch 移植。

忠实对应 src/caffe/layers/hashing_loss_layer.cpp（DMSH master 修复版）：
1. 标签重叠数 overlap = y_i · y_j（多标签为共享类别数，单标签退化为 0/1）。
2. 自适应采样：sim_ratio = overlap >= nsim 的对比例；当 sim_ratio < 0.1 时，
   非相似对以 p_dis = (sim_ratio/0.1 - sim_ratio)/(1 - sim_ratio) 的概率保留，
   相似对全部保留（原实现中 p_dis 默认 1.1，即不采样）。
3. 相似对（overlap >= 1）：
       margin_ij = K * 4 * bi_margin * exp(-overlap_ij)
       loss_ij   = max(||c_i - c_j||^2 - margin_ij, 0)
   共享标签越多，margin 越小，要求码字越接近。
4. 非相似对：loss_ij = max(2K - ||c_i - c_j||^2, 0)，
   即把汉明距离推到至少 K/2（±1 码下 ||c_i-c_j||^2 = 4 * Hamming）。
5. 量化正则：reg = mean_i sum_k |||c_ik| - 1||，总损失
       loss = mean(保留对的 pair_loss) + tradeoff * reg

注意：原 Caffe 实现损失除以 count_iter，但梯度用 alpha=1/(N(N-1)) 缩放，
两者归一化不一致。这里按上述损失定义交给 autograd 求导，梯度与损失严格一致。
"""
import torch
from torch import nn


class DMSHLoss(nn.Module):
    def __init__(self, nbit, margin=0.62, tradeoff=0.01, nsim=1, limit_low=0.1):
        super().__init__()
        self.nbit = nbit
        self.margin = margin
        self.tradeoff = tradeoff
        self.nsim = nsim
        self.limit_low = limit_low

    def forward(self, codes, labels):
        n = codes.size(0)
        device = codes.device
        if n < 2:
            zero = codes.sum() * 0.0
            return zero, {"pair_loss": 0.0, "reg": 0.0, "sim_ratio": 0.0, "kept_pairs": 0}

        labels = labels.float()
        overlap = labels @ labels.t()
        iu = torch.triu_indices(n, n, offset=1, device=device)
        ov = overlap[iu[0], iu[1]]  # (P,)

        sim_ratio = (ov >= self.nsim).float().mean()
        p_dis = torch.tensor(1.1, device=device)
        if sim_ratio.item() < self.limit_low:
            p_dis = (sim_ratio / self.limit_low - sim_ratio) / (1.0 - sim_ratio)

        sim = ov >= 1.0
        rand = torch.rand_like(ov)
        keep = sim | (rand <= p_dis)

        diff = codes[iu[0]] - codes[iu[1]]
        dist_sq = (diff * diff).sum(dim=1)

        sim_margin = self.nbit * 4.0 * self.margin * torch.exp(-ov)
        pair_loss = torch.where(
            sim,
            torch.clamp(dist_sq - sim_margin, min=0.0),
            torch.clamp(2.0 * self.nbit - dist_sq, min=0.0),
        )
        kept = pair_loss[keep]
        pair_term = kept.mean() if kept.numel() > 0 else codes.sum() * 0.0

        reg = (codes.abs() - 1.0).abs().sum(dim=1).mean()
        loss = pair_term + self.tradeoff * reg

        stats = {
            "pair_loss": float(pair_term.detach()),
            "reg": float(reg.detach()),
            "sim_ratio": float(sim_ratio.detach()),
            "kept_pairs": int(keep.sum()),
        }
        return loss, stats
