"""DMSH 的 PyTorch 网络结构。

对应 Caffe 版 leeton/finetune.prototxt 的小网络：
3 个 5x5 卷积(32/32/64) + FC500 + 哈希层，输入为 32x32 图像（CIFAR-10）。
哈希层是线性输出（不加 tanh/sigmoid），训练时由量化正则把输出压向 ±1，
检索时用 sign 二值化，与 Caffe 版 extract_code.sh 中按 0 阈值取码一致。
"""
import torch
from torch import nn


class DMSHNet(nn.Module):
    def __init__(self, nbit=48, in_channels=3, fc_dim=500):
        super().__init__()
        self.nbit = nbit
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8 -> 4
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, fc_dim),
            nn.ReLU(inplace=True),
        )
        # 对应 Caffe 的 ip1：高斯初始化 std=0.001
        self.hash_layer = nn.Linear(fc_dim, nbit)
        nn.init.normal_(self.hash_layer.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.hash_layer.bias)

    def forward(self, x):
        z = self.features(x)
        z = self.fc(z)
        return self.hash_layer(z)  # 连续哈希码，形状 (N, nbit)


def binarize(codes):
    """连续码 -> ±1 二值码。0 归为 +1，避免产生全 0 位。"""
    return torch.where(codes > 0, torch.ones_like(codes), -torch.ones_like(codes))
