"""DMSH 的 PyTorch 网络结构。

两种骨干：
- ``cifar``：对应 Caffe 版 leeton/finetune.prototxt 的小网络，
  3 个 5x5 卷积(32/32/64) + FC500 + 哈希层，输入 32x32（CIFAR-10）。
- ``resnet18``：用于 COCO 等大尺寸多标签数据，输入 224x224（或 128），
  去掉 ImageNet 分类头，用 512 维全局特征接 FC500 + 哈希层。

哈希层是线性输出（不加 tanh/sigmoid），训练时由量化正则把输出压向 ±1，
检索时用 sign 二值化，与 Caffe 版 extract_code.sh 中按 0 阈值取码一致。
"""
import torch
from torch import nn
from torchvision import models


class DMSHNet(nn.Module):
    def __init__(self, nbit=48, backbone="cifar", img_size=32, fc_dim=500,
                 in_channels=3, pretrained=False):
        super().__init__()
        self.nbit = nbit
        self.backbone_name = backbone

        if backbone == "cifar":
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
            pooled = img_size // 8  # 3 次下采样
            feat_dim = 64 * pooled * pooled
        elif backbone == "resnet18":
            net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            net.fc = nn.Identity()
            self.features = net
            feat_dim = 512
        else:
            raise ValueError(f"unsupported backbone: {backbone}")

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, fc_dim),
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
