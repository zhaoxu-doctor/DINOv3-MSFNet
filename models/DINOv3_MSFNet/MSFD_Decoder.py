import torch
import torch.nn as nn


class Up(nn.Module):
    def __init__(self, in_size, out_size):
        super(Up, self).__init__()
        self.conv1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up    = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        """inputs1: skip (浅层)，inputs2: 深层特征（上采样后）"""
        outputs = torch.cat([inputs1, self.up(inputs2)], dim=1)
        outputs = self.relu(self.conv1(outputs))
        outputs = self.relu(self.conv2(outputs))
        return outputs


# ─────────────────────────────────────────
# 通道投影模块
# ─────────────────────────────────────────
class ChannelProj(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelProj, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.proj(x)


class MSFD(nn.Module):
    """
    输入特征：
        skip1     : (B, 256, 128, 128)
        skip2     : (B, 512,  64,  64)
        skip3     : (B,1024,  32,  32)
        bottleneck: (B,1024,  16,  16)

    输出：(B, num_classes, 512, 512)
    """
    def __init__(self, num_classes=4):
        super(MSFD, self).__init__()

        # 通道投影（逐层压缩，符合经典 U-Net 风格）
        self.proj_bottleneck = ChannelProj(1024, 512)   # bottleneck 16×16
        self.proj_skip3      = ChannelProj(1024, 256)   # skip3 32×32
        self.proj_skip2      = ChannelProj( 512, 128)   # skip2 64×64
        self.proj_skip1      = ChannelProj( 256,  64)   # skip1 128×128

        # 解码块
        self.up_concat3 = Up(in_size=256 + 512, out_size=256)   # 32×32
        self.up_concat2 = Up(in_size=128 + 256, out_size=128)   # 64×64
        self.up_concat1 = Up(in_size=64  + 128, out_size=64)    # 128×128

        # 从 128×128 → 256×256 的过渡块
        self.up_to_256 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 从 256×256 → 512×512 的最终上采样块
        self.up_to_512 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # 分割头
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, skip1, skip2, skip3, bottleneck):
        """
        Args:
            skip1     : (B, 256, 128, 128)
            skip2     : (B, 512,  64,  64)
            skip3     : (B,1024,  32,  32)
            bottleneck: (B,1024,  16,  16)
        Returns:
            logits    : (B, num_classes, 512, 512)
        """
        # 通道投影
        f_b = self.proj_bottleneck(bottleneck)   # (B,512,16,16)
        f3  = self.proj_skip3(skip3)             # (B,256,32,32)
        f2  = self.proj_skip2(skip2)             # (B,128,64,64)
        f1  = self.proj_skip1(skip1)             # (B,64,128,128)

        # 解码过程
        up3 = self.up_concat3(f3, f_b)           # (B,256,32,32)
        up2 = self.up_concat2(f2, up3)           # (B,128,64,64)
        up1 = self.up_concat1(f1, up2)           # (B,64,128,128)

        # 上采样到 256×256
        up_256 = self.up_to_256(up1)             # (B,64,256,256)

        # 上采样到 512×512
        up_512 = self.up_to_512(up_256)          # (B,64,512,512)

        # 最终分割头
        logits = self.final(up_512)              # (B, num_classes, 512, 512)

        return logits

    def freeze_backbone(self):
        """冻结投影层"""
        for proj in [self.proj_bottleneck, self.proj_skip3, self.proj_skip2, self.proj_skip1]:
            for param in proj.parameters():
                param.requires_grad = False

    def unfreeze_backbone(self):
        """解冻投影层"""
        for proj in [self.proj_bottleneck, self.proj_skip3, self.proj_skip2, self.proj_skip1]:
            for param in proj.parameters():
                param.requires_grad = True



