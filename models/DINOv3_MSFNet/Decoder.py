import torch
import torch.nn as nn
import torch.nn.functional as F
from .MSFD_Decoder import MSFD
from .sample_blocks import FeatureFusionBlock, _make_scratch


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )

class FeatureEnhancementModule(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels=[256, 512, 1024, 1024],
                 num_modalities: int = 1):
        super(FeatureEnhancementModule, self).__init__()

        # 简单改动：由单个 1x1 Conv 改成 Conv + BN + ReLU
        # 输入输出通道不变
        self.projects = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            ) for out_channel in out_channels
        ])

        # 尺度变换逻辑保持不变
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0
            ),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0
            ),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1
            )
        ])

        self.num_modalities = num_modalities

    def forward(self, *features_list, patch_h=None, patch_w=None):
        if len(features_list) == 0:
            raise ValueError("At least one feature set must be provided")

        if patch_h is None or patch_w is None:
            raise ValueError("patch_h and patch_w must be provided")

        out = []

        for i, x in enumerate(features_list[0]):
            # x: [B, N, C] -> [B, C, H, W]
            x = x.permute(0, 2, 1).reshape(
                x.shape[0],
                x.shape[-1],
                patch_h,
                patch_w
            )

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        return out

class ConvBNReLU(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        pad: int = 1,
    ):
        layers = [
            nn.Conv2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      pad,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        super().__init__(*layers)

class CrossScaleFeatureModule(nn.Module):

    def __init__(self, in_d=None, out_d=64, drop_rate=0):
        super(CrossScaleFeatureModule, self).__init__()
        if in_d is None:
            raise ValueError("in_d must be provided")
        self.in_d = in_d
        self.mid_d = out_d // 2
        self.out_d = out_d

        # Define all conv_scale modules dynamically using a loop
        self.conv_scales = nn.ModuleDict()
        for scale in range(2, 6):  # For scales 2 to 5
            for i in range(2, 6):  # For each conv_scale1_c2 ... conv_scale5_c5
                key = f"conv_scale{i}_c{scale}"
                self.conv_scales[key] = self._create_conv_block(
                    self.in_d[scale - 1],
                    self.mid_d,
                    scale=i,
                    orig_scale=scale)

        # Fusion layers
        self.conv_aggregation_s2 = FeatureFusionModule(self.mid_d * 4,
                                                       self.in_d[1],
                                                       self.in_d[1], drop_rate)
        self.conv_aggregation_s3 = FeatureFusionModule(self.mid_d * 4,
                                                       self.in_d[2],
                                                       self.in_d[2], drop_rate)
        self.conv_aggregation_s4 = FeatureFusionModule(self.mid_d * 4,
                                                       self.in_d[3],
                                                       self.in_d[3], drop_rate)
        self.conv_aggregation_s5 = FeatureFusionModule(self.mid_d * 4,
                                                       self.in_d[4],
                                                       self.in_d[4], drop_rate)

    def _create_conv_block(self, in_channels, mid_channels, scale, orig_scale):
        layers = []
        if scale > orig_scale:  # Pooling for scales > 1
            layers.append(
                nn.MaxPool2d(
                    kernel_size=2**(scale - orig_scale),
                    stride=2**(scale - orig_scale),
                ))

        if scale == orig_scale:
            layers.extend([
                nn.Conv2d(in_channels,
                          mid_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ])
        elif scale != orig_scale:
            layers.extend([
                nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    mid_channels,
                    mid_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=mid_channels,
                ),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            ])

        return nn.Sequential(*layers)

    def forward(self, c2, c3, c4, c5):
        # Handle each scale's forward pass dynamically
        def process_scale(c, scale_idx):
            scale_outputs = []
            for i in range(2, 6):  # For scales 2 to 5
                key = f"conv_scale{i}_c{scale_idx + 2}"
                output = self.conv_scales[key](c)
                if i < scale_idx + 2:  # Interpolate as needed
                    output = F.interpolate(
                        output,
                        scale_factor=(
                            2**(scale_idx + 2 - i),
                            2**(scale_idx + 2 - i),
                        ),
                        mode="bilinear",
                    )
                scale_outputs.append(output)
            return scale_outputs


        # Get outputs for all input features
        c2_scales = process_scale(c2, 0)
        c3_scales = process_scale(c3, 1)
        c4_scales = process_scale(c4, 2)
        c5_scales = process_scale(c5, 3)

        # Aggregation and fusion
        s2 = self.conv_aggregation_s2(
            torch.cat([c2_scales[0], c3_scales[0], c4_scales[0], c5_scales[0]],
                      dim=1),
            c2,
        )
        s3 = self.conv_aggregation_s3(
            torch.cat([c2_scales[1], c3_scales[1], c4_scales[1], c5_scales[1]],
                      dim=1),
            c3,
        )
        s4 = self.conv_aggregation_s4(
            torch.cat([c2_scales[2], c3_scales[2], c4_scales[2], c5_scales[2]],
                      dim=1),
            c4,
        )
        s5 = self.conv_aggregation_s5(
            torch.cat([c2_scales[3], c3_scales[3], c4_scales[3], c5_scales[3]],
                      dim=1),
            c5,
        )

        return s2, s3, s4, s5

class FeatureFusionModule(nn.Module):

    def __init__(self, fuse_d, in_d, out_d, drop_rate):
        super(FeatureFusionModule, self).__init__()
        self.fuse_d = fuse_d
        self.in_d = in_d
        self.out_d = out_d
        self.conv_fuse = nn.Sequential(
            nn.Conv2d(self.fuse_d, self.fuse_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.fuse_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                self.fuse_d,
                self.fuse_d,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=self.fuse_d,
            ),
            nn.BatchNorm2d(self.fuse_d),
            nn.ReLU(inplace=True),
            # nn.Dropout(drop_rate),
            nn.Conv2d(self.fuse_d, self.out_d, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.out_d),
        )
        self.conv_identity = nn.Conv2d(self.in_d, self.out_d, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, c_fuse, c):
        c_fuse = self.conv_fuse(c_fuse)
        c_out = self.relu(c_fuse + self.conv_identity(c))

        return c_out

class Decoder_FEM(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128
        self.out = MSFD(n_classes)

    def forward(self, *modalities):
        x = modalities[0]
        features = x

        out = self.out(features[0],features[1],features[2],features[3])
        return out

class Decoder_FEM_CSFM(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128

        # TODO: change the input channels
        self.CSFM = CrossScaleFeatureModule([in_channels[0]] + in_channels,
                                              out_channels)
        self.out = MSFD(n_classes)

    def forward(self, *modalities):
        x = modalities[0]
        features = self.CSFM(*x)

        out = self.out(features[0],features[1],features[2],features[3])
        return out

class Decoder_CSFM(nn.Module):

    def __init__(
        self,
        n_classes,
        in_channels=[256, 512, 1024, 1024],
        out_channels=256,
    ):
        super().__init__()

        self.in_channels = in_channels  # 1024
        self.out_channels = out_channels  # 1024 // 8 = 128

        # TODO: change the input channels
        self.CSFM = CrossScaleFeatureModule([in_channels[0]] + in_channels,
                                              out_channels)

        self.out_conv = ConvBNReLU(out_channels, n_classes, 1, pad=0)

    def forward(self, *modalities):
        x = modalities[0]
        features = self.CSFM(*x)
        x1, x2, x3, x4 = features

        H, W = x1.shape[2:]
        p2 = x1
        for _x in [x2, x3, x4]:
            p2 = p2 + F.interpolate(
                _x,
                size=(H, W),
                mode="bilinear",
            )

        return self.out_conv(p2)


class DPTHead(nn.Module):

    def __init__(
        self,
        nclass,
        in_channels,
        features=256,
        use_bn=False,
        out_channels=[256, 512, 1024, 1024],
    ):
        super(DPTHead, self).__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])

        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=out_channels[0],
                               out_channels=out_channels[0],
                               kernel_size=4,
                               stride=4,
                               padding=0),
            nn.ConvTranspose2d(in_channels=out_channels[1],
                               out_channels=out_channels[1],
                               kernel_size=2,
                               stride=2,
                               padding=0),
            nn.Identity(),
            nn.Conv2d(in_channels=out_channels[3],
                      out_channels=out_channels[3],
                      kernel_size=3,
                      stride=2,
                      padding=1)
        ])

        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(features, nclass, kernel_size=1, stride=1, padding=0))

    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            x = x.permute(0, 2, 1).reshape(
                (x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[i](x)
            x = self.resize_layers[i](x)

            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4,
                                         layer_3_rn,
                                         size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3,
                                         layer_2_rn,
                                         size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out


class DPT(nn.Module):

    def __init__(self,
                 encoder_size='base',
                 nclass=21,
                 features=128,
                 out_channels=[96, 192, 384, 768],
                 use_bn=False,
                 backbone=None):
        super(DPT, self).__init__()

        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
            'large': [4, 11, 17, 23],
            'giant': [9, 19, 29, 39]
        }

        self.encoder_size = encoder_size
        self.backbone = backbone
        # Important: we freeze the backbone
        self.backbone.requires_grad_(False)
        self.head = DPTHead(nclass,
                            self.backbone.embed_dim,
                            features,
                            use_bn,
                            out_channels=out_channels)
        # self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def lock_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x):
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16
        # features = self.backbone.get_intermediate_layers(
        #     x, n = self.intermediate_layer_idx[self.encoder_size], reshape=True, norm=True
        # )
        features = self.backbone.get_intermediate_layers(
            x, n=self.intermediate_layer_idx[self.encoder_size])

        out = self.head(features, patch_h, patch_w)
        out = F.interpolate(out, (patch_h * 16, patch_w * 16),
                            mode='bilinear',
                            align_corners=True)
        return out