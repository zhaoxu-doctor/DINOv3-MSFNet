import math
import os
import sys

import torch.nn as nn
import torch.nn.functional as F

from .Decoder import FeatureEnhancementModule, Decoder_FEM, Decoder_FEM_CSFM, Decoder_CSFM
from .ResNet import ResNet50
from .linear_decoder import LinearHead
from .lora import LoRA

from dinov3.hub.backbones import dinov3_vitl16, dinov3_vits16plus, dinov3_vitb16, dinov3_vits16, dinov3_vit7b16

# 添加项目根目录到 Python 路径中，以便可以导入 dinov3 模块
deps_path = os.path.join(os.path.dirname(__file__), "task/segmentation")
sys.path.insert(0, deps_path)

BACKBONE_INTERMEDIATE_LAYERS = {
    "dinov3_vits16": [2, 5, 8, 11],
    "dinov3_vits16plus": [2, 5, 8, 11],
    "dinov3_vitb16": [2, 5, 8, 11],
    "dinov3_vitl16": [4, 11, 17, 23],
    "dinov3_vit7b16": [9, 19, 29, 39],
}


class DINOSegmentModule(nn.Module):

    def __init__(
        self,
        backbone_weights=None,
        freeze_backbone: bool = False,
        n_classes: int = 1000,
        # window_size=(224, 224),
        use_lora: bool = False,
        r: int = 3,
        decoder_type='Decoder',
        adapter_type=None,
        backbone_type='dinov3_vitl16',
        # lora_layers=None,
    ):
        super().__init__()

        dinov3_vits_dict = {
            "dinov3_vits16": dinov3_vits16,
            "dinov3_vits16plus": dinov3_vits16plus,
            "dinov3_vitb16": dinov3_vitb16,
            "dinov3_vitl16": dinov3_vitl16,
            "dinov3_vit7b16": dinov3_vit7b16
        }
        dinov3_vit = dinov3_vits_dict[backbone_type]
        self.backbone_type = backbone_type
        if backbone_weights is not None:
            self.backbone = dinov3_vit(weights=backbone_weights,
                                       pretrained=True)
        else:
            self.backbone = dinov3_vit(pretrained=False)

        # Important: we freeze the backbone
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        embed_dim = self.backbone.embed_dim

        self.FEM = None
        if adapter_type == 'FEM':
            self.FEM = FeatureEnhancementModule(embed_dim)

        # 根据类型选择解码器
        decoder_kwargs = {
            "n_classes": n_classes,
        }
        if adapter_type is None:
            # decoder_kwargs["in_channels"] = [embed_dim] * 4
            self.conv1 = nn.Conv2d(1024, 256,   kernel_size=1)
            self.conv2 = nn.Conv2d(1024, 512, kernel_size=1)
        if decoder_type == 'LinearHead':
            self.decoder = LinearHead(in_ch=embed_dim, n_classes=n_classes)
        elif decoder_type == 'Decoder_ONLYFEM':
            self.decoder = Decoder_FEM(**decoder_kwargs)
        elif decoder_type == 'Decoder_ONLYCSFM':
            self.decoder = Decoder_CSFM(**decoder_kwargs)
        elif decoder_type == 'Decoder_FEM_CSFM':
            self.decoder = Decoder_FEM_CSFM(**decoder_kwargs)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Add LoRA layers to the encoder
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_layers = list(range(len(self.backbone.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.backbone.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, *modalities):
        if len(modalities) == 0:
            raise ValueError("At least one modality must be provided")

        # 主输入x
        x = modalities[0]
        _, C, H, W = x.shape
        patch_h, patch_w = x.shape[-2] // 16, x.shape[-1] // 16

        scale_factors = [4, 2, 1, 0.5]

        if self.FEM is not None:
            outputs = self.backbone.get_intermediate_layers(
                x, n=BACKBONE_INTERMEDIATE_LAYERS[self.backbone_type])
            # 使用适配器处理多尺度特征
            multi_scale_features = self.FEM(outputs,
                                                patch_h=patch_h,
                                                patch_w=patch_w)
        else:
            outputs = self.backbone.get_intermediate_layers(
                x,
                n=BACKBONE_INTERMEDIATE_LAYERS[self.backbone_type],
                reshape=True)
            # 直接处理中间层输出
            multi_scale_features = []
            for i, output in enumerate(outputs):
                if i < len(scale_factors):

                    output = F.interpolate(output,
                                           scale_factor=scale_factors[i],
                                           mode="bilinear",
                                           align_corners=False)
                    if i == 0:
                        output = self.conv1(output)
                    if i == 1:
                        output = self.conv2(output)
                multi_scale_features.append(output)

        logits = self.decoder(multi_scale_features)


        _H, _W = logits.shape[2:]
        if _H != H or _W != W:
            # 确保输出大小与输入一致
            pred = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
            )
        else:
            pred = logits
        return pred


class ResNetSegmentModule(nn.Module):

    def __init__(
        self,
        n_classes: int = 1000,
        use_lora: bool = False,
        r: int = 3,
        decoder_type='Decoder',
        num_modalities: int = 1,
    ):
        super().__init__()

        self.backbone = ResNet50(pretrained=True)

        embed_dim = [256, 512, 1024, 2048]

        # 根据类型选择解码器
        decoder_kwargs = {
            "in_channels": embed_dim,
            "n_classes": n_classes,
            "num_modalities": num_modalities
        }

        if decoder_type == 'LinearHead':
            self.decoder = LinearHead(in_ch=embed_dim, n_classes=n_classes)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

        # Add LoRA layers to the encoder
        self.use_lora = use_lora
        if self.use_lora:
            self.lora_layers = list(range(len(self.backbone.blocks)))
            self.w_a = []
            self.w_b = []

            for i, block in enumerate(self.backbone.blocks):
                if i not in self.lora_layers:
                    continue
                w_qkv_linear = block.attn.qkv
                dim = w_qkv_linear.in_features

                w_a_linear_q, w_b_linear_q = self._create_lora_layer(dim, r)
                w_a_linear_v, w_b_linear_v = self._create_lora_layer(dim, r)

                self.w_a.extend([w_a_linear_q, w_a_linear_v])
                self.w_b.extend([w_b_linear_q, w_b_linear_v])

                block.attn.qkv = LoRA(
                    w_qkv_linear,
                    w_a_linear_q,
                    w_b_linear_q,
                    w_a_linear_v,
                    w_b_linear_v,
                )
            self._reset_lora_parameters()

    def _create_lora_layer(self, dim: int, r: int):
        w_a = nn.Linear(dim, r, bias=False)
        w_b = nn.Linear(r, dim, bias=False)
        return w_a, w_b

    def _reset_lora_parameters(self) -> None:
        for w_a in self.w_a:
            nn.init.kaiming_uniform_(w_a.weight, a=math.sqrt(5))
        for w_b in self.w_b:
            nn.init.zeros_(w_b.weight)

    def forward(self, *modalities):
        if len(modalities) == 0:
            raise ValueError("At least one modality must be provided")

        # 主输入x
        x = modalities[0]
        _, C, H, W = x.shape

        if len(modalities) == 1:
            outputs = self.backbone(x)

            logits = self.decoder(outputs)

        else:
            outputs_modalities = []
            for idx, modality_input in enumerate(modalities):
                if modality_input.shape[1] != C and idx > 0:
                    modality_input = modality_input.repeat(1, C, 1, 1)

                outputs_modality = self.backbone(modality_input)
                outputs_modalities.append(outputs_modality)

            # 将处理后的所有模态特征传递给解码器
            logits = self.decoder(*outputs_modalities)

        _H, _W = logits.shape[2:]
        if _H != H or _W != W:
            # 确保输出大小与输入一致
            pred = F.interpolate(
                logits,
                size=(H, W),
                mode="bilinear",
            )

        return pred


def build_model(
    model_name=None,
    backbone_weights=None,
    n_classes: int = 1000,
    use_lora: bool = False,
    r: int = 3,
    **kwargs,
):
    if model_name == 'DINOv3_ONLYFEM':
        model = DINOSegmentModule(
            backbone_weights=backbone_weights,
            n_classes=n_classes,
            use_lora=use_lora,
            r=r,
            adapter_type="FEM",
            decoder_type="Decoder_ONLYFEM",
            **kwargs,
        )

    if model_name == 'DINOv3_ONLYCSFM':
        model = DINOSegmentModule(
            backbone_weights=backbone_weights,
            n_classes=n_classes,
            use_lora=use_lora,
            r=r,
            adapter_type=None,
            decoder_type="Decoder_ONLYCSFM",
            **kwargs,
        )
    if model_name == 'DINOv3_FEM_CSFM':
        model = DINOSegmentModule(
            backbone_weights=backbone_weights,
            n_classes=n_classes,
            use_lora=use_lora,
            r=r,
            adapter_type="FEM",
            decoder_type="Decoder_FEM_CSFM",
            **kwargs,
        )
    elif model_name == 'DINOv3_Baseline':
        model = DINOSegmentModule(backbone_weights=backbone_weights,
                                  n_classes=n_classes,
                                  use_lora=use_lora,
                                  r=r,
                                  decoder_type="LinearHead",
                                  **kwargs)
    else:
        model = None

    return model
