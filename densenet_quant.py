"""
Inspired by:
    - https://arxiv.org/abs/1608.06993
    - https://github.com/liuzhuang13/DenseNet
    - https://github.com/pytorch/vision/blob/main/torchvision/models/densenet.py
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QDenseLayer(nn.Module):
    """
    input → BN1 → ReLU → Conv1(1×1) → BN2 → ReLU → Conv2(3×3) → output
    """

    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # Bottleneck 1×1
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False,
        )

        # 3×3 conv
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.drop_rate = drop_rate

        self.quantize = False
        self.bits = 8

        self.register_buffer("input_scale", None)
        self.register_buffer("input_offset", None)
        self.register_buffer("mid_scale", None)
        self.register_buffer("mid_offset", None)
        self.register_buffer("w1_scale", None)
        self.register_buffer("w2_scale", None)

        self.cache_mode = False
        self.reset_calibration_stats()

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(inputs, dim=1)

        out = self.bn1(x)
        out = self.relu1(out)

        if self.cache_mode:
            self.input_min = min(self.input_min, out.min().item())
            self.input_max = max(self.input_max, out.max().item())

        if self.quantize and self.input_scale is not None:
            out = self._quantize_activation(out, self.input_scale, self.input_offset)
            w1 = self._quantize_weights(self.conv1.weight, self.w1_scale)
            out = F.conv2d(out, w1, None, stride=1)
        else:
            out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)

        if self.cache_mode:
            self.mid_min = min(self.mid_min, out.min().item())
            self.mid_max = max(self.mid_max, out.max().item())

        if self.quantize and self.mid_scale is not None:
            out = self._quantize_activation(out, self.mid_scale, self.mid_offset)
            w2 = self._quantize_weights(self.conv2.weight, self.w2_scale)
            out = F.conv2d(out, w2, None, stride=1, padding=1)
        else:
            out = self.conv2(out)

        if self.drop_rate > 0 and self.training:
            out = F.dropout(out, p=self.drop_rate)

        return out

    def _quantize_activation(
        self, x: torch.Tensor, scale: torch.Tensor, offset: torch.Tensor
    ) -> torch.Tensor:
        """Asymmetric"""
        q_min = 0
        q_max = 2**self.bits - 1
        return torch.fake_quantize_per_tensor_affine(
            x, scale.item(), offset.item(), q_min, q_max
        )

    def _quantize_weights(self, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Symmetric"""
        q_min = -(2 ** (self.bits - 1))
        q_max = 2 ** (self.bits - 1) - 1
        return torch.fake_quantize_per_tensor_affine(w, scale.item(), 0, q_min, q_max)

    def reset_calibration_stats(self):
        self.input_min = float("inf")
        self.input_max = float("-inf")
        self.mid_min = float("inf")
        self.mid_max = float("-inf")

    def calibrate(self):
        if self.input_min == float("inf"):
            return

        device = self.conv1.weight.device
        qmax = 2**self.bits - 1

        # Input quantization params
        input_scale = max((self.input_max - self.input_min) / qmax, 1e-8)
        input_offset = int(round(-self.input_min / input_scale))

        self.input_scale = torch.tensor([input_scale], device=device)
        self.input_offset = torch.tensor([input_offset], device=device)

        # Mid quantization params
        mid_scale = max((self.mid_max - self.mid_min) / qmax, 1e-8)
        mid_offset = int(round(-self.mid_min / mid_scale))

        self.mid_scale = torch.tensor([mid_scale], device=device)
        self.mid_offset = torch.tensor([mid_offset], device=device)

        # Weight quantization params
        w_qmax = 2 ** (self.bits - 1) - 1
        self.w1_scale = torch.tensor(
            [self.conv1.weight.abs().max().item() / w_qmax], device=device
        )
        self.w2_scale = torch.tensor(
            [self.conv2.weight.abs().max().item() / w_qmax], device=device
        )

    def enable_quantization(self, bits: int = 8):
        self.quantize = True
        self.bits = bits
        self.calibrate()

    def disable_quantization(self):
        self.quantize = False


class QDenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            layer = QDenseLayer(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.layers.append(layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]

        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)

        return torch.cat(features, dim=1)

    def enable_cache_mode(self):
        for layer in self.layers:
            layer.cache_mode = True

    def disable_cache_mode(self):
        for layer in self.layers:
            layer.cache_mode = False

    def reset_calibration_stats(self):
        for layer in self.layers:
            layer.reset_calibration_stats()

    def calibrate_all(self):
        for layer in self.layers:
            layer.calibrate()

    def enable_quantization(self, bits: int = 8):
        for layer in self.layers:
            layer.enable_quantization(bits)

    def disable_quantization(self):
        for layer in self.layers:
            layer.disable_quantization()


class QTransition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(
            num_input_features, num_output_features, kernel_size=1, stride=1, bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        self.quantize = False
        self.bits = 8
        self.register_buffer("input_scale", None)
        self.register_buffer("input_offset", None)
        self.register_buffer("w_scale", None)

        self.cache_mode = False
        self.reset_calibration_stats()

    def reset_calibration_stats(self):
        self.input_min = float("inf")
        self.input_max = float("-inf")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = self.relu(out)

        if self.cache_mode:
            self.input_min = min(self.input_min, out.min().item())
            self.input_max = max(self.input_max, out.max().item())

        if self.quantize and self.input_scale is not None:
            out = torch.fake_quantize_per_tensor_affine(
                out,
                self.input_scale.item(),
                self.input_offset.item(),
                0,
                2**self.bits - 1,
            )
            w = torch.fake_quantize_per_tensor_affine(
                self.conv.weight,
                self.w_scale.item(),
                0,
                -(2 ** (self.bits - 1)),
                2 ** (self.bits - 1) - 1,
            )
            out = F.conv2d(out, w, None, stride=1)
        else:
            out = self.conv(out)

        out = self.pool(out)
        return out

    def calibrate(self):
        if self.input_min == float("inf"):
            return

        device = self.conv.weight.device
        qmax = 2**self.bits - 1

        input_scale = max((self.input_max - self.input_min) / qmax, 1e-8)
        input_offset = int(round(-self.input_min / input_scale))

        self.input_scale = torch.tensor([input_scale], device=device)
        self.input_offset = torch.tensor([input_offset], device=device)
        self.w_scale = torch.tensor(
            [self.conv.weight.abs().max().item() / (2 ** (self.bits - 1) - 1)],
            device=device,
        )

    def enable_quantization(self, bits: int = 8):
        self.quantize = True
        self.bits = bits
        self.calibrate()

    def disable_quantization(self):
        self.quantize = False


class MyQDenseNet(nn.Module):
    """
    CIFAR-specific DenseNet (see https://arxiv.org/abs/1608.06993)
    """

    def __init__(
        self,
        growth_rate: int = 12,
        block_config: Tuple[int, ...] = (16, 16, 16),
        num_init_features: Optional[int] = None,
        bn_size: int = 4,
        drop_rate: float = 0.2,
        num_classes: int = 10,
        reduction: float = 0.5,
    ):
        super().__init__()

        if num_init_features is None:
            num_init_features = 2 * growth_rate

        self.reduction = reduction

        self.conv0 = nn.Conv2d(
            3, num_init_features, kernel_size=3, padding=1, bias=False
        )

        self.first_quantize = False
        self.bits = 8
        self.register_buffer("first_input_scale", None)
        self.register_buffer("first_input_offset", None)
        self.register_buffer("first_w_scale", None)

        self.cache_mode = False
        self.first_input_min = float("inf")
        self.first_input_max = float("-inf")

        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = QDenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                out_features = int(num_features * reduction)
                trans = QTransition(num_features, out_features)
                self.transitions.append(trans)
                num_features = out_features

        # BN → ReLU → GlobalAvgPool → Linear
        self.bn_final = nn.BatchNorm2d(num_features)
        self.relu_final = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(num_features, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cache_mode:
            self.first_input_min = min(self.first_input_min, x.min().item())
            self.first_input_max = max(self.first_input_max, x.max().item())

        if self.first_quantize and self.first_input_scale is not None:
            x_q = torch.fake_quantize_per_tensor_affine(
                x,
                self.first_input_scale.item(),
                self.first_input_offset.item(),
                0,
                2**self.bits - 1,
            )
            w_q = torch.fake_quantize_per_tensor_affine(
                self.conv0.weight,
                self.first_w_scale.item(),
                0,
                -(2 ** (self.bits - 1)),
                2 ** (self.bits - 1) - 1,
            )
            out = F.conv2d(x_q, w_q, None, padding=1)
        else:
            out = self.conv0(x)

        for i, block in enumerate(self.blocks):
            out = block(out)
            if i < len(self.transitions):
                out = self.transitions[i](out)

        out = self.bn_final(out)
        out = self.relu_final(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def enable_cache_mode(self):
        self.cache_mode = True
        for block in self.blocks:
            block.enable_cache_mode()
        for trans in self.transitions:
            trans.cache_mode = True

    def disable_cache_mode(self):
        self.cache_mode = False
        for block in self.blocks:
            block.disable_cache_mode()
        for trans in self.transitions:
            trans.cache_mode = False

    def calibrate(self):
        self._calibrate_first_layer()

        for block in self.blocks:
            block.calibrate_all()
        for trans in self.transitions:
            trans.calibrate()

    def enable_quantization(self, bits: int = 8):
        self.bits = bits
        self.first_quantize = True
        self._calibrate_first_layer()
        for block in self.blocks:
            block.enable_quantization(bits)
        for trans in self.transitions:
            trans.enable_quantization(bits)

    def _calibrate_first_layer(self):
        if self.first_input_min == float("inf"):
            return

        qmax = 2**self.bits - 1
        input_scale = max((self.first_input_max - self.first_input_min) / qmax, 1e-8)
        input_offset = int(round(-self.first_input_min / input_scale))

        device = self.conv0.weight.device
        self.first_input_scale = torch.tensor([input_scale], device=device)
        self.first_input_offset = torch.tensor([input_offset], device=device)
        self.first_w_scale = torch.tensor(
            [self.conv0.weight.abs().max().item() / (2 ** (self.bits - 1) - 1)],
            device=device,
        )

    def disable_quantization(self):
        self.first_quantize = False
        for block in self.blocks:
            block.disable_quantization()
        for trans in self.transitions:
            trans.disable_quantization()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def densenet_bc_100_12(num_classes: int = 10, **kwargs) -> MyQDenseNet:
    """DenseNet-BC (L=100, k=12) для CIFAR-10"""
    return MyQDenseNet(
        growth_rate=12,
        block_config=(16, 16, 16),
        num_init_features=24,
        num_classes=num_classes,
        **kwargs,
    )


def densenet_bc_250_24(num_classes: int = 10, **kwargs) -> MyQDenseNet:
    """DenseNet-BC (L=250, k=24) для CIFAR-10"""
    return MyQDenseNet(
        growth_rate=24,
        block_config=(41, 41, 41),
        num_init_features=48,
        num_classes=num_classes,
        **kwargs,
    )


def densenet_bc_190_40(num_classes: int = 10, **kwargs) -> MyQDenseNet:
    """DenseNet-BC (L=190, k=40) для CIFAR-10 — лучший результат из статьи"""
    return MyQDenseNet(
        growth_rate=40,
        block_config=(31, 31, 31),
        num_init_features=80,
        num_classes=num_classes,
        **kwargs,
    )
