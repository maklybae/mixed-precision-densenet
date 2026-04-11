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
from torch.utils.checkpoint import checkpoint

from dq_quantizer import init_weight_quantizer, init_activation_quantizer, UniformQuantU3


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
        self.w_bins: Optional[int] = None
        self.a_bins: Optional[int] = None

        self.register_buffer("input_scale", None)
        self.register_buffer("input_offset", None)
        self.register_buffer("mid_scale", None)
        self.register_buffer("mid_offset", None)
        self.register_buffer("w1_scale", None)
        self.register_buffer("w2_scale", None)

        self.cache_mode = False
        self.reset_calibration_stats()

        # DQ (differentiable quantization) state
        self.dq_enabled = False
        self.dq_w1: Optional[nn.Module] = None
        self.dq_w2: Optional[nn.Module] = None
        self.dq_input_act: Optional[nn.Module] = None
        self.dq_mid_act: Optional[nn.Module] = None

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = torch.cat(inputs, dim=1)

        out = self.bn1(x)
        out = self.relu1(out)

        if self.cache_mode:
            batch_min = out.min().item()
            batch_max = out.max().item()
            self.input_min = min(self.input_min, batch_min)
            self.input_max = max(self.input_max, batch_max)
            self.input_mins_list.append(batch_min)
            self.input_maxs_list.append(batch_max)

        if self.dq_enabled and self.dq_input_act is not None:
            out = self.dq_input_act(out)
            w1 = self.dq_w1(self.conv1.weight)
            out = F.conv2d(out, w1, None, stride=1)
        elif self.quantize and self.input_scale is not None:
            out = self._quantize_activation(out, self.input_scale, self.input_offset)
            w1 = self._quantize_weights(self.conv1.weight, self.w1_scale)
            out = F.conv2d(out, w1, None, stride=1)
        else:
            out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)

        if self.cache_mode:
            batch_min = out.min().item()
            batch_max = out.max().item()
            self.mid_min = min(self.mid_min, batch_min)
            self.mid_max = max(self.mid_max, batch_max)
            self.mid_mins_list.append(batch_min)
            self.mid_maxs_list.append(batch_max)

        if self.dq_enabled and self.dq_mid_act is not None:
            out = self.dq_mid_act(out)
            w2 = self.dq_w2(self.conv2.weight)
            out = F.conv2d(out, w2, None, stride=1, padding=1)
        elif self.quantize and self.mid_scale is not None:
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
        """Asymmetric quantization for activations."""
        if self.a_bins is not None:
            q_min = 0
            q_max = self.a_bins - 1
        else:
            q_min = 0
            q_max = 2**self.bits - 1
        return torch.fake_quantize_per_tensor_affine(
            x, scale.item(), int(offset.item()), q_min, q_max
        )

    def _quantize_weights(self, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """Symmetric quantization for weights."""
        if self.w_bins is not None:
            half = self.w_bins // 2
            q_min = -half
            q_max = half
        else:
            q_min = -(2 ** (self.bits - 1))
            q_max = 2 ** (self.bits - 1) - 1
        return torch.fake_quantize_per_tensor_affine(w, scale.item(), 0, q_min, q_max)

    def reset_calibration_stats(self):
        self.input_min = float("inf")
        self.input_max = float("-inf")
        self.mid_min = float("inf")
        self.mid_max = float("-inf")
        self.input_mins_list: list[float] = []
        self.input_maxs_list: list[float] = []
        self.mid_mins_list: list[float] = []
        self.mid_maxs_list: list[float] = []

    def calibrate(self):
        if self.input_min == float("inf"):
            return

        device = self.conv1.weight.device

        # Activation bins
        if self.a_bins is not None:
            a_qmax = self.a_bins - 1
        else:
            a_qmax = 2**self.bits - 1

        # Input quantization params
        input_scale = max((self.input_max - self.input_min) / a_qmax, 1e-8)
        input_offset = int(round(-self.input_min / input_scale))

        self.input_scale = torch.tensor([input_scale], device=device)
        self.input_offset = torch.tensor([input_offset], device=device)

        # Mid quantization params
        mid_scale = max((self.mid_max - self.mid_min) / a_qmax, 1e-8)
        mid_offset = int(round(-self.mid_min / mid_scale))

        self.mid_scale = torch.tensor([mid_scale], device=device)
        self.mid_offset = torch.tensor([mid_offset], device=device)

        # Weight quantization params
        if self.w_bins is not None:
            w_qmax = self.w_bins // 2
        else:
            w_qmax = 2 ** (self.bits - 1) - 1
        self.w1_scale = torch.tensor(
            [max(self.conv1.weight.abs().max().item() / w_qmax, 1e-8)], device=device
        )
        self.w2_scale = torch.tensor(
            [max(self.conv2.weight.abs().max().item() / w_qmax, 1e-8)], device=device
        )

    def set_quant_params(
        self,
        input_scale: torch.Tensor,
        input_offset: torch.Tensor,
        mid_scale: torch.Tensor,
        mid_offset: torch.Tensor,
        w1_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w_bins: int,
        a_bins: int,
    ):
        """Set externally computed quantization parameters."""
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.mid_scale = mid_scale
        self.mid_offset = mid_offset
        self.w1_scale = w1_scale
        self.w2_scale = w2_scale
        self.w_bins = w_bins
        self.a_bins = a_bins

    def enable_quantization(self, bits: int = 8):
        self.quantize = True
        self.bits = bits
        self.w_bins = None
        self.a_bins = None
        self.calibrate()

    def enable_quantization_bins(self, w_bins: int, a_bins: int):
        """Enable quantization using bin counts (supports non-integer bitwidths)."""
        self.quantize = True
        self.w_bins = w_bins
        self.a_bins = a_bins
        self.calibrate()

    def disable_quantization(self):
        self.quantize = False

    def enable_dq(self, init_bitwidth: int = 4):
        self.dq_enabled = True
        self.quantize = False  # disable PTQ

        self.dq_w1 = init_weight_quantizer(
            self.conv1.weight, init_bitwidth=init_bitwidth,
        )
        self.dq_w2 = init_weight_quantizer(
            self.conv2.weight, init_bitwidth=init_bitwidth,
        )

        act_min = self.input_min if self.input_min != float("inf") else 0.0
        act_max = self.input_max if self.input_max != float("-inf") else 1.0
        self.dq_input_act = init_activation_quantizer(
            act_min=act_min, act_max=act_max, init_bitwidth=init_bitwidth,
        )

        mid_min = self.mid_min if self.mid_min != float("inf") else 0.0
        mid_max = self.mid_max if self.mid_max != float("-inf") else 1.0
        self.dq_mid_act = init_activation_quantizer(
            act_min=mid_min, act_max=mid_max, init_bitwidth=init_bitwidth,
        )

    def disable_dq(self):
        """Disable DQ, switch back to float mode."""
        self.dq_enabled = False


class QDenseBlock(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False,
    ):
        super().__init__()
        self.memory_efficient = memory_efficient
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
            if self.memory_efficient and any(f.requires_grad for f in features):
                new_features = checkpoint(layer, *features, use_reentrant=False)
            else:
                new_features = layer(*features)
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

    def enable_quantization_bins(self, w_bins: int, a_bins: int):
        for layer in self.layers:
            layer.enable_quantization_bins(w_bins, a_bins)

    def disable_quantization(self):
        for layer in self.layers:
            layer.disable_quantization()

    def enable_dq(self, init_bitwidth: int = 4):
        for layer in self.layers:
            layer.enable_dq(init_bitwidth)

    def disable_dq(self):
        for layer in self.layers:
            layer.disable_dq()


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
        self.w_bins: Optional[int] = None
        self.a_bins: Optional[int] = None
        self.register_buffer("input_scale", None)
        self.register_buffer("input_offset", None)
        self.register_buffer("w_scale", None)

        self.cache_mode = False
        self.reset_calibration_stats()

        # DQ (differentiable quantization) state
        self.dq_enabled = False
        self.dq_w: Optional[nn.Module] = None
        self.dq_input_act: Optional[nn.Module] = None

    def reset_calibration_stats(self):
        self.input_min = float("inf")
        self.input_max = float("-inf")
        self.input_mins_list: list[float] = []
        self.input_maxs_list: list[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn(x)
        out = self.relu(out)

        if self.cache_mode:
            batch_min = out.min().item()
            batch_max = out.max().item()
            self.input_min = min(self.input_min, batch_min)
            self.input_max = max(self.input_max, batch_max)
            self.input_mins_list.append(batch_min)
            self.input_maxs_list.append(batch_max)

        if self.dq_enabled and self.dq_input_act is not None:
            out = self.dq_input_act(out)
            w = self.dq_w(self.conv.weight)
            out = F.conv2d(out, w, None, stride=1)
        elif self.quantize and self.input_scale is not None:
            # Activation quantization
            if self.a_bins is not None:
                a_qmin, a_qmax = 0, self.a_bins - 1
            else:
                a_qmin, a_qmax = 0, 2**self.bits - 1
            out = torch.fake_quantize_per_tensor_affine(
                out,
                self.input_scale.item(),
                int(self.input_offset.item()),
                a_qmin,
                a_qmax,
            )
            # Weight quantization
            if self.w_bins is not None:
                half = self.w_bins // 2
                w_qmin, w_qmax = -half, half
            else:
                w_qmin = -(2 ** (self.bits - 1))
                w_qmax = 2 ** (self.bits - 1) - 1
            w = torch.fake_quantize_per_tensor_affine(
                self.conv.weight,
                self.w_scale.item(),
                0,
                w_qmin,
                w_qmax,
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

        if self.a_bins is not None:
            a_qmax = self.a_bins - 1
        else:
            a_qmax = 2**self.bits - 1

        input_scale = max((self.input_max - self.input_min) / a_qmax, 1e-8)
        input_offset = int(round(-self.input_min / input_scale))

        self.input_scale = torch.tensor([input_scale], device=device)
        self.input_offset = torch.tensor([input_offset], device=device)

        if self.w_bins is not None:
            w_qmax = self.w_bins // 2
        else:
            w_qmax = 2 ** (self.bits - 1) - 1
        self.w_scale = torch.tensor(
            [max(self.conv.weight.abs().max().item() / w_qmax, 1e-8)],
            device=device,
        )

    def set_quant_params(
        self,
        input_scale: torch.Tensor,
        input_offset: torch.Tensor,
        w_scale: torch.Tensor,
        w_bins: int,
        a_bins: int,
    ):
        """Set externally computed quantization parameters."""
        self.input_scale = input_scale
        self.input_offset = input_offset
        self.w_scale = w_scale
        self.w_bins = w_bins
        self.a_bins = a_bins

    def enable_quantization(self, bits: int = 8):
        self.quantize = True
        self.bits = bits
        self.w_bins = None
        self.a_bins = None
        self.calibrate()

    def enable_quantization_bins(self, w_bins: int, a_bins: int):
        """Enable quantization using bin counts."""
        self.quantize = True
        self.w_bins = w_bins
        self.a_bins = a_bins
        self.calibrate()

    def disable_quantization(self):
        self.quantize = False

    def enable_dq(self, init_bitwidth: int = 4):
        self.dq_enabled = True
        self.quantize = False

        self.dq_w = init_weight_quantizer(
            self.conv.weight, init_bitwidth=init_bitwidth,
        )

        act_min = self.input_min if self.input_min != float("inf") else 0.0
        act_max = self.input_max if self.input_max != float("-inf") else 1.0
        self.dq_input_act = init_activation_quantizer(
            act_min=act_min, act_max=act_max, init_bitwidth=init_bitwidth,
        )

    def disable_dq(self):
        """Disable DQ, switch back to float mode."""
        self.dq_enabled = False


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
        memory_efficient: bool = False,
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
        self.first_w_bins: Optional[int] = None
        self.first_a_bins: Optional[int] = None
        self.register_buffer("first_input_scale", None)
        self.register_buffer("first_input_offset", None)
        self.register_buffer("first_w_scale", None)

        self.cache_mode = False
        self.first_input_min = float("inf")
        self.first_input_max = float("-inf")
        self.first_input_mins_list: list[float] = []
        self.first_input_maxs_list: list[float] = []

        # DQ (differentiable quantization) state for first layer
        self.dq_first_enabled = False
        self.dq_first_w: Optional[nn.Module] = None

        # DQ state for classifier
        self.dq_classifier_w: Optional[nn.Module] = None
        self.dq_final_act: Optional[nn.Module] = None

        # Cache stats for final activation (after relu_final)
        self.final_act_min = float("inf")
        self.final_act_max = float("-inf")

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
                memory_efficient=memory_efficient,
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
            batch_min = x.min().item()
            batch_max = x.max().item()
            self.first_input_min = min(self.first_input_min, batch_min)
            self.first_input_max = max(self.first_input_max, batch_max)
            self.first_input_mins_list.append(batch_min)
            self.first_input_maxs_list.append(batch_max)

        if self.dq_first_enabled and self.dq_first_w is not None:
            w_q = self.dq_first_w(self.conv0.weight)
            out = F.conv2d(x, w_q, None, padding=1)
        elif self.first_quantize and self.first_input_scale is not None:
            # Activation quantization
            if self.first_a_bins is not None:
                a_qmin, a_qmax = 0, self.first_a_bins - 1
            else:
                a_qmin, a_qmax = 0, 2**self.bits - 1
            x_q = torch.fake_quantize_per_tensor_affine(
                x,
                self.first_input_scale.item(),
                int(self.first_input_offset.item()),
                a_qmin,
                a_qmax,
            )
            # Weight quantization
            if self.first_w_bins is not None:
                half = self.first_w_bins // 2
                w_qmin, w_qmax = -half, half
            else:
                w_qmin = -(2 ** (self.bits - 1))
                w_qmax = 2 ** (self.bits - 1) - 1
            w_q = torch.fake_quantize_per_tensor_affine(
                self.conv0.weight,
                self.first_w_scale.item(),
                0,
                w_qmin,
                w_qmax,
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

        if self.cache_mode:
            self.final_act_min = min(self.final_act_min, out.min().item())
            self.final_act_max = max(self.final_act_max, out.max().item())

        if self.dq_first_enabled and self.dq_final_act is not None:
            out = self.dq_final_act(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)

        if self.dq_first_enabled and self.dq_classifier_w is not None:
            w_q = self.dq_classifier_w(self.classifier.weight)
            out = F.linear(out, w_q, self.classifier.bias)
        else:
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

    def reset_all_calibration_stats(self):
        """Reset calibration statistics for all layers."""
        self.first_input_min = float("inf")
        self.first_input_max = float("-inf")
        self.first_input_mins_list = []
        self.first_input_maxs_list = []
        self.final_act_min = float("inf")
        self.final_act_max = float("-inf")
        for block in self.blocks:
            block.reset_calibration_stats()
        for trans in self.transitions:
            trans.reset_calibration_stats()

    def enable_quantization(self, bits: int = 8):
        self.bits = bits
        self.first_quantize = True
        self.first_w_bins = None
        self.first_a_bins = None
        self._calibrate_first_layer()
        for block in self.blocks:
            block.enable_quantization(bits)
        for trans in self.transitions:
            trans.enable_quantization(bits)

    def enable_quantization_bins(
        self, w_bins: int, a_bins: int, skip_first_last: bool = True
    ):
        """Enable quantization with bin counts. Supports non-integer bitwidths.

        Args:
            w_bins: Number of weight quantization bins.
            a_bins: Number of activation quantization bins.
            skip_first_last: If True, conv0 and classifier remain in float32.
        """
        if not skip_first_last:
            self.first_quantize = True
            self.first_w_bins = w_bins
            self.first_a_bins = a_bins
            self._calibrate_first_layer()
        else:
            self.first_quantize = False

        for block in self.blocks:
            block.enable_quantization_bins(w_bins, a_bins)
        for trans in self.transitions:
            trans.enable_quantization_bins(w_bins, a_bins)

    def _calibrate_first_layer(self):
        if self.first_input_min == float("inf"):
            return

        device = self.conv0.weight.device

        # Activation range
        if self.first_a_bins is not None:
            a_qmax = self.first_a_bins - 1
        else:
            a_qmax = 2**self.bits - 1
        input_scale = max(
            (self.first_input_max - self.first_input_min) / a_qmax, 1e-8
        )
        input_offset = int(round(-self.first_input_min / input_scale))

        self.first_input_scale = torch.tensor([input_scale], device=device)
        self.first_input_offset = torch.tensor([input_offset], device=device)

        # Weight range
        if self.first_w_bins is not None:
            w_qmax = self.first_w_bins // 2
        else:
            w_qmax = 2 ** (self.bits - 1) - 1
        self.first_w_scale = torch.tensor(
            [max(self.conv0.weight.abs().max().item() / w_qmax, 1e-8)],
            device=device,
        )

    def set_first_layer_params(
        self,
        input_scale: torch.Tensor,
        input_offset: torch.Tensor,
        w_scale: torch.Tensor,
        w_bins: int,
        a_bins: int,
    ):
        """Set externally computed quantization parameters for the first layer."""
        self.first_input_scale = input_scale
        self.first_input_offset = input_offset
        self.first_w_scale = w_scale
        self.first_w_bins = w_bins
        self.first_a_bins = a_bins

    def disable_quantization(self):
        self.first_quantize = False
        for block in self.blocks:
            block.disable_quantization()
        for trans in self.transitions:
            trans.disable_quantization()

    def enable_dq(self, init_bitwidth: int = 4):
        self.dq_first_enabled = True
        self.first_quantize = False  # disable PTQ

        # First layer: only quantize weights (raw input is NOT quantized per paper)
        self.dq_first_w = init_weight_quantizer(
            self.conv0.weight, init_bitwidth=init_bitwidth,
        )

        # Classifier weight quantizer (paper: "we quantize all layers")
        self.dq_classifier_w = init_weight_quantizer(
            self.classifier.weight, init_bitwidth=init_bitwidth,
        )

        # Final activation quantizer (unsigned — after relu_final, before classifier)
        final_min = self.final_act_min if self.final_act_min != float("inf") else 0.0
        final_max = self.final_act_max if self.final_act_max != float("-inf") else 1.0
        self.dq_final_act = init_activation_quantizer(
            act_min=final_min, act_max=final_max, init_bitwidth=init_bitwidth,
        )

        for block in self.blocks:
            block.enable_dq(init_bitwidth)
        for trans in self.transitions:
            trans.enable_dq(init_bitwidth)

    def disable_dq(self):
        """Disable DQ, switch back to float mode."""
        self.dq_first_enabled = False
        for block in self.blocks:
            block.disable_dq()
        for trans in self.transitions:
            trans.disable_dq()

    def get_quantizer_params(self):
        params = []
        for module in self.modules():
            if isinstance(module, UniformQuantU3):
                params.extend(module.parameters())
        return params

    def get_network_params(self):
        """Return all non-quantizer parameters (conv weights, BN, classifier)."""
        quant_param_ids = {id(p) for p in self.get_quantizer_params()}
        return [p for p in self.parameters() if id(p) not in quant_param_ids]

    def get_bitwidths(self):
        info = {}
        for name, module in self.named_modules():
            if isinstance(module, UniformQuantU3):
                info[name] = {
                    "bitwidth": module.bitwidth(),
                    "soft_bitwidth": module.bitwidth_soft().item(),
                    "d": module.d.item(),
                    "q_max": module.q_max.item(),
                    "signed": module.signed,
                }
        return info

    def get_soft_bitwidths(self) -> torch.Tensor:
        bitwidths = []
        for module in self.modules():
            if isinstance(module, UniformQuantU3):
                bitwidths.append(module.bitwidth_soft())
        return torch.stack(bitwidths)

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


def densenet_bc_88_12(num_classes: int = 10, **kwargs) -> MyQDenseNet:
    """DenseNet-BC (L=88, k=12) для CIFAR-10"""
    return MyQDenseNet(
        growth_rate=12,
        block_config=(14, 14, 14),
        num_init_features=24,
        num_classes=num_classes,
        **kwargs,
    )


def densenet_bc_76_12(num_classes: int = 10, **kwargs) -> MyQDenseNet:
    """DenseNet-BC (L=76, k=12) для CIFAR-10"""
    return MyQDenseNet(
        growth_rate=12,
        block_config=(12, 12, 12),
        num_init_features=24,
        num_classes=num_classes,
        **kwargs,
    )


def densenet_bc_64_12(num_classes: int = 10, **kwargs) -> MyQDenseNet:
    """DenseNet-BC (L=64, k=12) для CIFAR-10"""
    return MyQDenseNet(
        growth_rate=12,
        block_config=(10, 10, 10),
        num_init_features=24,
        num_classes=num_classes,
        **kwargs,
    )


def densenet_bc_52_12(num_classes: int = 10, **kwargs) -> MyQDenseNet:
    """DenseNet-BC (L=52, k=12) для CIFAR-10"""
    return MyQDenseNet(
        growth_rate=12,
        block_config=(8, 8, 8),
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


MODEL_REGISTRY = {
    "densenet_bc_100_12": densenet_bc_100_12,
    "densenet_bc_88_12": densenet_bc_88_12,
    "densenet_bc_76_12": densenet_bc_76_12,
    "densenet_bc_64_12": densenet_bc_64_12,
    "densenet_bc_52_12": densenet_bc_52_12,
    "densenet_bc_250_24": densenet_bc_250_24,
    "densenet_bc_190_40": densenet_bc_190_40,
}
