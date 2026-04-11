"""
Post-Training Quantization utilities for DenseNet-BC models.

Provides:
- Bin/bit conversion helpers for non-integer bitwidths
- Percentile-based calibration (robust to outliers)
- PTQCalibrator class for full calibration pipeline
- End-to-end post-quantization fine-tuning
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers: bits <-> bins conversion
# ---------------------------------------------------------------------------

def bits_to_bins(bits: float, signed: bool = False) -> int:
    """Convert bitwidth to number of quantization bins.

    For unsigned (activations after ReLU): bins = round(2^bits)
    For signed symmetric (weights): bins = round(2^bits) - 1  (odd, so zero is centered)
    """
    raw = round(2 ** bits)
    if signed:
        # Ensure odd number for symmetric quantization
        bins = raw - 1
        if bins % 2 == 0:
            bins -= 1
        return max(bins, 1)
    return max(raw, 2)


def bins_to_bits(bins: int, signed: bool = False) -> float:
    """Convert number of quantization bins to effective bitwidth."""
    if signed:
        return math.log2(bins + 1)
    return math.log2(bins)


# ---------------------------------------------------------------------------
# Range computation
# ---------------------------------------------------------------------------

def compute_activation_range(
    act_min: float,
    act_max: float,
    method: str = "percentile",
    percentile: float = 0.9999,
    per_batch_maxs: Optional[list[float]] = None,
) -> tuple[float, float]:
    """Compute activation clipping range.

    Args:
        act_min: Global minimum observed during calibration.
        act_max: Global maximum observed during calibration.
        method: 'minmax' or 'percentile'.
        percentile: Quantile for percentile method (e.g. 0.9999).
        per_batch_maxs: List of per-batch max values (used for percentile).

    Returns:
        (range_min, range_max) clipped range.
    """
    if method == "minmax":
        return act_min, act_max

    # Percentile: for activations after ReLU, min is always >= 0
    range_min = max(act_min, 0.0)

    if per_batch_maxs and len(per_batch_maxs) >= 2:
        sorted_maxs = sorted(per_batch_maxs)
        idx = max(0, int(len(sorted_maxs) * percentile) - 1)
        range_max = sorted_maxs[idx]
    else:
        range_max = act_max

    return range_min, range_max


def compute_weight_range(
    weight: torch.Tensor,
    method: str = "percentile",
    percentile: float = 0.9999,
    symmetric: bool = True,
) -> tuple[float, float]:
    """Compute weight clipping range.

    Args:
        weight: Weight tensor.
        method: 'minmax' or 'percentile'.
        percentile: Quantile for clipping.
        symmetric: If True, range is [-a, a].

    Returns:
        (range_min, range_max).
    """
    if symmetric:
        if method == "minmax":
            a = weight.abs().max().item()
        else:
            a = torch.quantile(weight.abs().float(), percentile).item()
        return -a, a
    else:
        if method == "minmax":
            return weight.min().item(), weight.max().item()
        else:
            lo = torch.quantile(weight.float(), 1.0 - percentile).item()
            hi = torch.quantile(weight.float(), percentile).item()
            return lo, hi


# ---------------------------------------------------------------------------
# Quantization parameter computation
# ---------------------------------------------------------------------------

def compute_quant_params(
    range_min: float,
    range_max: float,
    bins: int,
    symmetric: bool = False,
) -> tuple[float, int, int, int]:
    """Compute scale, zero_point, q_min, q_max for fake_quantize.

    Args:
        range_min: Minimum of clipping range.
        range_max: Maximum of clipping range.
        bins: Number of quantization bins.
        symmetric: If True, uses symmetric signed scheme.

    Returns:
        (scale, zero_point, q_min, q_max).
    """
    if symmetric:
        # Signed symmetric: q_min = -(bins//2), q_max = bins//2
        half = bins // 2
        q_min = -half
        q_max = half
        a = max(abs(range_min), abs(range_max))
        scale = max(a / half, 1e-8)
        zero_point = 0
    else:
        # Unsigned asymmetric: q_min = 0, q_max = bins - 1
        q_min = 0
        q_max = bins - 1
        scale = max((range_max - range_min) / (bins - 1), 1e-8)
        zero_point = int(round(-range_min / scale))
        zero_point = max(q_min, min(q_max, zero_point))

    return scale, zero_point, q_min, q_max


# ---------------------------------------------------------------------------
# PTQCalibrator
# ---------------------------------------------------------------------------

@dataclass
class PTQConfig:
    """Configuration for PTQ calibration and quantization."""
    w_bins: int = 15
    a_bins: int = 16
    calibration_method: str = "percentile"  # 'minmax' or 'percentile'
    percentile: float = 0.9999
    num_calibration_batches: int = 50
    corrected_inputs: bool = True
    skip_first_last: bool = True


class PTQCalibrator:
    """Post-Training Quantization calibrator for MyQDenseNet.

    Performs calibration using percentile-based or min-max range estimation,
    then sets quantization parameters on the model.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        device: torch.device,
        config: PTQConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.config = config

    def _collect_stats(self, num_batches: int):
        """Run forward passes to collect activation statistics."""
        self.model.eval()
        self.model.enable_cache_mode()

        with torch.no_grad():
            for i, (inputs, _) in enumerate(
                tqdm(self.train_loader, desc="Calibrating", total=num_batches)
            ):
                if i >= num_batches:
                    break
                inputs = inputs.to(self.device)
                self.model(inputs)

        self.model.disable_cache_mode()

    def _apply_params_to_layer(self, layer, w_bins: int, a_bins: int):
        """Compute and set quantization parameters for a single quantizable layer.

        Works with QDenseLayer and QTransition.
        """
        cfg = self.config
        method = cfg.calibration_method
        percentile = cfg.percentile

        from densenet_quant import QDenseLayer, QTransition

        if isinstance(layer, QDenseLayer):
            # --- Input activation (after BN1 + ReLU1) ---
            a_min, a_max = compute_activation_range(
                layer.input_min, layer.input_max,
                method=method, percentile=percentile,
                per_batch_maxs=getattr(layer, "input_maxs_list", None),
            )
            a_scale, a_zp, a_qmin, a_qmax = compute_quant_params(
                a_min, a_max, a_bins, symmetric=False,
            )

            # --- Mid activation (after BN2 + ReLU2) ---
            m_min, m_max = compute_activation_range(
                layer.mid_min, layer.mid_max,
                method=method, percentile=percentile,
                per_batch_maxs=getattr(layer, "mid_maxs_list", None),
            )
            m_scale, m_zp, _, _ = compute_quant_params(
                m_min, m_max, a_bins, symmetric=False,
            )

            # --- Weights ---
            w1_min, w1_max = compute_weight_range(
                layer.conv1.weight.data, method=method, percentile=percentile,
            )
            w1_scale, _, _, _ = compute_quant_params(
                w1_min, w1_max, w_bins, symmetric=True,
            )

            w2_min, w2_max = compute_weight_range(
                layer.conv2.weight.data, method=method, percentile=percentile,
            )
            w2_scale, _, _, _ = compute_quant_params(
                w2_min, w2_max, w_bins, symmetric=True,
            )

            device = layer.conv1.weight.device
            layer.set_quant_params(
                input_scale=torch.tensor([a_scale], device=device),
                input_offset=torch.tensor([a_zp], device=device),
                mid_scale=torch.tensor([m_scale], device=device),
                mid_offset=torch.tensor([m_zp], device=device),
                w1_scale=torch.tensor([w1_scale], device=device),
                w2_scale=torch.tensor([w2_scale], device=device),
                w_bins=w_bins,
                a_bins=a_bins,
            )

        elif isinstance(layer, QTransition):
            # --- Input activation ---
            a_min, a_max = compute_activation_range(
                layer.input_min, layer.input_max,
                method=method, percentile=percentile,
                per_batch_maxs=getattr(layer, "input_maxs_list", None),
            )
            a_scale, a_zp, _, _ = compute_quant_params(
                a_min, a_max, a_bins, symmetric=False,
            )

            # --- Weights ---
            w_min, w_max = compute_weight_range(
                layer.conv.weight.data, method=method, percentile=percentile,
            )
            w_scale, _, _, _ = compute_quant_params(
                w_min, w_max, w_bins, symmetric=True,
            )

            device = layer.conv.weight.device
            layer.set_quant_params(
                input_scale=torch.tensor([a_scale], device=device),
                input_offset=torch.tensor([a_zp], device=device),
                w_scale=torch.tensor([w_scale], device=device),
                w_bins=w_bins,
                a_bins=a_bins,
            )

    def _apply_params_to_first_layer(self, model, w_bins: int, a_bins: int):
        """Compute and set quantization parameters for the first conv layer."""
        cfg = self.config
        method = cfg.calibration_method
        percentile = cfg.percentile

        # Input activation (raw image data, can be negative after normalization)
        a_min = model.first_input_min
        a_max = model.first_input_max
        per_batch_maxs = getattr(model, "first_input_maxs_list", None)

        a_min_c, a_max_c = a_min, a_max
        if method == "percentile" and per_batch_maxs and len(per_batch_maxs) >= 2:
            sorted_maxs = sorted(per_batch_maxs)
            idx = max(0, int(len(sorted_maxs) * percentile) - 1)
            a_max_c = sorted_maxs[idx]
            # For first layer input, min can be negative (normalized images)
            per_batch_mins = getattr(model, "first_input_mins_list", None)
            if per_batch_mins and len(per_batch_mins) >= 2:
                sorted_mins = sorted(per_batch_mins)
                idx_min = max(0, int(len(sorted_mins) * (1.0 - percentile)))
                a_min_c = sorted_mins[idx_min]

        a_scale, a_zp, _, _ = compute_quant_params(
            a_min_c, a_max_c, a_bins, symmetric=False,
        )

        # Weights
        w_min, w_max = compute_weight_range(
            model.conv0.weight.data, method=method, percentile=percentile,
        )
        w_scale, _, _, _ = compute_quant_params(
            w_min, w_max, w_bins, symmetric=True,
        )

        device = model.conv0.weight.device
        model.set_first_layer_params(
            input_scale=torch.tensor([a_scale], device=device),
            input_offset=torch.tensor([a_zp], device=device),
            w_scale=torch.tensor([w_scale], device=device),
            w_bins=w_bins,
            a_bins=a_bins,
        )

    def calibrate(self, w_bins: int, a_bins: int):
        """Full calibration pipeline.

        1. Collect activation stats via forward passes
        2. Compute percentile/minmax ranges
        3. Set quantization parameters on the model
        4. Enable quantization
        """
        cfg = self.config

        # Reset and collect stats
        self.model.reset_all_calibration_stats()
        self._collect_stats(cfg.num_calibration_batches)

        # Apply params to first layer (unless skipping)
        if not cfg.skip_first_last:
            self._apply_params_to_first_layer(self.model, w_bins, a_bins)

        # Apply params to all blocks and transitions
        from densenet_quant import QDenseLayer, QTransition

        for block in self.model.blocks:
            for layer in block.layers:
                self._apply_params_to_layer(layer, w_bins, a_bins)

        for trans in self.model.transitions:
            self._apply_params_to_layer(trans, w_bins, a_bins)

        # Enable quantization with bin counts
        self.model.enable_quantization_bins(
            w_bins, a_bins, skip_first_last=cfg.skip_first_last,
        )

    def calibrate_corrected(self, w_bins: int, a_bins: int):
        """Two-pass calibration with corrected inputs.

        Pass 1: Calibrate on float model outputs.
        Pass 2: Re-calibrate with quantization enabled (corrected activation ranges).
        """
        # Pass 1
        self.calibrate(w_bins, a_bins)

        # Pass 2: re-collect stats with quantization active
        self.model.reset_all_calibration_stats()
        self._collect_stats(self.config.num_calibration_batches)

        # Re-apply params with corrected stats
        from densenet_quant import QDenseLayer, QTransition

        if not self.config.skip_first_last:
            self._apply_params_to_first_layer(self.model, w_bins, a_bins)

        for block in self.model.blocks:
            for layer in block.layers:
                self._apply_params_to_layer(layer, w_bins, a_bins)

        for trans in self.model.transitions:
            self._apply_params_to_layer(trans, w_bins, a_bins)


# ---------------------------------------------------------------------------
# End-to-end post-quantization fine-tuning
# ---------------------------------------------------------------------------

def ptq_finetune(
    model: nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-4,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
) -> tuple[float, dict]:
    """End-to-end fine-tuning with quantization enabled.

    BN running stats are frozen (eval mode). Trains all conv/linear weights
    with a small LR.

    Returns:
        (best_accuracy, history_dict)
    """
    model.to(device)
    # Freeze BN stats: keep entire model in eval, but enable gradient computation
    model.eval()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

    best_accuracy = 0.0
    history = {"ft_train_loss": [], "ft_train_acc": [], "ft_test_acc": []}

    for epoch in range(epochs):
        # Train (BN stays in eval via model.eval(), but grads flow)
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in tqdm(train_loader, desc=f"FT Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100.0 * correct / total
        avg_loss = total_loss / len(train_loader)

        # Evaluate
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_acc = 100.0 * test_correct / test_total
        best_accuracy = max(best_accuracy, test_acc)

        history["ft_train_loss"].append(avg_loss)
        history["ft_train_acc"].append(train_acc)
        history["ft_test_acc"].append(test_acc)

        scheduler.step()

        print(
            f"  FT Epoch {epoch+1}: "
            f"Train Loss={avg_loss:.4f}, Train Acc={train_acc:.2f}%, "
            f"Test Acc={test_acc:.2f}%"
        )

    print(f"  Fine-tuning best accuracy: {best_accuracy:.2f}%")
    return best_accuracy, history
