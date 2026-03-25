"""
HAWQ Hessian Sensitivity Analysis for DenseNet.

Computes per-block top Hessian eigenvalue via power iteration
to rank layers by quantization sensitivity.

Based on: HAWQ: Hessian AWare Quantization (arXiv:1905.03696)
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, List, Tuple

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from data_utils import get_cifar10_loaders
from densenet_quant import (
    MyQDenseNet,
    QDenseBlock,
    QTransition,
    densenet_bc_100_12,
    densenet_bc_190_40,
)

RESULTS_PATH = "./results"
os.makedirs(RESULTS_PATH, exist_ok=True)

MODEL_CONSTRUCTORS = {
    "densenet_bc_100_12": densenet_bc_100_12,
    "densenet_bc_190_40": densenet_bc_190_40,
}


@dataclass
class BlockInfo:
    block_id: int
    name: str
    params: List[nn.Parameter]
    num_params: int


def enumerate_blocks(model: MyQDenseNet) -> List[BlockInfo]:
    """Enumerate all quantizable blocks in the model."""
    blocks = []
    block_id = 0

    # Block 0: first conv
    blocks.append(BlockInfo(
        block_id=block_id,
        name="conv0",
        params=[model.conv0.weight],
        num_params=model.conv0.weight.numel(),
    ))
    block_id += 1

    # Dense blocks + transitions
    for bi, dense_block in enumerate(model.blocks):
        for li, layer in enumerate(dense_block.layers):
            params = [layer.conv1.weight, layer.conv2.weight]
            blocks.append(BlockInfo(
                block_id=block_id,
                name=f"block{bi}.layer{li}",
                params=params,
                num_params=sum(p.numel() for p in params),
            ))
            block_id += 1

        if bi < len(model.transitions):
            trans = model.transitions[bi]
            blocks.append(BlockInfo(
                block_id=block_id,
                name=f"transition{bi}",
                params=[trans.conv.weight],
                num_params=trans.conv.weight.numel(),
            ))
            block_id += 1

    # Classifier
    classifier_params = [model.classifier.weight]
    if model.classifier.bias is not None:
        classifier_params.append(model.classifier.bias)
    blocks.append(BlockInfo(
        block_id=block_id,
        name="classifier",
        params=classifier_params,
        num_params=sum(p.numel() for p in classifier_params),
    ))

    return blocks


def disable_inplace_relu(model: nn.Module):
    """Disable inplace=True on all ReLU modules for safe second-order gradients."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def _compute_norm(tensors: List[torch.Tensor]) -> float:
    """Compute L2 norm of a list of tensors treated as one flat vector."""
    return torch.sqrt(sum(torch.sum(t ** 2) for t in tensors)).item()


def hessian_vector_product(
    loss: torch.Tensor,
    params: List[nn.Parameter],
    v: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    Compute H @ v where H is the Hessian of `loss` w.r.t. `params`.

    Uses double backpropagation:
      1. g = d(loss)/d(params)  with create_graph=True
      2. gv = g^T v  (scalar)
      3. Hv = d(gv)/d(params)
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

    gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v))

    Hv = torch.autograd.grad(gv, params, retain_graph=True)

    return [hv.detach() for hv in Hv]


def power_iteration(
    model: nn.Module,
    block_info: BlockInfo,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_iters: int = 100,
    num_batches: int = 2,
    tol: float = 1e-3,
    seed: int = 42,
) -> Tuple[float, int]:
    """
    Compute the top Hessian eigenvalue for a single block via power iteration.

    Returns:
        (eigenvalue, num_converged_iters)
    """
    model.eval()

    # Initialize random vector v, normalized
    # Generate on CPU for reproducibility, then move to device
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed + block_info.block_id)

    v = [torch.randn(p.shape, generator=rng).to(device) for p in block_info.params]
    v_norm = _compute_norm(v)
    v = [vi / v_norm for vi in v]

    eigenvalue = 0.0

    pbar = tqdm(range(num_iters), desc=f"  {block_info.name}", leave=False)
    for iteration in pbar:
        # Accumulate Hv over multiple batches
        Hv_accum = [torch.zeros_like(p) for p in block_info.params]
        total_samples = 0

        batch_iter = iter(data_loader)
        for _ in range(num_batches):
            try:
                inputs, targets = next(batch_iter)
            except StopIteration:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_samples += batch_size

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            Hv_batch = hessian_vector_product(loss, block_info.params, v)

            for j in range(len(Hv_accum)):
                Hv_accum[j] += Hv_batch[j] * batch_size

            del loss, outputs, Hv_batch

        # Average over samples
        Hv = [h / total_samples for h in Hv_accum]

        # Eigenvalue estimate: v^T @ Hv
        new_eigenvalue = sum(
            torch.sum(vi * hvi).item()
            for vi, hvi in zip(v, Hv)
        )

        pbar.set_postfix({"lambda": f"{new_eigenvalue:.4e}"})

        # Update v = Hv / ||Hv||
        Hv_norm = _compute_norm(Hv)
        if Hv_norm < 1e-12:
            pbar.close()
            return 0.0, iteration

        v = [hvi / Hv_norm for hvi in Hv]

        # Convergence check
        if iteration > 0:
            relative_change = abs(new_eigenvalue - eigenvalue) / max(abs(eigenvalue), 1e-10)
            if relative_change < tol:
                pbar.close()
                return new_eigenvalue, iteration + 1

        eigenvalue = new_eigenvalue

        del Hv_accum, Hv

    pbar.close()
    return eigenvalue, num_iters


def run_hawq_analysis(
    model: MyQDenseNet,
    train_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_iters: int = 100,
    num_batches: int = 2,
    tol: float = 1e-3,
    seed: int = 42,
) -> List[Dict]:
    """Run HAWQ Hessian sensitivity analysis on all blocks."""
    criterion = nn.CrossEntropyLoss()
    blocks = enumerate_blocks(model)

    print(f"Found {len(blocks)} blocks to analyze")
    print(f"Power iteration: max {num_iters} iters, {num_batches} batches, tol={tol}")

    results = []

    for block_info in tqdm(blocks, desc="HAWQ Analysis"):
        # Disable requires_grad for all params, enable only for current block
        orig_grad_state = {}
        for name, param in model.named_parameters():
            orig_grad_state[name] = param.requires_grad
            param.requires_grad_(False)

        for param in block_info.params:
            param.requires_grad_(True)

        eigenvalue, converged_iters = power_iteration(
            model=model,
            block_info=block_info,
            data_loader=train_loader,
            criterion=criterion,
            device=device,
            num_iters=num_iters,
            num_batches=num_batches,
            tol=tol,
            seed=seed,
        )

        # Restore requires_grad state
        for name, param in model.named_parameters():
            param.requires_grad_(orig_grad_state[name])

        sensitivity = abs(eigenvalue) / block_info.num_params

        results.append({
            "block_id": block_info.block_id,
            "name": block_info.name,
            "num_params": block_info.num_params,
            "eigenvalue": eigenvalue,
            "abs_eigenvalue": abs(eigenvalue),
            "sensitivity": sensitivity,
            "converged_iters": converged_iters,
        })

        tqdm.write(
            f"  {block_info.name}: lambda={eigenvalue:.6e}, "
            f"n={block_info.num_params}, S={sensitivity:.6e}, "
            f"iters={converged_iters}"
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Sort by sensitivity descending, add rank
    results.sort(key=lambda r: r["sensitivity"], reverse=True)
    for rank, r in enumerate(results):
        r["rank"] = rank + 1

    return results


def plot_sensitivity_bar(results: List[Dict], save_path: str, title: str):
    """Bar chart of per-block sensitivity in original block order."""
    by_id = sorted(results, key=lambda r: r["block_id"])

    names = [r["name"] for r in by_id]
    sensitivities = [r["sensitivity"] for r in by_id]

    fig, ax = plt.subplots(figsize=(20, 6))

    colors = []
    for r in by_id:
        if r["name"] == "conv0":
            colors.append("#2196F3")
        elif r["name"] == "classifier":
            colors.append("#F44336")
        elif r["name"].startswith("transition"):
            colors.append("#FF9800")
        else:
            colors.append("#4CAF50")

    ax.bar(range(len(names)), sensitivities, color=colors, width=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_ylabel(r"Sensitivity  $S_i = |\lambda_i| / n_i$")
    ax.set_xlabel("Block")
    ax.set_title(title)
    ax.set_yscale("log")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="conv0"),
        Patch(facecolor="#4CAF50", label="DenseLayer"),
        Patch(facecolor="#FF9800", label="Transition"),
        Patch(facecolor="#F44336", label="classifier"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar chart: {save_path}")


def plot_sensitivity_heatmap(results: List[Dict], model: MyQDenseNet, save_path: str):
    """Heatmap of sensitivities for dense layers (n_blocks x n_layers_per_block)."""
    n_blocks = len(model.blocks)
    n_layers = len(model.blocks[0].layers)

    heatmap = np.full((n_blocks, n_layers), np.nan)
    result_by_name = {r["name"]: r for r in results}

    for bi in range(n_blocks):
        for li in range(n_layers):
            key = f"block{bi}.layer{li}"
            if key in result_by_name:
                val = result_by_name[key]["sensitivity"]
                heatmap[bi, li] = np.log10(val + 1e-20)

    fig, ax = plt.subplots(figsize=(18, 4))
    im = ax.imshow(heatmap, aspect="auto", cmap="YlOrRd", interpolation="nearest")

    ax.set_yticks(range(n_blocks))
    ax.set_yticklabels([f"Dense Block {i}" for i in range(n_blocks)])
    ax.set_xticks(range(n_layers))
    ax.set_xticklabels(range(n_layers), fontsize=8)
    ax.set_xlabel("Layer index within dense block")
    ax.set_title(r"HAWQ Hessian Sensitivity ($\log_{10} S_i$) -- DenseLayer blocks")

    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label(r"$\log_{10}(S_i)$")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap: {save_path}")


@click.command()
@click.option("--checkpoint", type=str, required=True, help="Path to pretrained float model checkpoint")
@click.option("--model", "model_fn", type=click.Choice(list(MODEL_CONSTRUCTORS.keys())),
              default="densenet_bc_100_12", help="Model architecture")
@click.option("--num-iters", type=int, default=100, help="Max power iteration steps per block")
@click.option("--num-batches", type=int, default=2, help="Number of training batches for Hessian estimation")
@click.option("--batch-size", type=int, default=128, help="Batch size for data loading")
@click.option("--tol", type=float, default=1e-3, help="Convergence tolerance for power iteration")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--device", type=str, default="auto", help="Device: cuda, mps, cpu, or auto")
@click.option("--name", type=str, default=None, help="Experiment name for output files (default: model name)")
def main(checkpoint, model_fn, num_iters, num_batches, batch_size, tol, seed, device, name):
    """HAWQ Hessian sensitivity analysis for DenseNet."""

    # Resolve device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    device = torch.device(device)
    print(f"Device: {device}")

    if name is None:
        name = model_fn

    torch.manual_seed(seed)

    # Load data
    train_loader, _ = get_cifar10_loaders(batch_size=batch_size)
    print(f"Loaded CIFAR-10 train: {len(train_loader)} batches of {batch_size}")

    # Load model
    print(f"Loading model {model_fn} from {checkpoint}")
    model = MODEL_CONSTRUCTORS[model_fn](num_classes=10)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # Disable in-place ReLU for second-order gradients
    disable_inplace_relu(model)

    # Run analysis
    start_time = time.time()
    results = run_hawq_analysis(
        model, train_loader, device,
        num_iters=num_iters, num_batches=num_batches, tol=tol, seed=seed,
    )
    elapsed = time.time() - start_time

    # Save JSON
    output = {
        "metadata": {
            "model": model_fn,
            "checkpoint": checkpoint,
            "num_blocks": len(results),
            "total_params": sum(r["num_params"] for r in results),
            "num_iters": num_iters,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "tolerance": tol,
            "seed": seed,
            "device": str(device),
            "elapsed_seconds": round(elapsed, 1),
        },
        "blocks": results,
    }

    json_path = f"{RESULTS_PATH}/hawq_sensitivity_{name}.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results: {json_path}")

    # Visualize
    bar_path = f"{RESULTS_PATH}/hawq_sensitivity_{name}_bar.png"
    heatmap_path = f"{RESULTS_PATH}/hawq_sensitivity_{name}_heatmap.png"

    plot_sensitivity_bar(results, bar_path, title=f"HAWQ Sensitivity: {name}")
    plot_sensitivity_heatmap(results, model, heatmap_path)

    # Print summary
    print(f"\n{'=' * 80}")
    print("HAWQ Sensitivity Ranking (top 10 most sensitive blocks):")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6} {'Block':<25} {'Params':<10} {'Eigenvalue':<15} {'S_i':<15}")
    print(f"{'-' * 80}")
    for r in results[:10]:
        print(
            f"{r['rank']:<6} {r['name']:<25} {r['num_params']:<10} "
            f"{r['eigenvalue']:<15.6e} {r['sensitivity']:<15.6e}"
        )

    print(f"\nCompleted in {timedelta(seconds=int(elapsed))}")


if __name__ == "__main__":
    main()
