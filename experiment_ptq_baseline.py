"""
PTQ Baseline experiments for DenseNet-BC models on CIFAR-10.

Two experiment types:
  1. Uniform: same bitwidth for weights and activations (incl. non-integer)
  2. Split:   different bitwidths for weights vs activations

Usage examples:
    # Full baseline for all models (uniform + split):
    python experiment_ptq_baseline.py --model all

    # Uniform only for one model, custom bitwidths:
    python experiment_ptq_baseline.py --model densenet_bc_52_12 \\
        --experiment-type uniform --uniform-bits "3,3.5,4,4.5,5,6,8"

    # Split only, no fine-tuning:
    python experiment_ptq_baseline.py --experiment-type split --no-finetune

    # Quick test with minmax calibration:
    python experiment_ptq_baseline.py --model densenet_bc_52_12 \\
        --calibration-method minmax --uniform-bits "8,4" --no-finetune
"""

import json
import os
from copy import deepcopy

import click
import mlflow
import torch
import torch.nn as nn
from tqdm import tqdm

from data_utils import get_cifar10_loaders
from densenet_quant import MODEL_REGISTRY
from ptq_utils import (
    PTQCalibrator,
    PTQConfig,
    bits_to_bins,
    bins_to_bits,
    ptq_finetune,
)

MODELS_PATH = "./models"
RESULTS_PATH = "./results"

for path in (MODELS_PATH, RESULTS_PATH):
    os.makedirs(path, exist_ok=True)

# ---------------------------------------------------------------------------
# Default experiment configurations
# ---------------------------------------------------------------------------

DEFAULT_UNIFORM_BITS = [3, 3.5, 4, 4.5, 5, 5.5, 6, 8]

DEFAULT_SPLIT_CONFIGS = [
    # (w_bits, a_bits)
    (3, 4),
    (3, 5),
    (3, 6),
    (3.5, 4.5),
    (3.5, 5.5),
    (3.5, 6.5),
    (4, 5),
    (4, 6),
    (4, 7),
    (4.5, 5.5),
    (4.5, 6.5),
    (4.5, 7.5),
    (5, 6),
    (5, 7),
    (5, 8),
]

# Model name -> checkpoint path mapping
CHECKPOINT_MAP = {
    "densenet_bc_52_12": f"{MODELS_PATH}/cifar10_densenet_bc_52_12.pt",
    "densenet_bc_64_12": f"{MODELS_PATH}/cifar10_densenet_bc_64_12.pt",
    "densenet_bc_76_12": f"{MODELS_PATH}/cifar10_densenet_bc_76_12.pt",
    "densenet_bc_88_12": f"{MODELS_PATH}/cifar10_densenet_bc_88_12.pt",
    "densenet_bc_100_12": f"{MODELS_PATH}/densenet_bc_100_12.pt",
}


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Single-model experiment runner
# ---------------------------------------------------------------------------

def run_ptq_experiment(
    model_name: str,
    checkpoint_path: str,
    train_loader,
    test_loader,
    device: torch.device,
    experiment_type: str,
    uniform_bits: list[float],
    split_configs: list[tuple[float, float]],
    calibration_method: str,
    percentile: float,
    num_calibration_batches: int,
    corrected_inputs: bool,
    skip_first_last: bool,
    do_finetune: bool,
    finetune_epochs: int,
    finetune_lr: float,
) -> dict:
    """Run PTQ baseline experiment for a single model."""

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    # Load model
    model_fn = MODEL_REGISTRY[model_name]
    net = model_fn(num_classes=10)
    net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    net = net.to(device)

    # Float32 baseline
    print("Evaluating float32 baseline...")
    float_acc = evaluate(net, test_loader, device)
    print(f"Float32 accuracy: {float_acc:.2f}%")

    results = {
        "model": model_name,
        "float32_accuracy": float_acc,
        "calibration": {
            "method": calibration_method,
            "percentile": percentile,
            "batches": num_calibration_batches,
            "corrected_inputs": corrected_inputs,
            "skip_first_last": skip_first_last,
        },
        "uniform": [],
        "split": [],
    }

    mlflow.log_metric("float32_accuracy", float_acc)

    # --- Uniform experiments ---
    if experiment_type in ("uniform", "both"):
        print(f"\n--- Uniform W=A experiments ---")
        for bits in uniform_bits:
            w_bins = bits_to_bins(bits, signed=True)
            a_bins = bits_to_bins(bits, signed=False)
            label = f"{bits}bit (w_bins={w_bins}, a_bins={a_bins})"
            print(f"\n  [{label}]")

            # Reload fresh model for each bitwidth
            net_q = model_fn(num_classes=10)
            net_q.load_state_dict(torch.load(checkpoint_path, map_location=device))
            net_q = net_q.to(device)

            config = PTQConfig(
                w_bins=w_bins,
                a_bins=a_bins,
                calibration_method=calibration_method,
                percentile=percentile,
                num_calibration_batches=num_calibration_batches,
                corrected_inputs=corrected_inputs,
                skip_first_last=skip_first_last,
            )
            calibrator = PTQCalibrator(net_q, train_loader, device, config)

            if corrected_inputs:
                calibrator.calibrate_corrected(w_bins, a_bins)
            else:
                calibrator.calibrate(w_bins, a_bins)

            ptq_acc = evaluate(net_q, test_loader, device)
            drop = float_acc - ptq_acc
            print(f"  PTQ accuracy: {ptq_acc:.2f}% (drop: {drop:.2f}%)")

            mlflow.log_metric(f"uniform_{bits}bit_ptq", ptq_acc)

            entry = {
                "bits": bits,
                "w_bins": w_bins,
                "a_bins": a_bins,
                "accuracy": round(ptq_acc, 4),
                "drop": round(drop, 4),
            }

            if do_finetune:
                ft_acc, ft_hist = ptq_finetune(
                    net_q, train_loader, test_loader, device,
                    epochs=finetune_epochs, lr=finetune_lr,
                )
                ft_drop = float_acc - ft_acc
                print(f"  FT accuracy:  {ft_acc:.2f}% (drop: {ft_drop:.2f}%)")
                entry["accuracy_ft"] = round(ft_acc, 4)
                entry["drop_ft"] = round(ft_drop, 4)
                mlflow.log_metric(f"uniform_{bits}bit_ft", ft_acc)

            results["uniform"].append(entry)

    # --- Split experiments ---
    if experiment_type in ("split", "both"):
        print(f"\n--- Split W!=A experiments ---")
        for w_bits, a_bits in split_configs:
            w_bins = bits_to_bins(w_bits, signed=True)
            a_bins = bits_to_bins(a_bits, signed=False)
            label = f"W{w_bits}bit/A{a_bits}bit (w_bins={w_bins}, a_bins={a_bins})"
            print(f"\n  [{label}]")

            net_q = model_fn(num_classes=10)
            net_q.load_state_dict(torch.load(checkpoint_path, map_location=device))
            net_q = net_q.to(device)

            config = PTQConfig(
                w_bins=w_bins,
                a_bins=a_bins,
                calibration_method=calibration_method,
                percentile=percentile,
                num_calibration_batches=num_calibration_batches,
                corrected_inputs=corrected_inputs,
                skip_first_last=skip_first_last,
            )
            calibrator = PTQCalibrator(net_q, train_loader, device, config)

            if corrected_inputs:
                calibrator.calibrate_corrected(w_bins, a_bins)
            else:
                calibrator.calibrate(w_bins, a_bins)

            ptq_acc = evaluate(net_q, test_loader, device)
            drop = float_acc - ptq_acc
            print(f"  PTQ accuracy: {ptq_acc:.2f}% (drop: {drop:.2f}%)")

            mlflow.log_metric(f"split_w{w_bits}_a{a_bits}_ptq", ptq_acc)

            entry = {
                "w_bits": w_bits,
                "a_bits": a_bits,
                "w_bins": w_bins,
                "a_bins": a_bins,
                "accuracy": round(ptq_acc, 4),
                "drop": round(drop, 4),
            }

            if do_finetune:
                ft_acc, ft_hist = ptq_finetune(
                    net_q, train_loader, test_loader, device,
                    epochs=finetune_epochs, lr=finetune_lr,
                )
                ft_drop = float_acc - ft_acc
                print(f"  FT accuracy:  {ft_acc:.2f}% (drop: {ft_drop:.2f}%)")
                entry["accuracy_ft"] = round(ft_acc, 4)
                entry["drop_ft"] = round(ft_drop, 4)
                mlflow.log_metric(f"split_w{w_bits}_a{a_bits}_ft", ft_acc)

            results["split"].append(entry)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_bits_list(ctx, param, value):
    """Parse comma-separated bitwidth list."""
    if value is None:
        return None
    try:
        return [float(x.strip()) for x in value.split(",")]
    except ValueError:
        raise click.BadParameter("Must be comma-separated numbers, e.g. '3,3.5,4,8'")


def parse_split_configs(ctx, param, value):
    """Parse split configs from JSON file or use defaults."""
    if value is None:
        return None
    if os.path.isfile(value):
        with open(value) as f:
            return [tuple(pair) for pair in json.load(f)]
    raise click.BadParameter(f"File not found: {value}")


@click.command()
@click.option(
    "--model",
    type=click.Choice(list(MODEL_REGISTRY.keys()) + ["all"]),
    default="all",
    help="Model architecture (or 'all' for all 5 baseline models)",
)
@click.option(
    "--checkpoint", type=str, default=None,
    help="Path to checkpoint (auto-detected if not specified)",
)
@click.option(
    "--device", type=click.Choice(["cuda", "cpu", "mps"]), default="cuda",
)
@click.option("--batch-size", type=int, default=128)
@click.option("--calibration-batches", type=int, default=50)
@click.option(
    "--calibration-method",
    type=click.Choice(["minmax", "percentile"]),
    default="percentile",
)
@click.option("--percentile", type=float, default=0.9999)
@click.option("--corrected-inputs/--no-corrected-inputs", default=True)
@click.option("--skip-first-last/--no-skip-first-last", default=True)
@click.option(
    "--experiment-type",
    type=click.Choice(["uniform", "split", "both"]),
    default="both",
)
@click.option(
    "--uniform-bits", type=str, default=None, callback=parse_bits_list,
    help="Comma-separated bitwidths for uniform experiment (e.g. '3,3.5,4,4.5,5,6,8')",
)
@click.option(
    "--split-configs", type=str, default=None, callback=parse_split_configs,
    help="JSON file with list of [w_bits, a_bits] pairs for split experiment",
)
@click.option("--finetune/--no-finetune", default=True)
@click.option("--finetune-epochs", type=int, default=5)
@click.option("--finetune-lr", type=float, default=1e-4)
@click.option("--mlflow-uri", type=str, default="http://89.169.147.243")
@click.option(
    "--experiment-name", type=str, default="densenet-cifar10-ptq-baseline",
)
@click.option("--name", type=str, default=None, help="Run name override")
def main(
    model,
    checkpoint,
    device,
    batch_size,
    calibration_batches,
    calibration_method,
    percentile,
    corrected_inputs,
    skip_first_last,
    experiment_type,
    uniform_bits,
    split_configs,
    finetune,
    finetune_epochs,
    finetune_lr,
    mlflow_uri,
    experiment_name,
    name,
):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {mlflow_uri}")
    print(f"MLflow experiment: {experiment_name}")

    device = torch.device(device)
    print(f"Using device: {device}")

    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    # Resolve bitwidth configs
    u_bits = uniform_bits if uniform_bits is not None else DEFAULT_UNIFORM_BITS
    s_configs = split_configs if split_configs is not None else DEFAULT_SPLIT_CONFIGS

    # Resolve model list
    if model == "all":
        model_names = list(CHECKPOINT_MAP.keys())
    else:
        model_names = [model]

    all_results = {}

    for model_name in model_names:
        # Resolve checkpoint
        ckpt = checkpoint if checkpoint else CHECKPOINT_MAP.get(model_name)
        if ckpt is None:
            ckpt = f"{MODELS_PATH}/{model_name}.pt"
        if not os.path.exists(ckpt):
            print(f"WARNING: Checkpoint not found: {ckpt}, skipping {model_name}")
            continue

        run_name = name if name else f"ptq_baseline_{model_name}"

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", model_name)
            mlflow.log_param("calibration_method", calibration_method)
            mlflow.log_param("percentile", percentile)
            mlflow.log_param("calibration_batches", calibration_batches)
            mlflow.log_param("corrected_inputs", corrected_inputs)
            mlflow.log_param("skip_first_last", skip_first_last)
            mlflow.log_param("experiment_type", experiment_type)
            mlflow.log_param("finetune", finetune)
            if finetune:
                mlflow.log_param("finetune_epochs", finetune_epochs)
                mlflow.log_param("finetune_lr", finetune_lr)
            mlflow.log_param("uniform_bits", str(u_bits))

            results = run_ptq_experiment(
                model_name=model_name,
                checkpoint_path=ckpt,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                experiment_type=experiment_type,
                uniform_bits=u_bits,
                split_configs=s_configs,
                calibration_method=calibration_method,
                percentile=percentile,
                num_calibration_batches=calibration_batches,
                corrected_inputs=corrected_inputs,
                skip_first_last=skip_first_last,
                do_finetune=finetune,
                finetune_epochs=finetune_epochs,
                finetune_lr=finetune_lr,
            )

            # Save results
            results_dir = f"{RESULTS_PATH}/cifar10_{model_name}"
            os.makedirs(results_dir, exist_ok=True)
            results_path = f"{results_dir}/ptq_baseline.json"
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {results_path}")
            mlflow.log_artifact(results_path)

            all_results[model_name] = results

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for mname, res in all_results.items():
        print(f"\n{mname} (float32: {res['float32_accuracy']:.2f}%):")
        if res["uniform"]:
            print("  Uniform:")
            for entry in res["uniform"]:
                line = f"    {entry['bits']:>5.1f} bit: {entry['accuracy']:.2f}%"
                if "accuracy_ft" in entry:
                    line += f" -> FT: {entry['accuracy_ft']:.2f}%"
                print(line)
        if res["split"]:
            print("  Split:")
            for entry in res["split"]:
                line = (
                    f"    W{entry['w_bits']:.1f}/A{entry['a_bits']:.1f}: "
                    f"{entry['accuracy']:.2f}%"
                )
                if "accuracy_ft" in entry:
                    line += f" -> FT: {entry['accuracy_ft']:.2f}%"
                print(line)

    print("\nDone!")


if __name__ == "__main__":
    main()
