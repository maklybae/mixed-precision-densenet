import json
import os
import time
from datetime import timedelta
from typing import Optional

import click
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data_utils import get_cifar10_loaders
from densenet_quant import MODEL_REGISTRY
from dq_quantizer import UniformQuantU3

MODELS_PATH = "./models"
RESULTS_PATH = "./results"

for path in (MODELS_PATH, RESULTS_PATH):
    if not os.path.exists(path):
        os.makedirs(path)


def clip_quant_grads(model: nn.Module):
    for module in model.modules():
        if isinstance(module, UniformQuantU3):
            d = module.d
            q_max = module.q_max

            if d.grad is not None:
                d.grad.data.clamp_(-d.data.item(), d.data.item())
            if q_max.grad is not None:
                q_max.grad.data.clamp_(-d.data.item(), d.data.item())


def clip_quant_vals(model: nn.Module):
    for module in model.modules():
        if isinstance(module, UniformQuantU3):
            with torch.no_grad():
                # Ensure d ≤ q_max
                d_val = module.d.data
                xmax_val = module.q_max.data

                min_val = torch.min(d_val, xmax_val - 1e-5)
                max_val = torch.max(d_val + 1e-5, xmax_val)

                module.d.data.copy_(min_val)
                module.q_max.data.copy_(max_val)

                # Re-clamp to allowed ranges
                module.d.data.clamp_(module.d_min, module.d_max)
                module.q_max.data.clamp_(module.xmax_min, module.xmax_max)


def calibrate_model(model, loader, num_batches: int = 25, device="cuda"):
    model = model.to(device)
    model.eval()
    model.enable_cache_mode()

    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loader, desc="Calibrating")):
            if i >= num_batches:
                break
            inputs = inputs.to(device)
            model(inputs)

    model.disable_cache_mode()
    model.calibrate()


def train_epoch(model, train_loader, optimizer, criterion, device,
                bitwidth_target=None, bitwidth_lambda=0.0, use_tqdm=True):
    model.train()
    total_ce_loss = 0.0
    total_penalty = 0.0
    correct = 0
    total = 0

    iterator = tqdm(train_loader, desc="Training") if use_tqdm else train_loader

    for inputs, targets in iterator:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        ce_loss = criterion(outputs, targets)

        loss = ce_loss
        if bitwidth_target is not None and bitwidth_lambda > 0:
            avg_bw = model.get_soft_bitwidths().mean()
            bw_penalty = bitwidth_lambda * (avg_bw - bitwidth_target) ** 2
            loss = ce_loss + bw_penalty
            total_penalty += bw_penalty.item()

        loss.backward()

        # Clip quantizer gradients before step (from paper)
        clip_quant_grads(model)

        optimizer.step()

        # Clip quantizer values after step (ensure d ≤ q_max)
        clip_quant_vals(model)

        total_ce_loss += ce_loss.item()
        _, predicted = outputs.detach().max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_ce_loss = total_ce_loss / len(train_loader)
    avg_penalty = total_penalty / len(train_loader)

    return avg_ce_loss, accuracy, avg_penalty


def evaluate(model, test_loader, criterion, device, use_tqdm=True):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    iterator = tqdm(test_loader, desc="Evaluating") if use_tqdm else test_loader

    with torch.no_grad():
        for inputs, targets in iterator:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(test_loader) if criterion is not None else 0.0

    return avg_loss, accuracy


def log_bitwidths(model, step: int):
    """Log per-layer bitwidths (discrete and soft) to MLflow."""
    bitwidths = model.get_bitwidths()

    all_bw = []
    all_soft_bw = []
    for name, info in bitwidths.items():
        bw = info["bitwidth"]
        soft_bw = info["soft_bitwidth"]
        all_bw.append(bw)
        all_soft_bw.append(soft_bw)
        short_name = name.replace(".", "_")
        mlflow.log_metric(f"bw/{short_name}", bw, step=step)
        mlflow.log_metric(f"bw_soft/{short_name}", soft_bw, step=step)

    if all_bw:
        avg_bw = sum(all_bw) / len(all_bw)
        min_bw = min(all_bw)
        max_bw = max(all_bw)
        mlflow.log_metric("bw/avg", avg_bw, step=step)
        mlflow.log_metric("bw/min", min_bw, step=step)
        mlflow.log_metric("bw/max", max_bw, step=step)

    if all_soft_bw:
        avg_soft = sum(all_soft_bw) / len(all_soft_bw)
        min_soft = min(all_soft_bw)
        max_soft = max(all_soft_bw)
        mlflow.log_metric("bw_soft/avg", avg_soft, step=step)
        mlflow.log_metric("bw_soft/min", min_soft, step=step)
        mlflow.log_metric("bw_soft/max", max_soft, step=step)

    return bitwidths


def save_bitwidths_snapshot(model, path: str):
    """Save full bitwidth info as JSON artifact."""
    bitwidths = model.get_bitwidths()
    with open(path, "w") as f:
        json.dump(bitwidths, f, indent=2)
    return bitwidths


def train_dq(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int = 160,
    lr: float = 0.01,
    quant_lr: Optional[float] = None,
    momentum: float = 0.9,
    weight_decay: float = 0.0002,
    lr_milestones: tuple = (80, 120),
    lr_gamma: float = 0.1,
    checkpoint_path: Optional[str] = None,
    bitwidth_target: Optional[float] = None,
    bitwidth_lambda: float = 0.0,
):
    """DQ training loop following the paper's recipe."""
    if quant_lr is None:
        quant_lr = lr

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Separate optimizer groups: network params and quantizer params
    network_params = model.get_network_params()
    quantizer_params = model.get_quantizer_params()

    print(f"Network parameters:   {sum(p.numel() for p in network_params):,}")
    print(f"Quantizer parameters: {len(quantizer_params):,} "
          f"({len(quantizer_params) // 2} quantizers × 2 [d, q_max])")

    optimizer = optim.SGD(
        [
            {"params": network_params, "lr": lr, "weight_decay": weight_decay},
            {"params": quantizer_params, "lr": quant_lr, "weight_decay": 0.0},
        ],
        momentum=momentum,
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(lr_milestones), gamma=lr_gamma
    )

    best_accuracy = 0.0
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "lr": [],
    }

    start_time = time.time()

    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        current_qlr = optimizer.param_groups[1]["lr"]
        print(f"\nEpoch {epoch + 1}/{epochs} "
              f"(LR: {current_lr:.6f}, Q-LR: {current_qlr:.6f})")

        train_loss, train_acc, train_penalty = train_epoch(
            model, train_loader, optimizer, criterion, device,
            bitwidth_target=bitwidth_target, bitwidth_lambda=bitwidth_lambda,
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(current_lr)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_acc", test_acc, step=epoch)
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        mlflow.log_metric("quant_learning_rate", current_qlr, step=epoch)

        with torch.no_grad():
            avg_soft_bw = model.get_soft_bitwidths().mean().item()
        mlflow.log_metric("bw/avg_soft", avg_soft_bw, step=epoch)

        if bitwidth_target is not None and bitwidth_lambda > 0:
            mlflow.log_metric("bw_penalty", train_penalty, step=epoch)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%")

        # Log bitwidths periodically (every 10 epochs + first + last)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            bw_info = log_bitwidths(model, step=epoch)
            bw_vals = [v["bitwidth"] for v in bw_info.values()]
            if bw_vals:
                print(f"Bitwidths: avg={sum(bw_vals)/len(bw_vals):.1f}, "
                      f"min={min(bw_vals)}, max={max(bw_vals)}")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  → Saved best model with accuracy: {best_accuracy:.2f}%")

        scheduler.step()

    elapsed_time = time.time() - start_time
    print(f"\nDQ training completed in {timedelta(seconds=int(elapsed_time))}")
    print(f"Best accuracy: {best_accuracy:.2f}%")

    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_metric("training_time_seconds", elapsed_time)

    return model, history, best_accuracy


@click.command()
@click.option(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to pre-trained float checkpoint",
)
@click.option(
    "--model",
    type=click.Choice(list(MODEL_REGISTRY.keys())),
    default="densenet_bc_100_12",
    help="Model architecture to use",
)
@click.option("--epochs", type=int, default=160, help="Number of DQ training epochs")
@click.option("--batch-size", type=int, default=128, help="Batch size")
@click.option("--lr", type=float, default=0.01, help="Learning rate for network params")
@click.option(
    "--quant-lr",
    type=float,
    default=None,
    help="Learning rate for quantizer params (default: same as --lr)",
)
@click.option("--momentum", type=float, default=0.9, help="SGD momentum")
@click.option("--weight-decay", type=float, default=0.0002, help="Weight decay")
@click.option("--init-bitwidth", type=int, default=4, help="Initial bitwidth for DQ")
@click.option(
    "--bitwidth-target",
    type=float,
    default=None,
    help="Target average bitwidth for penalty (None = no penalty)",
)
@click.option(
    "--bitwidth-lambda",
    type=float,
    default=0.01,
    help="Penalty weight for bitwidth target",
)
@click.option(
    "--calibration-batches",
    type=int,
    default=25,
    help="Number of batches for activation calibration",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu", "mps"]),
    default="cuda",
    help="Device",
)
@click.option(
    "--name",
    type=str,
    default="densenet_bc_100_12_dq",
    help="Experiment name for saving",
)
@click.option(
    "--mlflow-uri",
    type=str,
    default="http://89.169.147.243",
    help="MLflow tracking URI",
)
@click.option(
    "--experiment-name",
    type=str,
    default="densenet-cifar10-dq",
    help="MLflow experiment name",
)
def main(
    checkpoint,
    model,
    epochs,
    batch_size,
    lr,
    quant_lr,
    momentum,
    weight_decay,
    init_bitwidth,
    bitwidth_target,
    bitwidth_lambda,
    calibration_batches,
    device,
    name,
    mlflow_uri,
    experiment_name,
):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {mlflow_uri}")
    print(f"MLflow experiment: {experiment_name}")

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    device_str = device
    device = torch.device(device)
    print(f"Using device: {device}")

    lr_milestones = (80, 120)

    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    model_fn = MODEL_REGISTRY[model]
    net = model_fn(num_classes=10, memory_efficient=True)
    num_params = net.count_parameters()
    print(f"Model: {model}, parameters: {num_params:,}")

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Loading pre-trained weights from: {checkpoint}")
    net.load_state_dict(torch.load(checkpoint, map_location=device))
    net = net.to(device)

    print("\nEvaluating float baseline...")
    _, float_acc = evaluate(net, test_loader, None, device)
    print(f"Float32 Accuracy: {float_acc:.2f}%")

    print(f"\nCalibrating activation ranges ({calibration_batches} batches)...")
    calibrate_model(net, train_loader, num_batches=calibration_batches, device=device)
    print("Calibration complete.")

    print(f"\nEnabling DQ (U3 parametrization, init_bitwidth={init_bitwidth})...")
    net.enable_dq(init_bitwidth=init_bitwidth)
    net = net.to(device)

    init_bw_path = f"{RESULTS_PATH}/{name}_initial_bitwidths.json"
    save_bitwidths_snapshot(net, init_bw_path)

    total_params_after = net.count_parameters()
    quant_params = net.get_quantizer_params()
    print(f"Total parameters after DQ: {total_params_after:,} "
          f"(+{len(quant_params)} quantizer params)")

    dq_checkpoint_path = f"{MODELS_PATH}/{name}.pt"

    with mlflow.start_run(run_name=name):
        mlflow.log_param("mode", "dq_training")
        mlflow.log_param("model_name", name)
        mlflow.log_param("model_architecture", model)
        mlflow.log_param("num_parameters", num_params)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("device", device_str)
        mlflow.log_param("dataset", "CIFAR-10")
        mlflow.log_param("num_classes", 10)

        mlflow.log_param("float_checkpoint", checkpoint)
        mlflow.log_param("float_accuracy", float_acc)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("quant_learning_rate", quant_lr if quant_lr else lr)
        mlflow.log_param("momentum", momentum)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("lr_schedule", "MultiStepLR")
        mlflow.log_param("lr_milestones", str(list(lr_milestones)))
        mlflow.log_param("lr_gamma", 0.1)
        mlflow.log_param("optimizer", "SGD")
        mlflow.log_param("quantization_method", "DQ_U3")
        mlflow.log_param("quantizer_parametrization", "d_xmax (Case U3)")
        mlflow.log_param("init_bitwidth", init_bitwidth)
        mlflow.log_param("calibration_batches", calibration_batches)
        mlflow.log_param("num_quantizers", len(quant_params) // 2)
        mlflow.log_param("grad_clipping", "d.grad∈[-d,d], qmax.grad∈[-d,d]")
        mlflow.log_param("val_clipping", "d ≤ q_max")
        mlflow.log_param("bitwidth_target", bitwidth_target)
        mlflow.log_param("bitwidth_lambda", bitwidth_lambda)

        mlflow.log_artifact(init_bw_path)

        net, history, best_acc = train_dq(
            model=net,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            lr=lr,
            quant_lr=quant_lr,
            momentum=momentum,
            weight_decay=weight_decay,
            lr_milestones=lr_milestones,
            checkpoint_path=dq_checkpoint_path,
            bitwidth_target=bitwidth_target,
            bitwidth_lambda=bitwidth_lambda,
        )

        history_path = f"{RESULTS_PATH}/{name}_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        mlflow.log_artifact(history_path)

        final_bw_path = f"{RESULTS_PATH}/{name}_final_bitwidths.json"
        final_bw = save_bitwidths_snapshot(net, final_bw_path)
        mlflow.log_artifact(final_bw_path)

        print("\n" + "=" * 60)
        print("Final Bitwidths:")
        print("=" * 60)
        bw_vals = [v["bitwidth"] for v in final_bw.values()]
        if bw_vals:
            print(f"  Average: {sum(bw_vals) / len(bw_vals):.2f}")
            print(f"  Min:     {min(bw_vals)}")
            print(f"  Max:     {max(bw_vals)}")
            print(f"  Count:   {len(bw_vals)} quantizers")

        # Log final checkpoint
        if os.path.exists(dq_checkpoint_path):
            mlflow.log_artifact(dq_checkpoint_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
