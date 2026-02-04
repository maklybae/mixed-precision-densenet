"""
Usage:
    python experiment_densenet_cifar.py --mode train --epochs 300

    python experiment_densenet_cifar.py --mode quantize --checkpoint models/densenet_cifar10.pt

    python experiment_densenet_cifar.py --mode full --epochs 300
"""

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
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from densenet_quant import densenet_bc_190_40

MODELS_PATH = "./models"
RESULTS_PATH = "./results"

for path in (MODELS_PATH, RESULTS_PATH):
    if not os.path.exists(path):
        os.makedirs(path)


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 16):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion, device, use_tqdm=True):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    iterator = tqdm(train_loader, desc="Training") if use_tqdm else train_loader

    for inputs, targets in iterator:
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

    accuracy = 100.0 * correct / total
    avg_loss = total_loss / len(train_loader)

    return avg_loss, accuracy


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


def train_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs: int = 300,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    checkpoint_path: Optional[str] = None,
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True,
    )

    # Learning rate from https://arxiv.org/abs/1608.06993
    milestones = [int(epochs * 0.5), int(epochs * 0.75)]
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1
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
        print(f"\nEpoch {epoch + 1}/{epochs} (LR: {scheduler.get_last_lr()[0]:.6f})")

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["lr"].append(scheduler.get_last_lr()[0])

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_acc", train_acc, step=epoch)
        mlflow.log_metric("test_loss", test_loss, step=epoch)
        mlflow.log_metric("test_acc", test_acc, step=epoch)
        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=epoch)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            if checkpoint_path:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Saved best model with accuracy: {best_accuracy:.2f}%")

        scheduler.step()

    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {timedelta(seconds=int(elapsed_time))}")
    print(f"Best accuracy: {best_accuracy:.2f}%")

    mlflow.log_metric("best_accuracy", best_accuracy)
    mlflow.log_metric("training_time_seconds", elapsed_time)

    return model, history, best_accuracy


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


def quantize_and_evaluate(
    model, train_loader, test_loader, device, bits_list: list[int] = [16, 8, 4, 3, 2]
):
    model = model.to(device)
    model.eval()

    results = {}

    print("\n" + "=" * 50)
    print("Evaluating Float32 baseline...")
    _, float_acc = evaluate(model, test_loader, None, device)
    results["float32"] = float_acc
    print(f"Float32 Accuracy: {float_acc:.4f}%")

    print("\n" + "=" * 50)
    print("Calibrating quantizers...")
    calibrate_model(model, train_loader, num_batches=25, device=device)
    print("Calibration complete.")

    for bits in bits_list:
        print("\n" + "=" * 50)
        print(f"Evaluating {bits}-bit quantization...")

        model.enable_quantization(bits=bits)
        _, quant_acc = evaluate(model, test_loader, None, device)
        results[f"int{bits}"] = quant_acc

        accuracy_drop = float_acc - quant_acc
        print(f"{bits}-bit Accuracy: {quant_acc:.4f}% (drop: {accuracy_drop:.4f}%)")

        mlflow.log_metric(f"quant_acc_int{bits}", quant_acc)
        mlflow.log_metric(f"quant_drop_int{bits}", accuracy_drop)
        model.disable_quantization()

    return results


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["train", "quantize", "full"]),
    default="full",
    help="Mode",
)
@click.option("--epochs", type=int, default=300, help="No of training epochs")
@click.option("--batch-size", type=int, default=128, help="Batch size")
@click.option(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint for loading (for quantize mode)",
)
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Device (cuda/cpu)",
)
@click.option(
    "--name",
    type=str,
    default="densenet_bc_190_40",
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
    default="densenet-cifar10-quantization",
    help="MLflow experiment name",
)
def main(
    mode, epochs, batch_size, checkpoint, device, name, mlflow_uri, experiment_name
):
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)
    print(f"MLflow tracking URI: {mlflow_uri}")
    print(f"MLflow experiment: {experiment_name}")

    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    device_str = device
    device = torch.device(device)
    print(f"Using device: {device}")

    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)

    model = densenet_bc_190_40(num_classes=10)
    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")

    checkpoint_path = f"{MODELS_PATH}/{name}.pt"

    with mlflow.start_run(run_name=name):
        mlflow.log_param("mode", mode)
        mlflow.log_param("model_name", name)
        mlflow.log_param("model_architecture", "densenet_bc_190_40")
        mlflow.log_param("num_parameters", num_params)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("device", device_str)
        mlflow.log_param("dataset", "CIFAR-10")
        mlflow.log_param("num_classes", 10)

        if mode == "train" or mode == "full":
            print("\n" + "=" * 50)
            print("Starting training...")
            print("=" * 50)

            mlflow.log_param("epochs", epochs)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_param("momentum", 0.9)
            mlflow.log_param("weight_decay", 1e-4)
            mlflow.log_param("lr_schedule", "MultiStepLR")
            mlflow.log_param(
                "lr_milestones", f"[{int(epochs * 0.5)}, {int(epochs * 0.75)}]"
            )
            mlflow.log_param("optimizer", "SGD_Nesterov")

            model, history, best_acc = train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                epochs=epochs,
                checkpoint_path=checkpoint_path,
            )

            history_path = f"{RESULTS_PATH}/{name}_history.json"
            with open(history_path, "w") as f:
                json.dump(history, f, indent=2)
            print(f"Training history saved to {history_path}")

            mlflow.log_artifact(history_path)
            if os.path.exists(checkpoint_path):
                mlflow.log_artifact(checkpoint_path)

        if mode == "quantize" or mode == "full":
            if mode == "quantize":
                if checkpoint is None:
                    checkpoint = checkpoint_path

                if not os.path.exists(checkpoint):
                    raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

                print(f"\nLoading checkpoint: {checkpoint}")
                model.load_state_dict(torch.load(checkpoint, map_location=device))
            elif mode == "full":
                if os.path.exists(checkpoint_path):
                    model.load_state_dict(
                        torch.load(checkpoint_path, map_location=device)
                    )

            print("\n" + "=" * 50)
            print("Starting quantization experiments...")
            print("=" * 50)

            mlflow.log_param("quantization_method", "post_training_quantization")
            mlflow.log_param("calibration_batches", 25)
            mlflow.log_param("bits_list", "[16, 8, 6, 4, 3, 2]")

            quant_results = quantize_and_evaluate(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                device=device,
                bits_list=[16, 8, 6, 4, 3, 2],
            )

            results_path = f"{RESULTS_PATH}/{name}_quantization.json"
            with open(results_path, "w") as f:
                json.dump(quant_results, f, indent=2)

            print("\n" + "=" * 50)
            print("Quantization Results Summary:")
            print("=" * 50)
            for key, acc in quant_results.items():
                print(f"  {key}: {acc:.4f}%")

            print(f"\nResults saved to {results_path}")

            mlflow.log_artifact(results_path)
            mlflow.log_metric("float32_accuracy", quant_results.get("float32", 0))

    print("\nDone!")


if __name__ == "__main__":
    main()
