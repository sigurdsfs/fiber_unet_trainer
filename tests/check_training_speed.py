from __future__ import annotations

import argparse
import subprocess
import time

import torch

from fiberseg.config import load_config
from fiberseg.dataset import FiberDataModule
from fiberseg.lit_module import FiberSegmentationLitModule


def print_gpu_info():
    print("\n=== GPU info ===")

    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version:", torch.version.cuda)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        print(result.stdout.strip())
    except Exception as e:
        print("Could not run nvidia-smi:", e)


def benchmark_dataloader(cfg, num_batches: int):
    print("\n=== Dataloader benchmark ===")

    dm = FiberDataModule(cfg.data, cfg.augmentations)
    dm.setup("fit")

    loader = dm.train_dataloader()

    start = time.perf_counter()
    n_images = 0

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        images, masks = batch
        n_images += images.shape[0]

    elapsed = time.perf_counter() - start

    print(f"Batches loaded: {num_batches}")
    print(f"Images loaded: {n_images}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Batches/sec: {num_batches / elapsed:.3f}")
    print(f"Images/sec: {n_images / elapsed:.3f}")


def benchmark_train_steps(cfg, num_batches: int):
    print("\n=== Training-step benchmark ===")

    dm = FiberDataModule(cfg.data, cfg.augmentations)
    dm.setup("fit")

    loader = dm.train_dataloader()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FiberSegmentationLitModule(cfg.model, cfg.train).to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.perf_counter()
    n_images = 0

    for i, batch in enumerate(loader):
        if i >= num_batches:
            break

        images, masks = batch
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = model._loss(logits, masks)
        loss.backward()
        optimizer.step()

        n_images += images.shape[0]

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    print(f"Batches trained: {num_batches}")
    print(f"Images trained: {n_images}")
    print(f"Elapsed seconds: {elapsed:.2f}")
    print(f"Batches/sec: {num_batches / elapsed:.3f}")
    print(f"Images/sec: {n_images / elapsed:.3f}")

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Peak CUDA memory allocated: {peak_gb:.2f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-batches", type=int, default=50)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print_gpu_info()
    benchmark_dataloader(cfg, args.num_batches)
    benchmark_train_steps(cfg, args.num_batches)

    print("\n=== Config values affecting speed ===")
    print("batch_size:", cfg.data.batch_size)
    print("num_workers:", cfg.data.num_workers)
    print("patch_size:", cfg.data.patch_size)
    print("precision:", cfg.train.precision)
    print("matmul_precision:", cfg.train.matmul_precision)
    print("model:", cfg.model.architecture, cfg.model.encoder_name, cfg.model.encoder_weights)
    print("image_channels:", cfg.data.image_channels)


if __name__ == "__main__":
    main()