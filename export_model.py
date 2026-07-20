# export_model.py

"""
Run via: 

python export_model.py

"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DEFAULT_OUT_BASE = "exported_models"


def ask(prompt: str, default: str | None = None) -> str:
    if default:
        value = input(f"{prompt} [{default}]: ").strip()
        return value or default

    while True:
        value = input(f"{prompt}: ").strip()
        if value:
            return value
        print("Please enter a value.")


def find_checkpoints() -> list[Path]:
    roots = [Path("mlruns"), Path("lightning_logs"), Path(".")]
    seen: set[Path] = set()
    checkpoints: list[Path] = []

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*.ckpt"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            checkpoints.append(path)

    checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return checkpoints


def choose_checkpoint() -> Path:
    checkpoints = find_checkpoints()

    if not checkpoints:
        return Path(ask("Path to checkpoint .ckpt"))

    print("\nFound checkpoints:")
    for i, path in enumerate(checkpoints[:30], start=1):
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"{i:2d}. {path} ({size_mb:.1f} MB)")

    print("\nChoose a checkpoint number, or paste a different checkpoint path.")
    choice = ask("Checkpoint", "1")

    try:
        idx = int(choice)
        if 1 <= idx <= min(30, len(checkpoints)):
            return checkpoints[idx - 1]
    except ValueError:
        pass

    return Path(choice)


def find_mlflow_run_dir(checkpoint: Path) -> Path | None:
    """
    Expected local MLflow layout:
      mlruns/<experiment_id>/<run_id>/checkpoints/*.ckpt
      mlruns/<experiment_id>/<run_id>/artifacts/...
    """
    checkpoint = checkpoint.resolve()

    for parent in checkpoint.parents:
        if parent.name == "mlruns":
            return None

        if parent.parent.name == "mlruns":
            # parent is experiment folder, not run folder
            return None

        if parent.parent.parent.name == "mlruns":
            # parent is run folder:
            # mlruns / experiment_id / run_id
            return parent

    return None


def find_config_for_checkpoint(checkpoint: Path) -> Path | None:
    run_dir = find_mlflow_run_dir(checkpoint)
    if run_dir is None:
        return None

    candidates = [
        run_dir / "artifacts" / "config" / "resolved_config.yaml",
        run_dir / "artifacts" / "resolved_config.yaml",
        run_dir / "artifacts" / "config.yaml",
        run_dir / "artifacts" / "config" / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: search artifacts for likely config names
    artifacts = run_dir / "artifacts"
    if artifacts.exists():
        matches = list(artifacts.rglob("*config*.yaml")) + list(artifacts.rglob("*config*.yml"))
        if matches:
            matches.sort(key=lambda p: len(str(p)))
            return matches[0]

    return None


def main() -> int:
    print("TorchScript model export")
    print("========================\n")

    checkpoint = choose_checkpoint()

    if not checkpoint.exists():
        print(f"Checkpoint not found: {checkpoint}")
        return 1

    config = find_config_for_checkpoint(checkpoint)

    if config is not None:
        print("\nDetected config:")
        print(f"  {config}")
        use_detected = ask("Use this config? yes/no", "yes").lower()
        if use_detected not in {"y", "yes"}:
            config = None

    if config is None:
        config = Path(ask("Training config path"))

    if not config.exists():
        print(f"Config not found: {config}")
        return 1

    model_name = ask("Exported model name", checkpoint.stem)
    out_dir = ask("Output folder", str(Path(DEFAULT_OUT_BASE) / model_name))
    device = ask("Export device: cpu or cuda", "cpu").lower()

    if device not in {"cpu", "cuda"}:
        print("Invalid device. Use 'cpu' or 'cuda'.")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "fiberseg.tools.export_torchscript",
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--out-dir",
        out_dir,
        "--model-name",
        model_name,
        "--device",
        device,
    ]

    print("\nRunning export:\n")
    print(" ".join(f'"{x}"' if " " in x else x for x in cmd))
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\nExport finished successfully.")
        print(f"Exported package: {Path(out_dir).resolve()}")
    else:
        print("\nExport failed.")

    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())