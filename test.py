import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_existing_main(target_dir: Path, model: str, epochs: int, resume: bool):
    command = [sys.executable, "main.py", "--model", model, "--epochs", str(epochs)]
    if resume:
        command.append("--resume")

    subprocess.run(command, cwd=target_dir, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the existing MNIST and CIFAR-10 training code in one command."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Model architecture.",
    )
    parser.add_argument(
        "--mnist_epochs",
        type=int,
        default=10,
        help="Number of training epochs for MNIST.",
    )
    parser.add_argument(
        "--cifar10_epochs",
        type=int,
        default=30,
        help="Number of training epochs for CIFAR-10.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the checkpoint in the selected dataset folder.",
    )
    args = parser.parse_args()

    targets = [
        (ROOT / "MNIST", args.mnist_epochs),
        (ROOT / "CIFAR-10", args.cifar10_epochs),
    ]
    for target_dir, epochs in targets:
        print(f"Running training in {target_dir}")
        run_existing_main(target_dir=target_dir, model=args.model, epochs=epochs, resume=args.resume)


if __name__ == "__main__":
    main()
