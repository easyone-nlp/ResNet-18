import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DEFAULT_ATTACK_CONFIGS = {
    ("mnist", "targeted"): {
        "num_samples": 102,
        "pgd_num_steps": 40,
        "pgd_step_size": 0.01,
    },
    ("mnist", "untargeted"): {
        "num_samples": 100,
        "pgd_num_steps": 40,
        "pgd_step_size": 0.01,
    },
    ("cifar10", "targeted"): {
        "num_samples": 142,
        "pgd_num_steps": 10,
        "pgd_step_size": 0.01,
    },
    ("cifar10", "untargeted"): {
        "num_samples": 122,
        "pgd_num_steps": 10,
        "pgd_step_size": 0.01,
    },
}


def parse_csv(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def run_command(command, cwd: Path):
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def run_training(dataset_name: str, model: str, epochs: int, resume: bool):
    if dataset_name == "mnist":
        target_dir = ROOT / "MNIST"
        command = [sys.executable, "main.py", "--model", model, "--epochs", str(epochs)]
    elif dataset_name == "cifar10":
        target_dir = ROOT / "CIFAR-10"
        command = [
            sys.executable,
            "main.py",
            "--model",
            model,
            "--dataset",
            "cifar10",
            "--epochs",
            str(epochs),
        ]
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if resume:
        command.append("--resume")

    run_command(command=command, cwd=target_dir)


def run_fgsm(dataset_name: str, attack_type: str, target_class: int, epsilons: str, num_samples: int, batch_size: int):
    command = [
        sys.executable,
        str(ROOT / "run_fgsm.py"),
        "--dataset",
        dataset_name,
        "--attack",
        attack_type,
        "--epsilons",
        epsilons,
        "--num_samples",
        str(num_samples),
        "--batch_size",
        str(batch_size),
    ]
    if attack_type == "targeted":
        command.extend(["--target_class", str(target_class)])

    run_command(command=command, cwd=ROOT)


def run_pgd(
    dataset_name: str,
    attack_type: str,
    target_class: int,
    epsilons: str,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    step_size: float,
):
    command = [
        sys.executable,
        str(ROOT / "run_pgd.py"),
        "--dataset",
        dataset_name,
        "--attack",
        attack_type,
        "--epsilons",
        epsilons,
        "--num_samples",
        str(num_samples),
        "--batch_size",
        str(batch_size),
        "--num_steps",
        str(num_steps),
        "--step_size",
        str(step_size),
    ]
    if attack_type == "targeted":
        command.extend(["--target_class", str(target_class)])

    run_command(command=command, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="Train models and then run FGSM and PGD attacks from one command."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Model architecture.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="mnist,cifar10",
        help="Comma-separated datasets to run: mnist, cifar10.",
    )
    parser.add_argument(
        "--attack_modes",
        type=str,
        default="targeted,untargeted",
        help="Comma-separated attack modes to run: targeted, untargeted.",
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
    parser.add_argument(
        "--skip_train",
        action="store_true",
        help="Skip model training and only run attacks.",
    )
    parser.add_argument(
        "--skip_fgsm",
        action="store_true",
        help="Skip FGSM runs.",
    )
    parser.add_argument(
        "--skip_pgd",
        action="store_true",
        help="Skip PGD runs.",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=8,
        help="Target class for targeted attacks.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Override the default number of evaluation samples for all attack runs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for attack evaluation.",
    )
    parser.add_argument(
        "--fgsm_epsilons",
        type=str,
        default="0.05,0.1,0.2,0.3",
        help="Comma-separated epsilon values for FGSM.",
    )
    parser.add_argument(
        "--pgd_epsilons",
        type=str,
        default="0.05,0.1,0.2,0.3",
        help="Comma-separated epsilon values for PGD.",
    )
    parser.add_argument(
        "--pgd_num_steps",
        type=int,
        default=None,
        help="Override the default number of PGD steps for all PGD runs.",
    )
    parser.add_argument(
        "--pgd_step_size",
        type=float,
        default=None,
        help="Override the default step size for all PGD runs.",
    )
    args = parser.parse_args()

    datasets = parse_csv(args.datasets)
    attack_modes = parse_csv(args.attack_modes)

    epoch_map = {
        "mnist": args.mnist_epochs,
        "cifar10": args.cifar10_epochs,
    }

    for dataset_name in datasets:
        if dataset_name not in epoch_map:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        if not args.skip_train:
            print(f"=== Training: {dataset_name} ===")
            run_training(
                dataset_name=dataset_name,
                model=args.model,
                epochs=epoch_map[dataset_name],
                resume=args.resume,
            )

        for attack_type in attack_modes:
            if attack_type not in {"targeted", "untargeted"}:
                raise ValueError(f"Unsupported attack mode: {attack_type}")

            attack_config = DEFAULT_ATTACK_CONFIGS[(dataset_name, attack_type)]
            num_samples = attack_config["num_samples"] if args.num_samples is None else args.num_samples
            pgd_num_steps = attack_config["pgd_num_steps"] if args.pgd_num_steps is None else args.pgd_num_steps
            pgd_step_size = attack_config["pgd_step_size"] if args.pgd_step_size is None else args.pgd_step_size

            if not args.skip_fgsm:
                print(f"=== FGSM: dataset={dataset_name}, attack={attack_type} ===")
                run_fgsm(
                    dataset_name=dataset_name,
                    attack_type=attack_type,
                    target_class=args.target_class,
                    epsilons=args.fgsm_epsilons,
                    num_samples=num_samples,
                    batch_size=args.batch_size,
                )

            if not args.skip_pgd:
                print(f"=== PGD: dataset={dataset_name}, attack={attack_type} ===")
                run_pgd(
                    dataset_name=dataset_name,
                    attack_type=attack_type,
                    target_class=args.target_class,
                    epsilons=args.pgd_epsilons,
                    num_samples=num_samples,
                    batch_size=args.batch_size,
                    num_steps=pgd_num_steps,
                    step_size=pgd_step_size,
                )


if __name__ == "__main__":
    main()
