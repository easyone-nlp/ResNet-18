import argparse
import importlib.util
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from pgd_attack import PGDAttack


ROOT = Path(__file__).resolve().parent


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset_config(dataset_name: str):
    if dataset_name == "mnist":
        mean = (0.1307, 0.1307, 0.1307)
        std = (0.3081, 0.3081, 0.3081)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        dataset = datasets.MNIST(root=str(ROOT / "MNIST" / "data"), train=False, download=True, transform=transform)
        model_dir = ROOT / "MNIST"
    elif dataset_name == "cifar10":
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
        dataset = datasets.CIFAR10(
            root=str(ROOT / "CIFAR-10" / "data"),
            train=False,
            download=True,
            transform=transform,
        )
        model_dir = ROOT / "CIFAR-10"
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    clamp_min = torch.tensor([[(0.0 - m) / s] for m, s in zip(mean, std)], dtype=torch.float32).view(1, len(mean), 1, 1)
    clamp_max = torch.tensor([[(1.0 - m) / s] for m, s in zip(mean, std)], dtype=torch.float32).view(1, len(mean), 1, 1)
    return dataset, model_dir, clamp_min, clamp_max


def load_model(model_dir: Path, num_classes: int = 10):
    resnet_module = load_module(f"{model_dir.name}_resnet", model_dir / "resnet.py")
    model = resnet_module.resnet18(num_classes=num_classes)

    checkpoint = torch.load(model_dir / "checkpoint.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_dataloader(dataset_name: str, num_samples: int, batch_size: int):
    dataset, model_dir, clamp_min, clamp_max = dataset_config(dataset_name)
    subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return dataloader, model_dir, clamp_min, clamp_max


def run_attack(
    dataset_name: str,
    attack_type: str,
    epsilons,
    target_class: int,
    num_samples: int,
    batch_size: int,
    num_steps: int,
    step_size: float,
    random_start: bool,
):
    dataloader, model_dir, clamp_min, clamp_max = build_dataloader(dataset_name, num_samples, batch_size)
    model = load_model(model_dir=model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    target = target_class if attack_type == "targeted" else None
    attack = PGDAttack(
        model=model,
        epsilons=epsilons,
        step_size=step_size,
        num_steps=num_steps,
        test_dataloader=dataloader,
        device=device,
        target=target,
        clamp_min=clamp_min,
        clamp_max=clamp_max,
        max_samples=num_samples,
        random_start=random_start,
    )
    results = attack.run()

    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    image_path = results_dir / f"{dataset_name}_{attack_type}_pgd.png"
    json_path = results_dir / f"{dataset_name}_{attack_type}_pgd.json"
    attack.visualize(save_path=image_path)
    json_path.write_text(json.dumps(results, indent=2))
    print(f"Saved attack summary to {json_path}")
    print(f"Saved visualization to {image_path}")


def parse_epsilons(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description="Run PGD on the trained ResNet-18 models.")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], required=True)
    parser.add_argument("--attack", choices=["targeted", "untargeted"], required=True)
    parser.add_argument("--epsilons", type=str, default="0.05,0.1,0.2,0.3")
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--random_start", action="store_true", help="Start PGD from a random point inside the epsilon-ball.")
    args = parser.parse_args()

    run_attack(
        dataset_name=args.dataset,
        attack_type=args.attack,
        epsilons=parse_epsilons(args.epsilons),
        target_class=args.target_class,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        step_size=args.step_size,
        random_start=args.random_start,
    )


if __name__ == "__main__":
    main()
