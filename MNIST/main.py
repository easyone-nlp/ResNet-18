import argparse

from resnet import resnet18, resnet50
from train import train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        help="Model type: resnet18 or resnet50",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoint",
    )
    args = parser.parse_args()

    num_classes = 10

    if args.model == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif args.model == "resnet50":
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model}. Use resnet18 or resnet50.")

    train_model(
        model=model,
        num_epochs=args.epochs,
        resume=args.resume,
        num_classes=num_classes,
    )


if __name__ == "__main__":
    main()
