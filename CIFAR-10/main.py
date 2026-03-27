## 코드 출처 : https://velog.io/@tolerance0718/PyTorch-%EA%B8%B0%EB%B0%98-ResNet-%EB%AA%A8%EB%8D%B8-%EA%B5%AC%ED%98%84-%EC%BD%94%EB%93%9C-%EB%A6%AC%EB%B7%B0
import argparse
from resnet import resnet18, resnet50
from train import train_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='resnet18',
                        help='Model type: resnet18 or resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='Dataset name: cifar10 or cifar100')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    args = parser.parse_args()

    # 자동 클래스 수 설정
    dataset_name = args.dataset.lower()
    if dataset_name == 'cifar10':
        num_classes = 10
    elif dataset_name == 'cifar100':
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use cifar10 or cifar100.")

    # 모델 선택
    if args.model == 'resnet18':
        model = resnet18(num_classes=num_classes)
    elif args.model == 'resnet50':
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {args.model}. Use resnet18 or resnet50.")

    # 학습 시작
    train_model(
        model=model,
        num_epochs=args.epochs,
        resume=args.resume,
        dataset_name=dataset_name,
        num_classes=num_classes
    )

if __name__ == '__main__':
    main()