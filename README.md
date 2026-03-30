# ResNet-18

ResNet-18 training and FGSM attack scripts for MNIST and CIFAR-10.

## Training

- Train both datasets: `python test.py`
- Train only MNIST: `cd MNIST && python main.py --model resnet18 --epochs 10`
- Train only CIFAR-10: `cd CIFAR-10 && python main.py --model resnet18 --dataset cifar10 --epochs 30`

## FGSM

- Run targeted FGSM: `python run_fgsm.py --dataset cifar10 --attack targeted --target_class 8`
- Run untargeted FGSM: `python run_fgsm.py --dataset cifar10 --attack untargeted`

Generated result images and JSON summaries are written to `results/` and ignored by Git.
