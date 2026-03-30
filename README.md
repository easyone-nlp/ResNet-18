# ResNet-18

ResNet-18 training and adversarial attack scripts for MNIST and CIFAR-10.

## Repository Setup

```bash
git clone https://github.com/easyone-nlp/ResNet-18.git
cd ResNet-18
```

If you already have a Python environment for PyTorch, activate it before running the scripts.
This project uses `torch`, `torchvision`, and `matplotlib`.

## Project Structure

- `MNIST/`: MNIST ResNet training code and checkpoint
- `CIFAR-10/`: CIFAR-10 ResNet training code and checkpoint
- `test.py`: one entrypoint for training + FGSM + PGD
- `run_fgsm.py`: run FGSM only
- `run_pgd.py`: run PGD only
- `results/`: saved attack JSON summaries and visualization images

## Run Everything With One Command

`test.py` trains the selected model first and then runs FGSM and PGD attacks.
By default it runs:
- MNIST and CIFAR-10
- targeted and untargeted FGSM
- targeted and untargeted PGD

```bash
python test.py
```

Useful options:

```bash
python test.py --skip_train
python test.py --datasets mnist
python test.py --datasets cifar10 --attack_modes targeted
python test.py --pgd_num_steps 40 --pgd_step_size 0.01
python test.py --target_class 8 --num_samples 100 --batch_size 16
```

Default training epochs:
- MNIST: `10`
- CIFAR-10: `30`

## Train Models Only

Train MNIST only:

```bash
cd MNIST
python main.py --model resnet18 --epochs 10
```

Train CIFAR-10 only:

```bash
cd CIFAR-10
python main.py --model resnet18 --dataset cifar10 --epochs 30
```

Resume training from an existing checkpoint:

```bash
python main.py --model resnet18 --epochs 10 --resume
```

## FGSM

FGSM is a one-step attack that perturbs the input using the sign of the gradient.
This repository supports both targeted and untargeted FGSM.

Untargeted FGSM example:

```bash
python run_fgsm.py --dataset cifar10 --attack untargeted --num_samples 100 --batch_size 16
```

Targeted FGSM example:

```bash
python run_fgsm.py --dataset cifar10 --attack targeted --target_class 8 --num_samples 100 --batch_size 16
```

Custom epsilon values:

```bash
python run_fgsm.py --dataset mnist --attack targeted --target_class 8 --epsilons 0.05,0.1,0.2,0.3
```

## PGD

PGD is an iterative version of FGSM.
This repository supports both targeted and untargeted PGD.

Untargeted PGD example:

```bash
python run_pgd.py --dataset cifar10 --attack untargeted --num_samples 100 --batch_size 16 --num_steps 10 --step_size 0.01
```

Targeted PGD example:

```bash
python run_pgd.py --dataset cifar10 --attack targeted --target_class 8 --num_samples 100 --batch_size 16 --num_steps 10 --step_size 0.01
```

MNIST PGD example:

```bash
python run_pgd.py --dataset mnist --attack targeted --target_class 8 --epsilons 0.3 --num_steps 40 --step_size 0.01
```

PGD hyperparameters:
- `--epsilons`: total perturbation budget
- `--step_size`: step size per iteration
- `--num_steps`: number of iterations

## Output Files

Attack results are saved to `results/`.
Each run writes:
- a JSON summary file
- a PNG visualization file

Examples:
- `results/cifar10_targeted_fgsm.json`
- `results/cifar10_targeted_fgsm.png`
- `results/cifar10_targeted_pgd.json`
- `results/cifar10_targeted_pgd.png`

## Notes

- Targeted attacks require `--target_class`.
- `run_fgsm.py` and `run_pgd.py` load the trained checkpoint from each dataset folder.
- `test.py` uses a non-interactive matplotlib backend for subprocess runs so the full pipeline can run in server environments.
