import math
from pathlib import Path

import torch
import torch.nn as nn


class PGDAttack:
    def __init__(
        self,
        model,
        epsilons,
        step_size,
        num_steps,
        test_dataloader,
        device,
        target=None,
        clamp_min=None,
        clamp_max=None,
        max_samples=100,
    ):
        self.model = model
        self.epsilons = epsilons
        self.step_size = step_size
        self.num_steps = num_steps
        self.test_dataloader = test_dataloader
        self.device = device
        self.target = target
        self.max_samples = max_samples
        self.adv_examples = {}
        self.results = {}

        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    @property
    def is_targeted(self):
        return self.target is not None

    def _prepare_clamp(self, x):
        if self.clamp_min is None or self.clamp_max is None:
            return None, None

        clamp_min = self.clamp_min.to(x.device, x.dtype)
        clamp_max = self.clamp_max.to(x.device, x.dtype)
        return clamp_min, clamp_max

    def _project(self, original, perturbed, eps):
        delta = torch.clamp(perturbed - original, min=-eps, max=eps)
        projected = original + delta

        clamp_min, clamp_max = self._prepare_clamp(original)
        if clamp_min is not None and clamp_max is not None:
            projected = torch.max(torch.min(projected, clamp_max), clamp_min)
        return projected

    def _pgd(self, inputs, labels, eps):
        criterion = nn.CrossEntropyLoss()
        original = inputs.detach()
        perturbed = original.clone()

        for _ in range(self.num_steps):
            perturbed.requires_grad_(True)
            output = self.model(perturbed)

            if self.is_targeted:
                target_labels = torch.full_like(labels, fill_value=self.target, device=self.device)
                loss = criterion(output, target_labels)
            else:
                loss = criterion(output, labels)

            self.model.zero_grad()
            loss.backward()
            grad = perturbed.grad.detach()

            if self.is_targeted:
                updated = perturbed.detach() - self.step_size * grad.sign()
            else:
                updated = perturbed.detach() + self.step_size * grad.sign()

            perturbed = self._project(original=original, perturbed=updated, eps=eps).detach()

        return perturbed

    def _store_examples(self, eps, originals, adv_preds, adv_batch, valid_mask, success_mask):
        success_indices = torch.where(valid_mask & success_mask)[0].tolist()
        for idx in success_indices:
            if len(self.adv_examples[eps]) >= 5:
                break
            adv_ex = adv_batch[idx].detach().cpu()
            self.adv_examples[eps].append(
                {
                    "original_label": int(originals[idx].item()),
                    "adversarial_label": int(adv_preds[idx].item()),
                    "adversarial_tensor": adv_ex,
                }
            )

    def run(self):
        self.model.eval()

        for eps_real in self.epsilons:
            self.adv_examples[eps_real] = []
            evaluated = 0
            clean_correct = 0
            successful_attacks = 0

            for data, label in self.test_dataloader:
                if evaluated >= self.max_samples:
                    break

                remaining = self.max_samples - evaluated
                if data.size(0) > remaining:
                    data = data[:remaining]
                    label = label[:remaining]

                data, label = data.to(self.device), label.to(self.device)
                evaluated += data.size(0)

                with torch.no_grad():
                    output = self.model(data)
                    init_pred = output.argmax(dim=1)

                valid_mask = init_pred.eq(label)
                if self.is_targeted:
                    valid_mask = valid_mask & label.ne(self.target)

                clean_correct += int(valid_mask.sum().item())
                if not valid_mask.any():
                    continue

                perturbed_data = data.detach().clone()
                perturbed_data[valid_mask] = self._pgd(
                    inputs=data[valid_mask],
                    labels=label[valid_mask],
                    eps=eps_real,
                )

                with torch.no_grad():
                    adv_output = self.model(perturbed_data)
                    adv_pred = adv_output.argmax(dim=1)

                if self.is_targeted:
                    success_mask = adv_pred.eq(self.target)
                else:
                    success_mask = adv_pred.ne(label)

                successful_attacks += int((valid_mask & success_mask).sum().item())
                self._store_examples(
                    eps=eps_real,
                    originals=label,
                    adv_preds=adv_pred,
                    adv_batch=perturbed_data,
                    valid_mask=valid_mask,
                    success_mask=success_mask,
                )

            denominator = clean_correct if clean_correct > 0 else 1
            success_rate = successful_attacks / float(denominator)
            self.results[eps_real] = {
                "evaluated": evaluated,
                "clean_correct": clean_correct,
                "successful_attacks": successful_attacks,
                "success_rate": success_rate,
                "num_steps": self.num_steps,
                "step_size": self.step_size,
            }
            print(
                f"Epsilon: {eps_real}\t"
                f"Attack Success Rate = {successful_attacks} / {clean_correct} = {success_rate:.4f}"
            )

        return self.results

    def visualize(self, save_path=None):
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            print(f"Skipping visualization because matplotlib is unavailable: {exc}")
            return

        num_eps = len(self.adv_examples)
        max_cols = max((len(v) for v in self.adv_examples.values()), default=0)
        if num_eps == 0 or max_cols == 0:
            print("No adversarial examples collected for visualization.")
            return

        plt.figure(figsize=(max(8, max_cols * 3), max(4, num_eps * 3)))
        cnt = 0

        for eps, adv_examples in self.adv_examples.items():
            for index, data in enumerate(adv_examples):
                cnt += 1
                plt.subplot(num_eps, max_cols, cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if index == 0:
                    plt.ylabel(f"Eps: {eps}", fontsize=12)

                orig = data["original_label"]
                adv = data["adversarial_label"]
                adv_ex = data["adversarial_tensor"]

                if adv_ex.dim() == 3 and adv_ex.size(0) == 3:
                    img = adv_ex.permute(1, 2, 0).numpy()
                    img = (img - img.min()) / max(img.max() - img.min(), 1e-8)
                    plt.imshow(img)
                else:
                    img = adv_ex.squeeze().numpy()
                    plt.imshow(img, cmap="gray")

                plt.title(f"{orig} -> {adv}")

            cnt = int(math.ceil(cnt / max_cols) * max_cols)

        plt.tight_layout()
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.show()
