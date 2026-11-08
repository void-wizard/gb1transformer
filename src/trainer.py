import numpy as np
import torch

from .metrics import compute_metrics


def train_one_epoch(model, loader, optimizer, criterion, device):
    """训练一轮，返回样本加权平均损失"""

    model.train()
    total_loss = 0.0

    for step, batch in enumerate(loader, start=1):
        inputs = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()

        # 按样本数加权批次损失
        total_loss += loss.item() * labels.size(0)

        if step % 10 == 0:
            print(
                f"  批次 {step}/{len(loader)}，loss={loss.item():.4f}",
                flush=True,
            )

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    """计算整个评估集的回归与排序指标"""

    model.eval()
    targets, predictions = [], []

    for batch in loader:
        outputs = model(batch["input_ids"].to(device))
        targets.append(batch["labels"].numpy())
        predictions.append(outputs.cpu().numpy())

    # 合并全量预测后计算排序指标
    return compute_metrics(
        np.concatenate(targets),
        np.concatenate(predictions),
        top_k=100,
    )