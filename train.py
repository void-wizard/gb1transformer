from pathlib import Path

import matplotlib

# 使用无界面绘图后端
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn

from src.data_loader import create_data_loader
from src.model import ScratchTransformer
from src.trainer import train_one_epoch, evaluate
from plot_training import plot_history


ROOT = Path(__file__).resolve().parent
EPOCHS = 30
BATCH_SIZE = 256

MODEL_CONFIG = {
    "d_model": 128,
    "nhead": 4,
    "num_layers": 4,
    "dropout": 0.1,
}


def main():
    """训练 Transformer 并评估最佳模型"""

    torch.manual_seed(42)
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"运行设备：{device}", flush=True)

    # 输出目录已存在时停止，避免覆盖训练结果
    output = (
        ROOT / "outputs"
        / "GB1transformer"
    )
    output.mkdir(parents=True)

    data_dir = ROOT / "data" / "splits"
    train_data = pd.read_csv(data_dir / "train.csv")
    val_data = pd.read_csv(data_dir / "validation.csv")

    train_loader = create_data_loader(
        train_data, BATCH_SIZE, shuffle=True
    )
    val_loader = create_data_loader(val_data, BATCH_SIZE)

    model = ScratchTransformer(**MODEL_CONFIG).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-2,
    )

    best_score = -float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}", flush=True)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_metrics = evaluate(model, val_loader, device)

        history.append({
            "epoch": epoch,
            "train_mse": train_loss,
            **{
                f"val_{key}": value
                for key, value in val_metrics.items()
            },
        })

        pd.DataFrame(history).to_csv(
            output / "history.csv", index=False
        )

        if val_metrics["spearman"] > best_score:
            best_score = val_metrics["spearman"]

            torch.save({
                "epoch": epoch,
                "model_config": MODEL_CONFIG,
                "model_state_dict": model.state_dict(),
            }, output / "best.pt")

        print(
            f"训练 MSE={train_loss:.4f}，"
            f"验证 MSE={val_metrics['mse']:.4f}，"
            f"Spearman={val_metrics['spearman']:.4f}",
            flush=True,
        )

    # 训练 MSE 为参数更新期间的样本加权均值
    pd.DataFrame(history).plot(
        x="epoch",
        y=["train_mse", "val_mse"],
        ylabel="MSE",
    )
    plt.tight_layout()
    plt.savefig(output / "loss.png", dpi=150)
    plt.close()

    if not (output / "best.pt").exists():
        raise RuntimeError(
            "验证 Spearman 始终无效，未保存最佳模型"
        )

    # 使用验证集最佳权重评估测试集
    checkpoint = torch.load(
        output / "best.pt",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_data = pd.read_csv(
        data_dir / "private" / "test.csv"
    )
    test_loader = create_data_loader(test_data, BATCH_SIZE)
    test_metrics = evaluate(model, test_loader, device)

    pd.DataFrame([test_metrics]).to_csv(
        output / "test_metrics.csv", index=False
    )
    plot_history(output / "history.csv")

    print(f"最佳模型来自第 {checkpoint['epoch']} 轮")
    print("测试结果：", test_metrics)
    print(f"结果目录：{output}")


if __name__ == "__main__":
    main()