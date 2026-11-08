import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_history(csv_path):
    """读取训练日志并在同目录保存指标曲线"""

    csv_path = Path(csv_path)
    history = pd.read_csv(csv_path)

    panels = [
        (
            ["train_mse", "val_mse"],
            "Training and validation loss",
            "MSE",
        ),
        (
            ["val_spearman"],
            "Validation Spearman",
            "Correlation",
        ),
        (
            ["val_pearson"],
            "Validation Pearson",
            "Correlation",
        ),
        (
            ["val_top_k_hit"],
            "Validation Top-k hit rate",
            "Hit rate",
        ),
        (
            ["val_top_k_mean_fitness"],
            "Selected Top-k measured fitness",
            "Mean fitness",
        ),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    for ax, (columns, title, ylabel) in zip(axes.flat, panels):
        history.plot(
            x="epoch",
            y=columns,
            ax=ax,
            title=title,
            ylabel=ylabel,
        )
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)

    axes[1, 2].set_visible(False)
    fig.tight_layout()

    output_path = csv_path.parent / "training_curves.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print(f"曲线已保存至：{output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "history",
        type=Path,
        help="history.csv 的路径",
    )
    args = parser.parse_args()
    plot_history(args.history)