from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.esm_data import load_esm_data
from src.metrics import compute_metrics


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "esm2"


def trainer_metrics(result):
    """计算 ESM 回归指标"""

    return compute_metrics(
        result.label_ids,
        # 模型输出为 [样本数, 1]
        result.predictions.squeeze(-1),
        top_k=100,
    )


def save_curves(trainer, output):
    """保存训练日志和验证指标曲线"""

    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(output / "history.csv", index=False)

    validation = history.dropna(subset=["eval_loss"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    history.dropna(subset=["loss"]).plot(
        x="epoch",
        y="loss",
        ax=axes[0, 0],
        label="Train",
    )
    validation.plot(
        x="epoch",
        y="eval_loss",
        ax=axes[0, 0],
        label="Validation",
    )
    axes[0, 0].set_ylabel("MSE")

    for ax, columns, title in [
        (
            axes[0, 1],
            ["eval_spearman", "eval_pearson"],
            "Correlation",
        ),
        (
            axes[1, 0],
            ["eval_top_k_hit"],
            "Top-k hit rate",
        ),
        (
            axes[1, 1],
            ["eval_top_k_mean_fitness"],
            "Top-k measured fitness",
        ),
    ]:
        validation.plot(
            x="epoch",
            y=columns,
            ax=ax,
            title=title,
        )

    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=200)
    plt.close(fig)


def main():
    """微调 ESM2 并评估最佳模型"""

    if torch.cuda.device_count() > 1:
        raise RuntimeError(
            "请用 CUDA_VISIBLE_DEVICES 选择一张 GPU"
        )

    set_seed(42)

    output = (
        ROOT / "outputs" / "esm2"
    )
    split_dir = ROOT / "data" / "splits"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    # 回归损失使用 MSE
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        num_labels=1,
        problem_type="regression",
        local_files_only=True,
    )

    train_data = load_esm_data(
        split_dir / "train.csv",
        tokenizer,
    )
    val_data = load_esm_data(
        split_dir / "validation.csv",
        tokenizer,
    )

    args = TrainingArguments(
        output_dir=str(output),
        num_train_epochs=12,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.05,

        # 按验证集 Spearman 选择最佳权重
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="spearman",
        greater_is_better=True,
        save_total_limit=2,

        bf16=(
            torch.cuda.is_available()
            and torch.cuda.is_bf16_supported()
        ),
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=val_data,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=trainer_metrics,
    )

    # 主干和回归头共同训练
    trainer.train()

    # 导出训练结束时已恢复的最佳权重
    trainer.save_model(str(output / "best_model"))
    trainer.save_state()
    save_curves(trainer, output)

    test_data = load_esm_data(
        split_dir / "private" / "test.csv",
        tokenizer,
    )
    test_metrics = trainer.evaluate(
        test_data,
        metric_key_prefix="test",
    )
    trainer.save_metrics("test", test_metrics)

    print("测试结果：", test_metrics)
    print(f"结果目录：{output}")


if __name__ == "__main__":
    main()