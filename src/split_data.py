import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "splits"

SEED = 42

def split_data(frame: pd.DataFrame) -> pd.DataFrame:
    """添加 split 列，野生型固定归入训练集"""

    # 固定划分前的样本顺序
    frame = frame.sort_values("id").reset_index(drop=True).copy()

    if frame["sequence"].duplicated().any():
        raise ValueError("数据中存在重复序列，请先处理后再划分")

    # 野生型固定归入训练集
    variants = frame.loc[frame["n_mutations"] > 0]

    # 训练集占非野生型样本的 60%
    _, remaining = train_test_split(
        variants,
        test_size=0.40,
        random_state=SEED,
    )

    # 候选池占非野生型样本的 20%
    pool, remaining = train_test_split(
        remaining,
        test_size=0.50,
        random_state=SEED,
    )

    # 验证集和测试集各占 10%
    validation, test = train_test_split(
        remaining,
        test_size=0.50,
        random_state=SEED,
    )

    frame["split"] = "train"

    for name, subset in (
        ("validation", validation),
        ("test", test),
        ("pool", pool),
    ):
        frame.loc[subset.index, "split"] = name

    return frame


def main():
    """划分数据并保存文件"""

    frame = pd.read_csv(DATA_DIR / "processed.csv")
    frame = split_data(frame)

    private_dir = OUTPUT_DIR / "private"
    private_dir.mkdir(parents=True, exist_ok=True)

    for name in ("train", "validation"):
        subset = frame.loc[frame["split"] == name]
        subset.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)

    # 测试集和候选池标签单独保存
    for name in ("test", "pool"):
        subset = frame.loc[frame["split"] == name]
        subset.to_csv(private_dir / f"{name}.csv", index=False)

    # 公开候选移除可推算 fitness 的原始计数
    public_columns = [
        "id",
        "sequence",
        "mutations",
        "n_mutations",
    ]
    candidates = frame.loc[
        frame["split"] == "pool",
        public_columns,
    ]
    candidates.to_csv(OUTPUT_DIR / "candidates.csv", index=False)

    frame[["id", "split"]].to_csv(
        OUTPUT_DIR / "assignments.csv",
        index=False,
    )

    metadata = {
        "seed": SEED,
        "fractions_of_non_wt": {
            "train": 0.60,
            "validation": 0.10,
            "test": 0.10,
            "pool": 0.20,
        },
        "wild_type_split": "train",
        "counts": {
            name: int(count)
            for name, count in frame["split"].value_counts().items()
        },
    }

    (OUTPUT_DIR / "split_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(frame["split"].value_counts())
    print(f"划分结果已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()