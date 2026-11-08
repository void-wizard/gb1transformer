import pandas as pd
from datasets import Dataset


def load_esm_data(csv_path, tokenizer):
    """读取序列和 fitness，返回 ESM 回归数据集"""

    frame = pd.read_csv(
        csv_path,
        usecols=["sequence", "fitness"],
    )

    frame = frame.rename(columns={"fitness": "labels"})
    frame["labels"] = frame["labels"].astype("float32")

    dataset = Dataset.from_pandas(
        frame,
        preserve_index=False,
    )

    def tokenize(batch):
        return tokenizer(
            batch["sequence"],
            truncation=False,
        )

    return dataset.map(
        tokenize,
        batched=True,
        remove_columns=["sequence"],
    )