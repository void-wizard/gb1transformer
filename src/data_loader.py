from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .tokenizer import encode_sequence

class GB1Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.sequences = frame["sequence"].tolist()

        self.labels = torch.tensor(
            frame["fitness"].to_numpy(),
            dtype=torch.float32,
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        input_ids = torch.tensor(
            encode_sequence(self.sequences[index]),
            dtype=torch.long,
        )

        return {
            "input_ids": input_ids,
            "labels": self.labels[index],
        }

def create_data_loader(
    frame: pd.DataFrame,
    batch_size: int = 256,
    shuffle: bool = False,
):
    """创建 GB1 DataLoader"""

    dataset = GB1Dataset(frame)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )

if __name__ == "__main__":
    data_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "processed.csv"
    )

    frame = pd.read_csv(data_path, nrows=8)
    loader = create_data_loader(frame, batch_size=4)

    batch = next(iter(loader))

    print("input_ids shape:", batch["input_ids"].shape)
    print("input_ids dtype:", batch["input_ids"].dtype)
    print("labels shape:", batch["labels"].shape)
    print("labels dtype:", batch["labels"].dtype)