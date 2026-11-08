from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

# 突变位点从 1 开始编号
WT_SEQUENCE = (
    "MQYKLILNGKTLKGETTTEAVDAATAEKVFKQY"
    "ANDNGVDGEWTYDDATKTFTVTE"
)

def make_record(substitutions, input_count, selection_count):
    # 将突变和计数组合为变体记录

    sequence = list(WT_SEQUENCE)
    mutation_names = []

    for original, position, replacement in sorted(
        substitutions, key=lambda item: item[1]
    ):
        sequence[position - 1] = replacement
        mutation_names.append(f"{original}{position}{replacement}")

    return {
        "sequence": "".join(sequence),
        "mutations": ":".join(mutation_names) or "WT",
        "n_mutations": len(substitutions),
        "input_count": input_count,
        "selection_count": selection_count,
    }

def prepare_data():
    # 合并原始数据并计算 WT 归一化的对数富集值

    singles = pd.read_csv(
        DATA_DIR / "single_mutants.csv", encoding="utf-8-sig"
    )
    doubles = pd.read_csv(
        DATA_DIR / "double_mutants.csv", encoding="utf-8-sig"
    )
    wild_type = pd.read_csv(
        DATA_DIR / "wild_type.csv", encoding="utf-8-sig"
    ).iloc[0]

    wt_input = int(wild_type["Input Count"])
    wt_selection = int(wild_type["Selection Count"])

    records = [make_record([], wt_input, wt_selection)]

    for row in singles.itertuples(index=False, name=None):
        original, position, replacement, input_count, selection_count = row

        records.append(
            make_record(
                [(original, int(position), replacement)],
                input_count,
                selection_count,
            )
        )

    for row in doubles.itertuples(index=False, name=None):
        # 忽略末尾的单点标签，双点标签由自身计数计算
        a, p, b, c, q, d, input_count, selection_count, *_ = row

        records.append(
            make_record(
                [(a, int(p), b), (c, int(q), d)],
                input_count,
                selection_count,
            )
        )

    frame = pd.DataFrame(records)

    # 伪计数用于处理零计数
    pseudocount = 0.5
    wt_log_enrichment = np.log(
        (wt_selection + pseudocount) / (wt_input + pseudocount)
    )

    frame["fitness"] = np.log(
        (frame["selection_count"] + pseudocount)
        / (frame["input_count"] + pseudocount)
    ) - wt_log_enrichment

    # 按突变描述生成稳定 ID
    frame = frame.sort_values("mutations").reset_index(drop=True)
    frame.insert(0, "id", frame.index)

    return frame

if __name__ == "__main__":
    data = prepare_data()

    output_path = DATA_DIR / "processed.csv"
    data.to_csv(output_path, index=False)

    print(data.head())
    print(f"Total records: {len(data)}")
    print(f"Saved to: {output_path}")