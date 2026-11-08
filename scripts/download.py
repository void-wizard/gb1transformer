from pathlib import Path

import requests
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models" / "esm2"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

def download_data():
    data_url = (
        "https://raw.githubusercontent.com/gitter-lab/nn4dms/"
        "434e354828f306cbb6796033ef1169cbe5207350/source_data/gb1"
    )

    files = [
        "single_mutants.csv",
        "double_mutants.csv",
        "wild_type.csv",
        "mmc2.xlsx",
        "README.md",
    ]

    for filename in files:
        path = DATA_DIR / filename

        if path.exists():
            print(f"Already exists: {filename}")
            continue

        print(f"Downloading: {filename}")
        response = requests.get(f"{data_url}/{filename}", timeout=120)
        response.raise_for_status()
        path.write_bytes(response.content)

def download_esm():
    snapshot_download(
        repo_id="facebook/esm2_t12_35M_UR50D",
        revision="6fbf070e65b0b7291e7bbcd451118c216cff79d8",
        local_dir=MODEL_DIR,
        allow_patterns=[
            "model.safetensors",
            "*.json",
            "vocab.txt",
            "README.md",
        ],
    )

if __name__ == "__main__":
    download_data()
    download_esm()

    print("Download complete")