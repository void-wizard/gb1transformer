import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .model import ScratchTransformer
from .tokenizer import encode_sequence


class FitnessPredictor:
    """Transformer 与 ESM2 的批量预测接口"""

    def __init__(
        self,
        model_path,
        model_type="scratch",
        device="cpu",
    ):
        self.model_type = model_type
        self.device = torch.device(device)

        if model_type == "scratch":
            checkpoint = torch.load(
                model_path,
                map_location="cpu",
                weights_only=True,
            )

            self.model = ScratchTransformer(
                **checkpoint["model_config"]
            )
            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

        elif model_type == "esm2":
            # 加载微调权重和 tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self.model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_path,
                    local_files_only=True,
                )
            )

        else:
            raise ValueError(
                "model_type 必须是 scratch 或 esm2"
            )

        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def predict(self, sequences: list[str]) -> list[float]:
        """批量预测对数富集值，返回顺序与输入序列一致"""
        predictions = []

        # 分批推理控制内存占用
        for start in range(0, len(sequences), 32):
            batch = sequences[start:start + 32]

            token_ids = [
                encode_sequence(sequence)
                for sequence in batch
            ]

            if self.model_type == "scratch":
                inputs = torch.tensor(
                    token_ids,
                    dtype=torch.long,
                    device=self.device,
                )
                scores = self.model(inputs)

            else:
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                ).to(self.device)

                scores = self.model(**inputs).logits.squeeze(-1)

            predictions.extend(scores.cpu().tolist())

        return predictions