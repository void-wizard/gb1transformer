import torch
from torch import nn

from .tokenizer import AMINO_ACIDS, SEQUENCE_LENGTH


class ScratchTransformer(nn.Module):
    """完整序列的 Transformer Encoder 回归模型"""

    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        """配置编码器维度、头数、层数和 dropout

        d_model 必须能被 nhead 整除
        """
        super().__init__()

        self.token_embedding = nn.Embedding(
            num_embeddings=len(AMINO_ACIDS),
            embedding_dim=d_model,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=SEQUENCE_LENGTH,
            embedding_dim=d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
            # 等长序列不使用嵌套张量
            enable_nested_tensor=False,
        )

        # 分别初始化复制的编码层权重
        for block in self.encoder.layers:
            for parameter in block.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            # 线性输出允许负 fitness
            nn.Linear(d_model, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """将 [batch, 56] 的 token ID 转为 [batch] 的 fitness 预测"""
        positions = torch.arange(
            input_ids.size(1),
            device=input_ids.device,
        )

        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )

        # 双向注意力覆盖完整序列
        hidden = self.encoder(hidden)

        # 对所有位点做平均池化
        pooled = hidden.mean(dim=1)

        # 保留 batch 维度
        return self.head(pooled).squeeze(-1)