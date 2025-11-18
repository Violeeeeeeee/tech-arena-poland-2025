import torch
import torch.nn as nn
import torch.nn.functional as F

# Клас, що об'єднує Pooling та Dense Layer
class DensePoolingHead(nn.Module):
    """
    Sentence Embedding Student Head: combines Hybrid Attention Pooling,
    MLP (Dense Layer) with Dropout, Residual Connection, and final LayerNorm/Normalize.

    Вхід:
    - x (torch.Tensor): Token embeddings [Batch, Seq_Len, Hidden_Dim=768]
    """

    def __init__(self, input_dim: int = 768, output_dim: int = 64, dropout: float = 0.9, activation: str = 'gelu', **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 1. Hybrid Pooling (Learned Hybrid Attention)
        self.gate = nn.Parameter(torch.randn(input_dim) / (input_dim ** 0.5))
        self.alpha = nn.Parameter(torch.tensor(0.7)) # Learned weight for Attention vs Mean
        self.pool_ln = nn.LayerNorm(input_dim)

        # 2. MLP Head (Dense Layer)
        hidden = max(256, output_dim * 3)
        act_func = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden),
            act_func,
            nn.Dropout(dropout),
            nn.Linear(hidden, output_dim)
        )

        # 3. Residual Connection
        self.out_ln = nn.LayerNorm(output_dim)
        self.res_gate = nn.Parameter(torch.zeros(output_dim))
        self.res_proj = nn.Linear(input_dim, output_dim)


    # Метод для SBERT: повертає розмірність ембеддінгу
    def get_sentence_embedding_dimension(self) -> int:
        return self.output_dim

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:

        # 1. Pooling
        # x: [B, L, D]
        # Маска на токени, які не є нульовими (це важливо для змінної довжини)
        mask = (x.abs().sum(dim=-1) > 0)

        # Attention scores
        scores = (x @ self.gate) / (self.input_dim ** 0.5)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1).unsqueeze(-1)
        attn_pooled = (attn * x).sum(dim=1) # [B, D]

        # Mean Pooling
        mask_expanded = mask.unsqueeze(-1)
        mean_pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)

        # Hybrid Combination
        pooled = self.alpha * attn_pooled + (1 - self.alpha) * mean_pooled
        pooled = self.pool_ln(pooled)

        # 2. MLP Forward
        out = self.mlp(pooled)
        out = self.out_ln(out)

        # 3. Residual Connection
        pooled_proj = self.res_proj(pooled)
        out = self.res_gate * out + (1 - self.res_gate) * pooled_proj # Вихід MLP з Residuals

        # 4. Final Normalize
        # F.normalize(out, p=2, dim=-1) # Залишаємо це опціонально, але для SBERT це важливо

        # У вашому старому коді була фінальна нормалізація. В SBERT це часто окремий модуль
        return F.normalize(out, p=2, dim=-1)
