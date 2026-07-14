"""
llama_full.py — Llama-3.1 Block 전체 (reduce 포함)
RMSNorm의 mean(x²) + Softmax 포함 → 미구현 연산 포함
"""
import torch
import torch.nn.functional as F


class LlamaFullBlock(torch.nn.Module):
    def __init__(self, dim: int = 64, n_heads: int = 4):
        super().__init__()
        self.dim      = dim
        self.n_heads  = n_heads
        self.head_dim = dim // n_heads

        # RMSNorm (mean 포함)
        self.norm_weight = torch.nn.Parameter(torch.ones(dim))

        # Attention
        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

        # FFN (SiLU)
        self.w1 = torch.nn.Linear(dim, dim * 2, bias=False)
        self.w2 = torch.nn.Linear(dim * 2, dim, bias=False)
        self.w3 = torch.nn.Linear(dim, dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMSNorm (reduce 포함)
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        x_norm = x / rms * self.norm_weight

        # Attention
        q = self.wq(x_norm)
        k = self.wk(x_norm)
        v = self.wv(x_norm)

        # Softmax (reduce 포함)
        scale  = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn   = F.softmax(scores, dim=-1)
        out    = torch.matmul(attn, v)
        out    = self.wo(out)

        # FFN (SiLU)
        ffn = self.w2(F.silu(self.w1(x_norm)) * self.w3(x_norm))

        return out + ffn


def get_model():
    return LlamaFullBlock()


def get_sample_inputs():
    return (torch.randn(1, 64),)
