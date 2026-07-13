"""
transformer_no_softmax.py
-------------------------
Transformer Attention 블록 (softmax 제외)
reduce 없이 gemm/batch_gemm/elementwise만 사용

구조:
  x [1, seq, dim]
  → Q/K/V projection (gemm)
  → QK^T (batch_gemm)
  → scale (mul)
  → AV (batch_gemm)
  → output projection (gemm)
  → FFN (gemm + silu + gemm)
"""
import torch

class AttentionNoSoftmax(torch.nn.Module):
    def __init__(self, dim=64, n_heads=4, seq=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.seq = seq

        self.wq = torch.nn.Linear(dim, dim, bias=False)
        self.wk = torch.nn.Linear(dim, dim, bias=False)
        self.wv = torch.nn.Linear(dim, dim, bias=False)
        self.wo = torch.nn.Linear(dim, dim, bias=False)

        # FFN
        self.gate = torch.nn.Linear(dim, dim * 4, bias=False)
        self.up   = torch.nn.Linear(dim, dim * 4, bias=False)
        self.down = torch.nn.Linear(dim * 4, dim, bias=False)

    def forward(self, x):
        B, S, D = x.shape  # [1, seq, dim]

        # Q/K/V projection
        q = self.wq(x)  # [1, seq, dim]
        k = self.wk(x)
        v = self.wv(x)

        # Reshape to [B, n_heads, seq, head_dim]
        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # QK^T scaled (no softmax)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # AV
        out = torch.matmul(attn, v)

        # Reshape back
        out = out.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection
        out = self.wo(out)

        # FFN (SiLU)
        out = self.down(torch.nn.functional.silu(self.gate(out)) * self.up(out))

        return out


def get_model():
    return AttentionNoSoftmax()

def get_sample_inputs():
    return (torch.randn(1, 8, 64),)
