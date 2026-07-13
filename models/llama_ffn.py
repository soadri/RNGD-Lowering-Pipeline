"""
llama_ffn.py
------------
Llama-3.1의 Feed Forward Network 서브모듈.
SiLU 활성화 함수(gate * up → down 구조)를 포함한다.

등장 op (8종):
  지원: arith.addf, arith.mulf, arith.divf, linalg.matmul
  미지원: arith.negf, linalg.fill, linalg.generic(SiLU), math.exp
예상 커버리지: ~50%
"""
import torch
import torch.nn.functional as F


class LlamaFFN(torch.nn.Module):
    def __init__(self, dim: int = 64, hidden: int = 256):
        super().__init__()
        self.gate = torch.nn.Linear(dim, hidden, bias=False)
        self.up   = torch.nn.Linear(dim, hidden, bias=False)
        self.down = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def get_model():
    return LlamaFFN(dim=64, hidden=256)


def get_sample_inputs():
    return (torch.randn(1, 64),)
