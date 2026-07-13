"""
llama_rmsnorm.py
----------------
Llama-3.1의 RMSNorm 서브모듈.

등장 op (8종):
  지원: arith.addf, arith.mulf, arith.divf, math.powf, math.rsqrt
  미지원: linalg.generic(reduce/mean), arith.truncf, linalg.fill
예상 커버리지: ~62.5%
"""
import torch


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, dim: int = 64, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def get_model():
    return LlamaRMSNorm(dim=64)


def get_sample_inputs():
    return (torch.randn(1, 64),)
