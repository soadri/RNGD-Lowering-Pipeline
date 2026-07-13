"""
example_rmsnorm.py
------------------
CI 커버리지 분석 예시용 모델.
RMSNorm은 pow2/rsqrt/add/mul을 포함하며,
아직 미지원인 reduce(mean) 연산이 섞여 있어 부분 커버리지가 예상된다.
"""

import torch


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x^2 → mean → +eps → rsqrt → x * rsqrt_val * weight
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def get_model():
    return RMSNorm(dim=64)


def get_sample_inputs():
    return (torch.randn(1, 64),)
