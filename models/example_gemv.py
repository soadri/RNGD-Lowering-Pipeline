"""
example_gemv.py
---------------
linalg.matvec → rngd.gemv 변환 검증 모델
A: [256, 2048] bf16, x: [2048] bf16 → y: [256] bf16
"""
import torch

class GemvModel(torch.nn.Module):
    def forward(self, A, x):
        return torch.mv(A, x)

def get_model():
    return GemvModel()

def get_sample_inputs():
    return (
        torch.randn(256, 2048).to(torch.bfloat16),
        torch.randn(2048).to(torch.bfloat16),
    )
