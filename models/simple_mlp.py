"""
simple_mlp.py
-------------
reduce 없이 gemm + elementwise만으로 구성된 간단한 MLP.
지원 연산만 사용: linalg.matmul, arith.addf, sigmoid

구조:
  x [1x64] → Linear(64→128) → Sigmoid → Linear(128→32) → output [1x32]
"""
import torch


class SimpleMLP(torch.nn.Module):
    def __init__(self, in_dim=64, hidden=128, out_dim=32):
        super().__init__()
        # bias=False — bias 있으면 addf가 별도 블록으로 나옴 (지원됨)
        self.fc1 = torch.nn.Linear(in_dim, hidden, bias=False)
        self.fc2 = torch.nn.Linear(hidden, out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)          # linalg.matmul
        x = torch.sigmoid(x)     # sigmoid
        x = self.fc2(x)          # linalg.matmul
        return x


def get_model():
    return SimpleMLP()


def get_sample_inputs():
    return (torch.randn(1, 64),)
