"""
example_linear.py
-----------------
CI 커버리지 분석 예시용 모델.
torch.nn.Linear는 linalg.matmul + linalg.generic(arith.addf)로 컴파일된다.
"""

import torch


def get_model():
    return torch.nn.Linear(64, 32)


def get_sample_inputs():
    return (torch.randn(1, 64),)
