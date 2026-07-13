"""
llama_ffn_real.py
-----------------
Llama-3.1 8B FFN 블록을 실제 가중치(랜덤 초기화)로 실행.
dim=4096, hidden=14336 (실제 Llama-3.1 8B 크기)
"""
import torch

class LlamaFFN(torch.nn.Module):
    def __init__(self, dim=4096, hidden=14336):
        super().__init__()
        self.gate = torch.nn.Linear(dim, hidden, bias=False)
        self.up   = torch.nn.Linear(dim, hidden, bias=False)
        self.down = torch.nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))

def get_model():
    return LlamaFFN()

def get_sample_inputs():
    return (torch.randn(1, 4096),)
