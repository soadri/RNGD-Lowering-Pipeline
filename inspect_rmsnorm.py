"""
Llama-3.1-8B의 실제 RMSNorm 연산이 어떤 linalg op으로 컴파일되는지 확인.
RMSNorm(x) = x / sqrt(mean(x^2, dim=-1) + eps) * weight

실제 Llama-3.1-8B 설정값: hidden_size=4096, rms_norm_eps=1e-5
"""
import torch
import torch_mlir
from torch_mlir.ir import Context, Module

HIDDEN_SIZE = 4096
RMS_NORM_EPS = 1e-5
SEQ_LEN = 1
BATCH = 1


class LlamaRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


model = LlamaRMSNorm(HIDDEN_SIZE, RMS_NORM_EPS)
example_input = torch.randn(BATCH, SEQ_LEN, HIDDEN_SIZE)

try:
    compiled = torch_mlir.compile(model, (example_input,), output_type="linalg-on-tensors")
    ir_text = str(compiled)
    print("=== 컴파일 성공 ===")
    print(ir_text)

    ctx = Context()
    module = Module.parse(ir_text, ctx)

    def walk(op, depth=0):
        indent = "  " * depth
        print(f"{indent}op.name = {op.name}  (operands={len(op.operands)}, results={len(op.results)})")
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner, depth + 1)

    print("\n=== 구조 순회 ===")
    walk(module.operation)

    print("\n=== 등장한 고유 op 이름 목록 ===")
    seen = set()
    def collect_names(op):
        seen.add(str(op.name))
        for region in op.regions:
            for block in region.blocks:
                for inner in block.operations:
                    collect_names(inner)
    collect_names(module.operation)
    for name in sorted(seen):
        print(f"  {name}")

except Exception as e:
    print("=== 컴파일 실패 ===")
    print(type(e).__name__, ":", str(e)[:2000])
