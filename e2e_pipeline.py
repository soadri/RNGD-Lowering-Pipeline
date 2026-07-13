"""
RNGD lowering 파일럿 — v6: Family A(elementwise) + Family B(contraction) 통합 자동화

deploy_and_test.py는 변경 없이 그대로 재사용한다 (e2e_{op}_kernel.rs 등
파일명 규칙만 지키면 어떤 연산이든 배치+테스트가 되도록 이미 설계되어 있음).
"""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

import torch
import torch_mlir
from torch_mlir.ir import (
    Context, Module, Operation, InsertionPoint, StringAttr, RankedTensorType, FloatAttr,
)


def _list_op_names(op) -> list:
    """op을 루트로 모든 하위 op 이름을 재귀 수집 (builtin.module 자신은 제외)."""
    names = []

    def walk(o, is_root=False):
        if not is_root:
            names.append(str(o.name))
        for region in o.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner)

    walk(op, is_root=True)
    return names


def _removed_added(ops_before: list, ops_after: list):
    """before/after 사이에서 사라진 op 이름 집합과 새로 생긴 op 이름 집합을 구한다."""
    before_c, after_c = Counter(ops_before), Counter(ops_after)
    removed = before_c - after_c
    added = after_c - before_c
    return set(removed.keys()), set(added.keys())


def _diff_summary(ops_before: list, ops_after: list) -> str:
    """before/after op 목록을 비교해 '무엇이 사라지고 무엇이 생겼는지' 요약 텍스트 생성."""
    before_c, after_c = Counter(ops_before), Counter(ops_after)
    removed = before_c - after_c
    added = after_c - before_c

    def fmt(counter):
        if not counter:
            return "(없음)"
        return ", ".join(f"{name}×{cnt}" if cnt > 1 else name for name, cnt in counter.items())

    return (
        f"op 개수: {len(ops_before)}개 → {len(ops_after)}개\n"
        f"제거된 op: {fmt(removed)}\n"
        f"추가된 op: {fmt(added)}"
    )


HL_MARK = "@@HL@@"
HL_SEP = "@@SEP@@"


def _mark_ir_annotated(ir_text: str, annotations: list) -> str:
    """annotations: (matcher_substring, label) 목록. 각 줄에서 매칭되는 라벨을 찾아
    HL_MARK<라벨>HL_SEP<원본줄> 형태로 감싼다. 프론트엔드가 이 마커를 보고
    하이라이트 + 라벨 텍스트를 함께 표시한다."""
    if not annotations:
        return ir_text
    lines = ir_text.splitlines()
    out = []
    for line in lines:
        labels = [label for matcher, label in annotations if matcher and matcher in line]
        labels = list(dict.fromkeys(labels))  # 중복 제거, 순서 유지
        if labels:
            out.append(f"{HL_MARK}{' / '.join(labels)}{HL_SEP}{line}")
        else:
            out.append(line)
    return "\n".join(out)


# =====================================================================
# Family A: elementwise (linalg.generic 기반)
# =====================================================================

ARITH_BINARY_OPS = {
    "arith.addf": "add",
    "arith.subf": "sub",
    "arith.mulf": "mul",
    "arith.divf": "div",
}

RNGD_OP_TO_FP_VARIANT = {
    "add": ("FpBinaryOp::AddF", True),
    "sub": ("FpBinaryOp::SubF", True),
    "mul": ("FpBinaryOp::MulF(FpMulAlu::Mul0)", True),
    "div": ("FpBinaryOp::DivF", True),  # 빌드+테스트 통과로 confirmed 승격
}

# 단항(unary) elementwise 연산. math.rsqrt는 하드웨어에 Rsqrt가 따로 없어서
# Sqrt(FpUnaryOp) 다음 1.0/x(vector_fp_div 스칼라 상수)로 조합해서 구현한다.
MATH_UNARY_OPS = {
    "math.rsqrt": "rsqrt",
    "math.sqrt":  "sqrt",
    "math.exp":   "exp",
}

# =====================================================================
# Family B: contraction (named op 기반)
# =====================================================================

NAMED_CONTRACTION_OPS = {
    "linalg.batch_matmul": "batch_gemm",   # confirmed: torch.bmm 실측
    "linalg.matmul": "gemm",                # confirmed: torch.matmul(2D) 실측
    "linalg.dot": "dot_product",            # confirmed: torch.matmul(1D,1D) 실측 (torch.dot 자체는 미지원)
}


@dataclass
class RngdSpec:
    family: Literal["elementwise", "contraction"]
    rngd_op: str
    shape: list = field(default_factory=list)      # elementwise: 출력 shape
    batch: Optional[int] = None                     # contraction: V
    m: Optional[int] = None
    k: Optional[int] = None
    n: Optional[int] = None


def rewrite_to_rngd(module: Module):
    """Family A/B 통합 rewrite. 첫 매치만 처리.
    반환: (RngdSpec | None, annotations_before, annotations_after)
    annotations_* 는 (matcher_substring, label) 튜플 목록 — IR 텍스트에서 그 문자열이
    포함된 줄에 label을 붙여 하이라이트하기 위한 용도."""
    result: dict = {"spec": None, "ann_before": [], "ann_after": []}

    def check_elementwise(op, preceding) -> bool:
        if str(op.name) != "linalg.generic":
            return False
        try:
            iter_types = op.attributes["iterator_types"]
            all_parallel = all("parallel" in str(t) for t in iter_types)
        except KeyError:
            all_parallel = False
        if not all_parallel:
            return False

        num_ins = len(op.operands) - len(op.results)
        body_block = op.regions[0].blocks[0]
        body_op_names = [str(inner.name) for inner in body_block.operations]
        binary_ops_present = [n for n in body_op_names if n in ARITH_BINARY_OPS]
        unary_ops_present = [n for n in body_op_names if n in MATH_UNARY_OPS]
        powf_ops_present = [n for n in body_op_names if n == "math.powf"]
        other_ops = [
            n for n in body_op_names
            if n.startswith(("arith.", "math."))
            and n not in ARITH_BINARY_OPS and n not in MATH_UNARY_OPS and n != "math.powf"
        ]
        if other_ops:
            return False

        # --- 이항(binary) elementwise: add/sub/mul/div ---
        if num_ins == 2 and len(binary_ops_present) == 1 and not unary_ops_present:
            arith_op_name = binary_ops_present[0]
            rngd_op_name = ARITH_BINARY_OPS[arith_op_name]
            lhs, rhs = op.operands[0], op.operands[1]
            result_type = op.results[0].type
            shape = list(RankedTensorType(result_type).shape)

            result["ann_before"] = [
                (arith_op_name, f'연산 종류 결정 → rngd.elementwise attribute op="{rngd_op_name}"'),
                ("linalg.generic", "이 블록 전체가 rngd.elementwise 하나로 대체됨"),
                ("tensor.empty", "출력 버퍼 초기화 — 대체 후 불필요해져 제거됨"),
                ("linalg.yield", "generic 내부 결과 반환 — 대체 후 제거됨"),
            ]

            with InsertionPoint(op):
                new_op = Operation.create(
                    "rngd.elementwise",
                    results=[result_type],
                    operands=[lhs, rhs],
                    attributes={"op": StringAttr.get(rngd_op_name)},
                    loc=op.location,
                )
            op.results[0].replace_all_uses_with(new_op.results[0])
            op.erase()

            result["ann_after"] = [
                ("rngd.elementwise", f'실제 연산자 — attribute op="{rngd_op_name}"에 연산 종류가 담김'),
            ]
            result["spec"] = RngdSpec(family="elementwise", rngd_op=rngd_op_name, shape=shape)
            return True

        # --- 단항(unary) elementwise: rsqrt 등 ---
        if num_ins == 1 and len(unary_ops_present) == 1 and not binary_ops_present:
            math_op_name = unary_ops_present[0]
            rngd_op_name = MATH_UNARY_OPS[math_op_name]
            lhs = op.operands[0]
            result_type = op.results[0].type
            shape = list(RankedTensorType(result_type).shape)

            result["ann_before"] = [
                (math_op_name, f'연산 종류 결정 → rngd.elementwise attribute op="{rngd_op_name}"'),
                ("linalg.generic", "이 블록 전체가 rngd.elementwise 하나로 대체됨"),
                ("tensor.empty", "출력 버퍼 초기화 — 대체 후 불필요해져 제거됨"),
                ("linalg.yield", "generic 내부 결과 반환 — 대체 후 제거됨"),
            ]

            with InsertionPoint(op):
                new_op = Operation.create(
                    "rngd.elementwise",
                    results=[result_type],
                    operands=[lhs],
                    attributes={"op": StringAttr.get(rngd_op_name)},
                    loc=op.location,
                )
            op.results[0].replace_all_uses_with(new_op.results[0])
            op.erase()

            result["ann_after"] = [
                ("rngd.elementwise", f'실제 연산자 — attribute op="{rngd_op_name}"에 연산 종류가 담김 (하드웨어엔 Sqrt만 있어 1.0/sqrt(x)로 조합)'),
            ]
            result["spec"] = RngdSpec(family="elementwise", rngd_op=rngd_op_name, shape=shape)
            return True

        # --- 단항(unary) elementwise: pow(x, 2.0) -> pow2 (지수가 정확히 2.0인지 실제 값 검증) ---
        if num_ins == 1 and len(powf_ops_present) == 1 and not binary_ops_present and not unary_ops_present:
            powf_op = next(o for o in body_block.operations if str(o.name) == "math.powf")
            exponent_operand = powf_op.operands[1]
            exponent_def = exponent_operand.owner
            exp_val = None
            if exponent_def is not None and hasattr(exponent_def, "attributes"):
                try:
                    exp_val = FloatAttr(exponent_def.attributes["value"]).value
                except (KeyError, ValueError, TypeError):
                    exp_val = None

            if exp_val != 2.0:
                return False

            rngd_op_name = "pow2"
            lhs = op.operands[0]
            result_type = op.results[0].type
            shape = list(RankedTensorType(result_type).shape)

            result["ann_before"] = [
                ("math.powf", f'연산 종류 결정 (지수={exp_val}) -> rngd.elementwise attribute op="{rngd_op_name}"'),
                ("linalg.generic", "이 블록 전체가 rngd.elementwise 하나로 대체됨"),
                ("tensor.empty", "출력 버퍼 초기화 — 대체 후 불필요해져 제거됨"),
                ("linalg.yield", "generic 내부 결과 반환 — 대체 후 제거됨"),
            ]

            with InsertionPoint(op):
                new_op = Operation.create(
                    "rngd.elementwise",
                    results=[result_type],
                    operands=[lhs],
                    attributes={"op": StringAttr.get(rngd_op_name)},
                    loc=op.location,
                )
            op.results[0].replace_all_uses_with(new_op.results[0])
            op.erase()

            result["ann_after"] = [
                ("rngd.elementwise", f'실제 연산자 — attribute op="{rngd_op_name}"에 연산 종류가 담김 (하드웨어엔 Pow가 없어 MulF(x,x)로 조합)'),
            ]
            result["spec"] = RngdSpec(family="elementwise", rngd_op=rngd_op_name, shape=shape)
            return True

        return False

    def check_contraction(op) -> bool:
        if str(op.name) not in NAMED_CONTRACTION_OPS:
            return False
        rngd_op_name = NAMED_CONTRACTION_OPS[str(op.name)]
        original_op_name = str(op.name)

        lhs, rhs, outs_init = op.operands[0], op.operands[1], op.operands[2]
        result_type = op.results[0].type
        shape = list(RankedTensorType(result_type).shape)       # [V,M,N] 또는 [M,N]
        lhs_shape = list(RankedTensorType(lhs.type).shape)        # [V,M,K] 또는 [M,K]

        result["ann_before"] = [
            (original_op_name, f"이 연산 전체가 rngd.{rngd_op_name} 하나로 대체됨"),
            ("linalg.fill", "출력 0-초기화 — 대체 후 흡수되어 사라짐 (DCE로 제거)"),
        ]

        with InsertionPoint(op):
            new_op = Operation.create(
                f"rngd.{rngd_op_name}",
                results=[result_type],
                operands=[lhs, rhs],
                attributes={},
                loc=op.location,
            )
        op.results[0].replace_all_uses_with(new_op.results[0])
        op.erase()

        # DCE: linalg.fill(+tensor.empty, arith.constant) 정리
        fill_op = outs_init.owner
        if fill_op is not None and str(fill_op.name) == "linalg.fill":
            if len(list(fill_op.results[0].uses)) == 0:
                fill_operands = list(fill_op.operands)
                fill_op.erase()
                for operand in fill_operands:
                    producer = operand.owner
                    if producer is not None and hasattr(producer, "results"):
                        if len(list(producer.results[0].uses)) == 0:
                            producer.erase()

        if len(shape) == 3:
            batch, m, n = shape
            k = lhs_shape[2]
        elif len(shape) == 2:
            # 배치 축이 없는 2D matmul(gemm) -> batch=None (codegen에서 V=1로 취급)
            batch = None
            m, n = shape
            k = lhs_shape[1]
        else:
            # 스칼라 출력(dot_product) -> M=1, N=1로 취급, K는 입력 벡터 길이
            # (V=1,M=1,N=1인 batch_gemm과 평평한 버퍼 크기가 정확히 일치)
            batch = None
            m, n = 1, 1
            k = lhs_shape[0]

        result["ann_after"] = [
            (f"rngd.{rngd_op_name}", "Contraction Engine(TRF + contract_outer/packet/time/lane)이 실제 계산 수행"),
        ]
        result["spec"] = RngdSpec(family="contraction", rngd_op=rngd_op_name, batch=batch, m=m, n=n, k=k)
        return True

    def walk_block(block, preceding=None) -> bool:
        preceding = preceding or []
        for op in list(block.operations):
            if check_elementwise(op, preceding) or check_contraction(op):
                return True
            for region in op.regions:
                for inner_block in region.blocks:
                    if walk_block(inner_block):
                        return True
        return False

    walk_block(module.operation.regions[0].blocks[0])
    return result["spec"], result["ann_before"], result["ann_after"]


# =====================================================================
# 코드 생성 — Family A
# =====================================================================

def _gen_kernel_elementwise(rngd_op: str, axis_size: int):
    fp_variant, confirmed = RNGD_OP_TO_FP_VARIANT[rngd_op]
    status = "confirmed" if confirmed else "EXPERIMENTAL"

    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.elementwise(op="{rngd_op}") — {status}
axes![A = {axis_size}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn pilot_e2e_{rngd_op}_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<f32, Chip, m![A]>,
    rhs: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {{
    let lhs_dm = lhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);
    let rhs_dm = rhs.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 1 << 12);

    let rhs_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![A % 8]> = ctx
        .sub
        .begin(rhs_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![A % 8 / 8], m![A % 8 % 8]>()
        .to_vrf(0);

    let result = ctx
        .main
        .begin(lhs_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary({fp_variant}, &rhs_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>(1 << 13);

    result.to_hbm(&mut ctx.tdma, 1 << 28)
}}
"""

    host_and_test_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_{rngd_op}_kernel::{{A, pilot_e2e_{rngd_op}_kernel}};

mod reference_data_e2e_{rngd_op};
use reference_data_e2e_{rngd_op}::{{CHECK_N, reference_a, reference_b, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let lhs = HostTensor::<f32, m![A]>::from_buf(reference_a());
    let rhs = HostTensor::<f32, m![A]>::from_buf(reference_b());
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_e2e_{rngd_op}_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Pilot E2E {rngd_op}: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {{
        let mut ctx = Context::acquire();
        let lhs = HostTensor::<f32, m![A]>::from_buf(reference_a());
        let rhs = HostTensor::<f32, m![A]>::from_buf(reference_b());
        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

        let out_hbm = launch(pilot_e2e_{rngd_op}_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {{
            println!("  [{{i}}]: {{}} | {{}}", expected[i], actual[i]);
        }}
        for i in 0..CHECK_N {{
            assert!(
                (expected[i] - actual[i]).abs() < 1e-4,
                "mismatch at i={{i}}: pytorch={{}}  rngd_sim={{}}", expected[i], actual[i]
            );
        }}
    }}
}}
"""
    return kernel_rs, host_and_test_rs


def _gen_reference_elementwise(a, b, expected, axis_size):
    check_n = len(a)
    a_full = a + [0.0] * (axis_size - check_n)
    b_full = b + [0.0] * (axis_size - check_n)

    def lit(arr):
        return "vec![" + ", ".join(f"{float(v):.8}_f32" for v in arr) + "]"

    return (
        "// AUTO-GENERATED — expected는 실제 PyTorch forward() 출력값\n"
        f"pub const CHECK_N: usize = {check_n};\n"
        f"pub fn reference_a() -> Vec<f32> {{ {lit(a_full)} }}\n"
        f"pub fn reference_b() -> Vec<f32> {{ {lit(b_full)} }}\n"
        f"pub fn reference_expected() -> Vec<f32> {{ {lit(expected)} }}\n"
    )


# =====================================================================
# 코드 생성 — Family A 단항(unary) elementwise (rsqrt 등)
# =====================================================================

# 하드웨어에 직접 있는 FpUnaryOp 하나로 안 끝나는 경우, 뒤에 이어붙일 체인.
# rsqrt = 1.0 / sqrt(x): Sqrt(단항) 다음 스칼라 상수 1.0으로 나누기.
UNARY_OP_CHAIN = {
    "rsqrt": ".vector_fp_unary(FpUnaryOp::Sqrt)\n        .vector_fp_div_with_mode(BinaryArgMode::Mode10, 1.0_f32)",
    "sqrt":  ".vector_fp_unary(FpUnaryOp::Sqrt)",
    "exp":   ".vector_fp_unary(FpUnaryOp::Exp)",
}


def _gen_kernel_elementwise_unary(rngd_op: str, axis_size: int):
    chain = UNARY_OP_CHAIN[rngd_op]

    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.elementwise(op="{rngd_op}") — 단항(unary), EXPERIMENTAL
// 하드웨어엔 Rsqrt가 직접 없어서 Sqrt(FpUnaryOp) 다음 1.0/x(vector_fp_div 스칼라)로 조합.
axes![A = {axis_size}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn pilot_e2e_{rngd_op}_kernel(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {{
    let input_dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);

    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        {chain}
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>(1 << 13);

    result.to_hbm(&mut ctx.tdma, 1 << 28)
}}
"""

    host_and_test_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_{rngd_op}_kernel::{{A, pilot_e2e_{rngd_op}_kernel}};

mod reference_data_e2e_{rngd_op};
use reference_data_e2e_{rngd_op}::{{CHECK_N, reference_a, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let input = HostTensor::<f32, m![A]>::from_buf(reference_a());
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;
    let _out_hbm = launch(pilot_e2e_{rngd_op}_kernel, (&mut ctx, &input_hbm)).await;
    println!("Pilot E2E {rngd_op}: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {{
        let mut ctx = Context::acquire();
        let input = HostTensor::<f32, m![A]>::from_buf(reference_a());
        let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;

        let out_hbm = launch(pilot_e2e_{rngd_op}_kernel, (&mut ctx, &input_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {{
            println!("  [{{i}}]: {{}} | {{}}", expected[i], actual[i]);
        }}
        for i in 0..CHECK_N {{
            let diff = (expected[i] - actual[i]).abs();
            let tol = (0.02 * expected[i].abs()).max(1e-3);
            assert!(
                diff <= tol,
                "mismatch at i={{i}}: pytorch={{}}  rngd_sim={{}}  diff={{}} > tol={{}}", expected[i], actual[i], diff, tol
            );
        }}
    }}
}}
"""
    return kernel_rs, host_and_test_rs


def _gen_reference_elementwise_unary(a, expected, axis_size):
    check_n = len(a)
    a_full = a + [1.0] * (axis_size - check_n)   # rsqrt(0)=inf 방지용으로 0 대신 1.0 패딩
    expected_full = expected + [1.0] * (axis_size - check_n)

    def lit(arr):
        return "vec![" + ", ".join(f"{float(v):.8}_f32" for v in arr) + "]"

    return (
        "// AUTO-GENERATED — expected는 실제 PyTorch forward() 출력값\n"
        f"pub const CHECK_N: usize = {check_n};\n"
        f"pub fn reference_a() -> Vec<f32> {{ {lit(a_full)} }}\n"
        f"pub fn reference_expected() -> Vec<f32> {{ {lit(expected_full)} }}\n"
    )


def _gen_kernel_elementwise_square(axis_size: int):
    """x^2 = x*x. 하드웨어에 Pow가 없어서, 같은 입력을 두 번(메인 스트림 + VRF로)
    적재해 MulF(x,x)로 구현한다 — 이항 elementwise 커널과 거의 동일한 구조."""
    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.elementwise(op="pow2") — 단항(unary), EXPERIMENTAL
// 하드웨어엔 Pow가 없어서 같은 입력을 두 번 적재해 MulF(x,x)로 조합.
axes![A = {axis_size}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![A / 8 # 256];

#[device(chip = 1)]
pub fn pilot_e2e_pow2_kernel(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A]>,
) -> HbmTensor<f32, Chip, m![A]> {{
    let input_dm = input.to_dm::<Cluster, Slice, m![A % 8]>(&mut ctx.tdma, 0);

    // 같은 입력을 VRF로도 적재 — 자기 자신과 곱하기 위함
    let self_vrf: VrfTensor<f32, Chip, Cluster, Slice, m![A % 8]> = ctx
        .sub
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![A % 8 / 8], m![A % 8 % 8]>()
        .to_vrf(0);

    let result = ctx
        .main
        .begin(input_dm.view())
        .fetch::<m![1], m![A % 8]>()
        .collect::<m![1], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![1, A / 4 % 2], m![A % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &self_vrf)
        .vector_widen_concat::<m![1], m![A % 8]>()
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit::<m![A % 8]>(1 << 13);

    result.to_hbm(&mut ctx.tdma, 1 << 28)
}}
"""

    host_and_test_rs = """\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_pow2_kernel::{A, pilot_e2e_pow2_kernel};

mod reference_data_e2e_pow2;
use reference_data_e2e_pow2::{CHECK_N, reference_a, reference_expected};

#[tokio::main]
async fn main() {
    let mut ctx = Context::acquire();
    let input = HostTensor::<f32, m![A]>::from_buf(reference_a());
    let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;
    let _out_hbm = launch(pilot_e2e_pow2_kernel, (&mut ctx, &input_hbm)).await;
    println!("Pilot E2E pow2: kernel ran");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn matches_actual_pytorch_output() {
        let mut ctx = Context::acquire();
        let input = HostTensor::<f32, m![A]>::from_buf(reference_a());
        let input_hbm = input.to_hbm(&mut ctx.pdma, 0).await;

        let out_hbm = launch(pilot_e2e_pow2_kernel, (&mut ctx, &input_hbm)).await;
        let actual: Vec<f32> = out_hbm.to_host::<m![A]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== PyTorch 실제 출력 vs RNGD 시뮬레이터 출력 ===");
        for i in 0..CHECK_N {
            println!("  [{}]: {} | {}", i, expected[i], actual[i]);
        }
        for i in 0..CHECK_N {
            let diff = (expected[i] - actual[i]).abs();
            let tol = (0.02 * expected[i].abs()).max(1e-3);
            assert!(
                diff <= tol,
                "mismatch at i={}: pytorch={}  rngd_sim={}  diff={} > tol={}", i, expected[i], actual[i], diff, tol
            );
        }
    }
}
"""
    return kernel_rs, host_and_test_rs


# =====================================================================
# 코드 생성 — Family B (batch_gemm)
# =====================================================================

def _gen_kernel_batch_gemm(v: int, m: int, k: int, n: int, op_name: str = "batch_gemm"):
    assert v <= 256, f"V={v} > 256: 이 템플릿은 V<=256 가정(Slice=m![V # 256])에서만 검증됨"

    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.{op_name} — confirmed (pilot_batch_gemm 구조 재사용)
// 배치 축이 없는 2D matmul(gemm)은 V=1인 batch_gemm으로 취급 (동일 검증 구조 재사용)
axes![V = {v}, M = {m}, K = {k}, N = {n}];

pub type Chip = m![1];
pub type Cluster = m![1];
// V<=256 가정: 하드웨어 제약(Slice ∈ {{64,128,192,256}})에 맞춰 256-slice 공간에 매핑
pub type Slice = m![V # 256];
// contract_lane의 OutPacket::SIZE는 정확히 8이어야 함(실측 확인) -> N<8이면 8-lane 공간에 패딩
pub type Lane = m![N # 8];

#[device(chip = 1)]
pub fn pilot_e2e_{op_name}_kernel(
    ctx: &mut Context,
    a: &HbmTensor<bf16, Chip, m![V, M, K]>,
    b: &HbmTensor<bf16, Chip, m![V, K, N]>,
) -> HbmTensor<bf16, Chip, m![V, M, N]> {{
    let a: DmTensor<bf16, Chip, Cluster, Slice, m![M, K]> = a.to_dm(&mut ctx.tdma, 0);
    let b: DmTensor<bf16, Chip, Cluster, Slice, m![K, N]> = b.to_dm(&mut ctx.tdma, 1 << 12);

    let b_trf: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![K]> = ctx
        .sub
        .begin(b.view())
        .fetch::<m![N], m![K]>()
        .collect::<m![N, K / 16], m![K % 16]>()
        .to_trf(TrfAddress::Full);

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![M, N]> = ctx
        .main
        .begin(a.view())
        .fetch::<m![M], m![K]>()
        .collect::<m![M, K / 16], m![K % 16]>()
        .contract_outer::<m![M], m![K], _, _>(&b_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![M]>()
        .contract_lane::<m![M], m![N # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![N # 16]>()
        .commit_trim::<m![N]>()
        .commit(0);

    result.to_hbm(&mut ctx.tdma, 2 << 28)
}}
"""

    host_and_test_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_{op_name}_kernel::{{V, M, K, N, pilot_e2e_{op_name}_kernel}};

mod reference_data_e2e_{op_name};
use reference_data_e2e_{op_name}::{{reference_a, reference_b, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let a = HostTensor::<bf16, m![V, M, K]>::from_buf(reference_a());
    let b = HostTensor::<bf16, m![V, K, N]>::from_buf(reference_b());
    let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_e2e_{op_name}_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
    println!("Pilot E2E {op_name}: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    /// expected는 실제 PyTorch 모델의 입력을 bf16으로 양자화한 뒤 그 값으로
    /// (f32 누산 + bf16 반올림) 계산한 값이다 — RNGD가 실제로 낼 수 있는
    /// "현실적인 정답"이며, Rust가 자체 재계산한 값이 아니다.
    #[tokio::test]
    async fn matches_bf16_reference() {{
        let mut ctx = Context::acquire();
        let a = HostTensor::<bf16, m![V, M, K]>::from_buf(reference_a());
        let b = HostTensor::<bf16, m![V, K, N]>::from_buf(reference_b());
        let a_hbm = a.to_hbm(&mut ctx.pdma, 0 << 28).await;
        let b_hbm = b.to_hbm(&mut ctx.pdma, 1 << 28).await;

        let out_hbm = launch(pilot_e2e_{op_name}_kernel, (&mut ctx, &a_hbm, &b_hbm)).await;
        let actual: Vec<bf16> = out_hbm.to_host::<m![V, M, N]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        println!("=== 값 비교 (앞 8개) ===");
        for i in 0..8.min(actual.len()) {{
            println!("  [{{i}}]: {{:?}} | {{:?}}", expected[i], actual[i]);
        }}

        for (idx, (&e, &av)) in expected.iter().zip(&actual).enumerate() {{
            let diff = (f32::from(av) - f32::from(e)).abs();
            let tol = (0.05 * f32::from(e).abs()).max(1.0);
            assert!(diff <= tol, "{op_name} mismatch at idx={{idx}}: expected {{e:?}}, actual {{av:?}}");
        }}
    }}
}}
"""
    return kernel_rs, host_and_test_rs


def _gen_reference_batch_gemm(a_bf16, b_bf16, expected_bf16):
    def build_const(arr, const_name):
        # 함수 호출(bf16::from_f32)을 원소마다 반복하면 AST 노드가 수만 개가 되어
        # 컴파일이 느려짐. 대신 순수 숫자 리터럴 배열(파싱만 하면 됨) + 단일
        # .map(bf16::from_f32) 호출로 바꿔 호출 표현식은 소스에 1번만 나오게 한다.
        values_str = ", ".join(f"{float(v):.8}_f32" for v in arr)
        return f"const {const_name}: [f32; {len(arr)}] = [{values_str}];\n"

    a_const = build_const(a_bf16, "A_F32_VALUES")
    b_const = build_const(b_bf16, "B_F32_VALUES")
    e_const = build_const(expected_bf16, "EXPECTED_F32_VALUES")

    return (
        "// AUTO-GENERATED — expected는 실제 PyTorch 입력을 bf16 양자화 후 계산한 값\n"
        "use furiosa_opt_std::prelude::*;\n\n"
        f"{a_const}"
        f"{b_const}"
        f"{e_const}"
        "pub fn reference_a() -> Vec<bf16> { A_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
        "pub fn reference_b() -> Vec<bf16> { B_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
        "pub fn reference_expected() -> Vec<bf16> { EXPECTED_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
    )


# =====================================================================
# 코드 생성 — dot_product (batch_gemm과 별도 구조 — 검증된 dot_product_kernel.rs 그대로 반영)
# =====================================================================

def _gen_kernel_dot_product(axis_size: int):
    kernel_rs = f"""\
use furiosa_opt_std::prelude::*;

// AUTO-GENERATED from rngd.dot_product — 검증된 dot_product_kernel.rs 구조 그대로 사용.
// batch_gemm 템플릿의 degenerate(M=1,N=1) 축소판이 하드웨어 제약(Time::SIZE가
// Lane::SIZE(8)로 나눠지지 않음)에 걸려 실패했기 때문에, 별도의 벡터 전체-reduction
// 전용 구조를 사용한다.
axes![A = {axis_size}];

pub type Chip = m![1];
pub type Cluster = m![1 # 2];
pub type Slice = m![1 # 256];
pub type Time = m![1];
pub type Lane = m![1];

#[device(chip = 1)]
pub fn pilot_e2e_dot_product_kernel(
    ctx: &mut Context,
    lhs: &HbmTensor<bf16, Chip, m![A]>,
    rhs: &HbmTensor<bf16, Chip, m![A]>,
) -> HbmTensor<bf16, Chip, m![1]> {{
    let lhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = lhs.to_dm(&mut ctx.tdma, 0);
    let rhs: DmTensor<bf16, Chip, Cluster, Slice, m![A]> = rhs.to_dm(&mut ctx.tdma, 1 << 12);

    let rhs: TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![A]> = ctx
        .sub
        .begin(rhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![{{ Time }}, A / 16], m![A % 16]>()
        .to_trf(TrfAddress::Full);

    let result: DmTensor<bf16, Chip, Cluster, Slice, m![1 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<Time, m![A]>()
        .collect::<m![A / 16], m![A % 16]>()
        .contract_outer::<m![A / 32], m![A % 32], _, _>(&rhs)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![1 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![1 # 16]>()
        .commit_trim::<m![1 # 8]>()
        .commit(1 << 13);

    result.to_hbm(&mut ctx.tdma, 2 << 28)
}}
"""

    host_and_test_rs = f"""\
use furiosa_opt_std::prelude::*;
use rngd_tcp_kernel_dev::kernel::pilot_e2e_dot_product_kernel::{{A, pilot_e2e_dot_product_kernel}};

mod reference_data_e2e_dot_product;
use reference_data_e2e_dot_product::{{reference_a, reference_b, reference_expected}};

#[tokio::main]
async fn main() {{
    let mut ctx = Context::acquire();
    let lhs = HostTensor::<bf16, m![A]>::from_buf(reference_a());
    let rhs = HostTensor::<bf16, m![A]>::from_buf(reference_b());
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;
    let _out_hbm = launch(pilot_e2e_dot_product_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
    println!("Pilot E2E dot_product: kernel ran");
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[tokio::test]
    async fn matches_bf16_reference() {{
        let mut ctx = Context::acquire();
        let lhs = HostTensor::<bf16, m![A]>::from_buf(reference_a());
        let rhs = HostTensor::<bf16, m![A]>::from_buf(reference_b());
        let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0).await;
        let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

        let out_hbm = launch(pilot_e2e_dot_product_kernel, (&mut ctx, &lhs_hbm, &rhs_hbm)).await;
        let actual_buf: Vec<bf16> = out_hbm.to_host::<m![1]>(&mut ctx.pdma).await.to_buf();
        let expected = reference_expected();

        if let Some(&actual) = actual_buf.first() {{
            println!("=== 값 비교 ===");
            println!("  expected={{:?}} actual={{:?}}", expected, actual);
            let diff = (f32::from(actual) - f32::from(expected)).abs();
            let tol = (0.05 * f32::from(expected).abs()).max(1.0);
            assert!(diff <= tol, "dot_product mismatch: expected {{expected:?}}, actual {{actual:?}}, diff {{diff}} > tol {{tol}}");
        }}
    }}
}}
"""
    return kernel_rs, host_and_test_rs


def _gen_reference_dot_product(a_bf16, b_bf16, expected_bf16, axis_size):
    a_full = a_bf16 + [0.0] * (axis_size - len(a_bf16))
    b_full = b_bf16 + [0.0] * (axis_size - len(b_bf16))

    def build_const(arr, const_name):
        values_str = ", ".join(f"{float(v):.8}_f32" for v in arr)
        return f"const {const_name}: [f32; {len(arr)}] = [{values_str}];\n"

    a_const = build_const(a_full, "A_F32_VALUES")
    b_const = build_const(b_full, "B_F32_VALUES")

    return (
        "// AUTO-GENERATED — expected는 실제 PyTorch 입력(bf16 양자화)의 dot product\n"
        "// 나머지 원소는 0으로 패딩 (reduction sum이라 0-패딩은 결과에 영향 없음)\n"
        "use furiosa_opt_std::prelude::*;\n\n"
        f"{a_const}"
        f"{b_const}"
        "pub fn reference_a() -> Vec<bf16> { A_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
        "pub fn reference_b() -> Vec<bf16> { B_F32_VALUES.iter().map(|&v| bf16::from_f32(v)).collect() }\n"
        f"pub fn reference_expected() -> bf16 {{ bf16::from_f32({float(expected_bf16):.8}_f32) }}\n"
    )


# =====================================================================
# 실행
# =====================================================================

if __name__ == "__main__":
    import sys

    AXIS_SIZE = 2048
    OUT_DIR = Path("generated")
    OUT_DIR.mkdir(exist_ok=True)

    # 인자 없이 실행하면 5개 전부, 인자를 주면 그 연산만 생성한다.
    # 예: python e2e_pipeline.py add batch_gemm
    requested_ops = set(sys.argv[1:]) if len(sys.argv) > 1 else None

    class ElementwiseBinary(torch.nn.Module):
        def __init__(self, op_name):
            super().__init__()
            self.op_name = op_name

        def forward(self, x, y):
            if self.op_name == "add":
                return x + y
            if self.op_name == "sub":
                return x - y
            if self.op_name == "mul":
                return x * y
            if self.op_name == "div":
                return x / y

    class UnaryElementwise(torch.nn.Module):
        def __init__(self, op_name):
            super().__init__()
            self.op_name = op_name

        def forward(self, x):
            if self.op_name == "rsqrt":
                return torch.rsqrt(x)
            if self.op_name == "sqrt":
                return torch.sqrt(x)
            if self.op_name == "exp":
                return torch.exp(x)
            if self.op_name == "pow2":
                return torch.pow(x, 2.0)

    class BatchMatmul(torch.nn.Module):
        def forward(self, x, y):
            return torch.bmm(x, y)

    class Matmul2D(torch.nn.Module):
        def forward(self, x, y):
            return torch.matmul(x, y)

    class DotProduct(torch.nn.Module):
        # torch.dot(aten.dot)은 이 torch_mlir 버전에서 미지원 확인됨.
        # torch.matmul(1D, 1D)은 내적과 동일 시맨틱이며 linalg.dot으로 lowering됨 (실측 확인).
        def forward(self, x, y):
            return torch.matmul(x, y)

    # --- Family A ---
    elementwise_ops = [op for op in ("add", "sub", "mul", "div") if requested_ops is None or op in requested_ops]
    for op_name in elementwise_ops:
        print(f"\n{'='*70}\n [Family A] op_name = {op_name}\n{'='*70}")

        torch.manual_seed(123)
        model = ElementwiseBinary(op_name)
        x = torch.randn(4, 4)
        if op_name == "div":
            # 분모가 0 근처면 값이 폭발해 오차 비교(1e-4)가 무의미해짐 -> 절댓값 + 오프셋으로 0 회피
            y = torch.randn(4, 4).abs() + 0.5
        else:
            y = torch.randn(4, 4)
        with torch.no_grad():
            actual_output = model(x, y)

        compiled = torch_mlir.compile(model, (x, y), output_type="linalg-on-tensors")
        ir_text_before = str(compiled)
        ctx = Context()
        module = Module.parse(ir_text_before, ctx)
        ops_before = _list_op_names(module.operation)
        spec, ann_before, ann_after = rewrite_to_rngd(module)
        ir_text_after = str(module)
        ops_after = _list_op_names(module.operation)
        print(f"spec: {spec}")
        assert spec is not None and spec.family == "elementwise" and spec.rngd_op == op_name

        kernel_rs, host_test_rs = _gen_kernel_elementwise(op_name, AXIS_SIZE)
        reference_rs = _gen_reference_elementwise(
            x.flatten().tolist(), y.flatten().tolist(), actual_output.flatten().tolist(), AXIS_SIZE
        )

        prefix = f"e2e_{op_name}"
        with open(OUT_DIR / f"{prefix}_kernel.rs", "w") as f:
            f.write(kernel_rs)
        with open(OUT_DIR / f"{prefix}_host_test.rs", "w") as f:
            f.write(host_test_rs)
        with open(OUT_DIR / f"{prefix}_reference_data.rs", "w") as f:
            f.write(reference_rs)
        with open(OUT_DIR / f"{prefix}_ir_before.mlir", "w") as f:
            f.write(ir_text_before)
        with open(OUT_DIR / f"{prefix}_ir_after.mlir", "w") as f:
            f.write(ir_text_after)
        with open(OUT_DIR / f"{prefix}_ir_before_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_before, ann_before))
        with open(OUT_DIR / f"{prefix}_ir_after_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_after, ann_after))
        with open(OUT_DIR / f"{prefix}_ir_diff.txt", "w") as f:
            f.write(_diff_summary(ops_before, ops_after))
        print(f"생성 완료: {prefix}_*.rs, {prefix}_ir_before.mlir, {prefix}_ir_after.mlir, {prefix}_ir_diff.txt")

    # --- Family A (단항/unary) ---
    unary_elementwise_ops = [op for op in ("rsqrt", "sqrt", "exp", "pow2") if requested_ops is None or op in requested_ops]
    for op_name in unary_elementwise_ops:
        print(f"\n{'='*70}\n [Family A-unary] op_name = {op_name}\n{'='*70}")

        torch.manual_seed(123)
        model = UnaryElementwise(op_name)
        if op_name in ("rsqrt", "sqrt"):
            x = torch.rand(4, 4) + 0.5  # rsqrt/sqrt 정의역(양수) 보장
        else:
            x = torch.randn(4, 4)  # pow2는 정의역 제한 없음
        with torch.no_grad():
            actual_output = model(x)

        compiled = torch_mlir.compile(model, (x,), output_type="linalg-on-tensors")
        ir_text_before = str(compiled)
        ctx = Context()
        module = Module.parse(ir_text_before, ctx)
        ops_before = _list_op_names(module.operation)
        spec, ann_before, ann_after = rewrite_to_rngd(module)
        ir_text_after = str(module)
        ops_after = _list_op_names(module.operation)
        print(f"spec: {spec}")
        assert spec is not None and spec.family == "elementwise" and spec.rngd_op == op_name

        if op_name == "pow2":
            kernel_rs, host_test_rs = _gen_kernel_elementwise_square(AXIS_SIZE)
        else:
            kernel_rs, host_test_rs = _gen_kernel_elementwise_unary(op_name, AXIS_SIZE)
        reference_rs = _gen_reference_elementwise_unary(
            x.flatten().tolist(), actual_output.flatten().tolist(), AXIS_SIZE
        )

        prefix = f"e2e_{op_name}"
        with open(OUT_DIR / f"{prefix}_kernel.rs", "w") as f:
            f.write(kernel_rs)
        with open(OUT_DIR / f"{prefix}_host_test.rs", "w") as f:
            f.write(host_test_rs)
        with open(OUT_DIR / f"{prefix}_reference_data.rs", "w") as f:
            f.write(reference_rs)
        with open(OUT_DIR / f"{prefix}_ir_before.mlir", "w") as f:
            f.write(ir_text_before)
        with open(OUT_DIR / f"{prefix}_ir_after.mlir", "w") as f:
            f.write(ir_text_after)
        with open(OUT_DIR / f"{prefix}_ir_before_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_before, ann_before))
        with open(OUT_DIR / f"{prefix}_ir_after_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_after, ann_after))
        with open(OUT_DIR / f"{prefix}_ir_diff.txt", "w") as f:
            f.write(_diff_summary(ops_before, ops_after))
        print(f"생성 완료: {prefix}_*.rs, {prefix}_ir_before.mlir, {prefix}_ir_after.mlir, {prefix}_ir_diff.txt")



    # --- Family B ---
    contraction_cases_all = [
        # (op_name, model, seed, x_shape, y_shape)
        ("batch_gemm", BatchMatmul(), 42, (32, 32, 32), (32, 32, 8)),
        ("gemm", Matmul2D(), 43, (32, 32), (32, 8)),
        ("dot_product", DotProduct(), 44, (32,), (32,)),
    ]
    contraction_cases = [
        c for c in contraction_cases_all if requested_ops is None or c[0] in requested_ops
    ]

    for op_name, model, seed, x_shape, y_shape in contraction_cases:
        print(f"\n{'='*70}\n [Family B] op_name = {op_name}\n{'='*70}")

        torch.manual_seed(seed)
        x = torch.randn(*x_shape)
        y = torch.randn(*y_shape)

        compiled = torch_mlir.compile(model, (x, y), output_type="linalg-on-tensors")
        ir_text_before = str(compiled)
        ctx = Context()
        module = Module.parse(ir_text_before, ctx)
        ops_before = _list_op_names(module.operation)
        spec, ann_before, ann_after = rewrite_to_rngd(module)
        ir_text_after = str(module)
        ops_after = _list_op_names(module.operation)
        print(f"spec: {spec}")
        assert spec is not None and spec.family == "contraction" and spec.rngd_op == op_name

        # 배치 축이 없으면(2D gemm) V=1로 취급 — batch_gemm과 동일 검증 구조 재사용
        v = spec.batch if spec.batch is not None else 1

        x_bf16_f32 = x.to(torch.bfloat16).to(torch.float32)
        y_bf16_f32 = y.to(torch.bfloat16).to(torch.float32)
        with torch.no_grad():
            expected_f32_acc = model(x_bf16_f32, y_bf16_f32)
        expected_bf16 = expected_f32_acc.to(torch.bfloat16)

        kernel_rs, host_test_rs = _gen_kernel_batch_gemm(v, spec.m, spec.k, spec.n, op_name=op_name)
        reference_rs = _gen_reference_batch_gemm(
            x_bf16_f32.flatten().tolist(),
            y_bf16_f32.flatten().tolist(),
            expected_bf16.to(torch.float32).flatten().tolist(),
        )

        if op_name == "dot_product":
            # batch_gemm 템플릿의 degenerate(M=1,N=1) 케이스가 하드웨어 제약에 걸려
            # 실패했으므로, 검증된 dot_product_kernel.rs 구조를 그대로 쓰는 전용 함수로 대체.
            kernel_rs, host_test_rs = _gen_kernel_dot_product(AXIS_SIZE)
            reference_rs = _gen_reference_dot_product(
                x_bf16_f32.flatten().tolist(),
                y_bf16_f32.flatten().tolist(),
                expected_bf16.to(torch.float32).item(),
                AXIS_SIZE,
            )

        prefix = f"e2e_{op_name}"
        with open(OUT_DIR / f"{prefix}_kernel.rs", "w") as f:
            f.write(kernel_rs)
        with open(OUT_DIR / f"{prefix}_host_test.rs", "w") as f:
            f.write(host_test_rs)
        with open(OUT_DIR / f"{prefix}_reference_data.rs", "w") as f:
            f.write(reference_rs)
        with open(OUT_DIR / f"{prefix}_ir_before.mlir", "w") as f:
            f.write(ir_text_before)
        with open(OUT_DIR / f"{prefix}_ir_after.mlir", "w") as f:
            f.write(ir_text_after)
        with open(OUT_DIR / f"{prefix}_ir_before_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_before, ann_before))
        with open(OUT_DIR / f"{prefix}_ir_after_marked.mlir", "w") as f:
            f.write(_mark_ir_annotated(ir_text_after, ann_after))
        with open(OUT_DIR / f"{prefix}_ir_diff.txt", "w") as f:
            f.write(_diff_summary(ops_before, ops_after))
        print(f"생성 완료: {prefix}_*.rs, {prefix}_ir_before.mlir, {prefix}_ir_after.mlir, {prefix}_ir_diff.txt")
