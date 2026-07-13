"""
init_registry.py
================
rngd_registry.db를 처음 생성하고, 지금까지 e2e_pipeline.py / dashboard_app.py에
하드코딩되어 있던 변환 규칙·근거·하드웨어 제약·로드맵을 한 번에 백필한다.

사용법:
    python init_registry.py            # 첫 실행 — DB 생성 + 백필
    python init_registry.py --reset    # DB를 지우고 처음부터 다시 채움

스키마 변경이 생기면 이 파일을 직접 수정하면 된다.
"""

import argparse
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "rngd_registry.db"

# ──────────────────────────────────────────────────────────────────────────────
# 스키마
# ──────────────────────────────────────────────────────────────────────────────

SCHEMA = """
         CREATE TABLE IF NOT EXISTS ops (
                                            op_name         TEXT PRIMARY KEY,   -- 'add', 'rsqrt', 'batch_gemm', ...
                                            family          TEXT NOT NULL,       -- 'elementwise' | 'contraction'
                                            source_op       TEXT NOT NULL,       -- 'linalg.generic(arith.addf)'
                                            target_op       TEXT NOT NULL,       -- 'rngd.elementwise'
                                            dtype           TEXT NOT NULL,       -- 'f32' | 'bf16'
                                            tolerance       TEXT NOT NULL,       -- '1e-4' | 'max(2%, 1e-3)' | ...
                                            experimental    INTEGER NOT NULL DEFAULT 0,  -- 0 or 1 (bool)
                                            status          TEXT NOT NULL DEFAULT 'done',  -- 'done' | 'wip' | 'broken'
                                            compares        TEXT,
                                            meaning_pass    TEXT,
                                            meaning_fail    TEXT,
                                            detail          TEXT
         );

         CREATE TABLE IF NOT EXISTS evidence (
                                                 id              INTEGER PRIMARY KEY AUTOINCREMENT,
                                                 op_name         TEXT NOT NULL REFERENCES ops(op_name) ON DELETE CASCADE,
             evidence_type   TEXT NOT NULL,  -- 'static'(소스 코드 확인) | 'runtime'(시뮬레이터 실행)
             claim           TEXT NOT NULL,
             source          TEXT NOT NULL
             );

         CREATE TABLE IF NOT EXISTS hardware_constraints (
                                                             id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                                                             description         TEXT NOT NULL,   -- 사람이 읽는 제약 설명
                                                             discovery_source    TEXT NOT NULL    -- panic 메시지 원문 + 파일:라인
         );

         CREATE TABLE IF NOT EXISTS op_constraint_link (
                                                           op_name         TEXT NOT NULL REFERENCES ops(op_name) ON DELETE CASCADE,
             constraint_id   INTEGER NOT NULL REFERENCES hardware_constraints(id) ON DELETE CASCADE,
             PRIMARY KEY (op_name, constraint_id)
             );

         CREATE TABLE IF NOT EXISTS roadmap (
                                                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                                                name    TEXT NOT NULL,
                                                status  TEXT NOT NULL,  -- '미착수' | '설계중' | '막힘'
                                                note    TEXT
         ); \
         """

# ──────────────────────────────────────────────────────────────────────────────
# 백필 데이터
# ──────────────────────────────────────────────────────────────────────────────

OPS = [
    {
        "op_name": "add",
        "family": "elementwise",
        "source_op": "linalg.generic(arith.addf)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "1e-4",
        "experimental": 0,
        "status": "done",
        "compares": "PyTorch가 계산한 x + y",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 오차 1e-4 이내로 일치.",
        "meaning_fail": "두 출력이 오차 1e-4를 초과. 변환 규칙 또는 생성된 커널 코드 확인 필요.",
        "detail": "생성된 커널 체인: narrow_split → vector_fp_binary(AddF) → widen_concat",
    },
    {
        "op_name": "sub",
        "family": "elementwise",
        "source_op": "linalg.generic(arith.subf)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "1e-4",
        "experimental": 0,
        "status": "done",
        "compares": "PyTorch가 계산한 x - y",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 오차 1e-4 이내로 일치.",
        "meaning_fail": "두 출력이 오차 1e-4를 초과. 변환 규칙 또는 생성된 커널 코드 확인 필요.",
        "detail": "생성된 커널 체인: narrow_split → vector_fp_binary(SubF) → widen_concat",
    },
    {
        "op_name": "mul",
        "family": "elementwise",
        "source_op": "linalg.generic(arith.mulf)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "1e-4",
        "experimental": 0,
        "status": "done",
        "compares": "PyTorch가 계산한 x * y",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 오차 1e-4 이내로 일치.",
        "meaning_fail": "두 출력이 오차 1e-4를 초과. 변환 규칙 또는 생성된 커널 코드 확인 필요.",
        "detail": "생성된 커널 체인: narrow_split → vector_fp_binary(MulF) → widen_concat",
    },
    {
        "op_name": "div",
        "family": "elementwise",
        "source_op": "linalg.generic(arith.divf)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "1e-4",
        "experimental": 0,
        "status": "done",
        "compares": "PyTorch가 계산한 x / y (분모는 0.5 이상으로 생성, 0 근처 회피)",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 오차 1e-4 이내로 일치.",
        "meaning_fail": "두 출력이 오차 1e-4를 초과. 변환 규칙 또는 생성된 커널 코드 확인 필요.",
        "detail": "생성된 커널 체인: narrow_split → vector_fp_binary(DivF) → widen_concat",
    },
    {
        "op_name": "rsqrt",
        "family": "elementwise",
        "source_op": "linalg.generic(math.rsqrt)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "max(2%, 1e-3)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch가 계산한 rsqrt(x) = 1/sqrt(x) (x는 0.5~1.5 범위로 생성, 정의역 보장)",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 허용 오차 이내로 일치. 하드웨어에 Rsqrt가 직접 없어 Sqrt+나누기 조합으로 구현했는데도 정확함을 확인.",
        "meaning_fail": "오차가 허용 범위를 초과. Sqrt+1.0나누기 조합 로직 확인 필요.",
        "detail": "생성된 커널 체인: narrow_split → vector_fp_unary(Sqrt) → vector_fp_div_with_mode(Mode10, 1.0) → widen_concat. 단항(입력 텐서 1개) 연산을 처음 지원하는 케이스.",
    },
    {
        "op_name": "sqrt",
        "family": "elementwise",
        "source_op": "linalg.generic(math.sqrt)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "max(2%, 1e-3)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch가 계산한 sqrt(x) (x는 0.5~1.5 범위로 생성, 정의역 보장)",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 허용 오차 이내로 일치. FpUnaryOp::Sqrt 단독으로 구현.",
        "meaning_fail": "오차가 허용 범위를 초과. Sqrt 커널 로직 확인 필요.",
        "detail": "생성된 커널 체인: narrow_split → vector_fp_unary(Sqrt) → widen_concat. rsqrt와 달리 역수 없이 Sqrt 단독.",
    },
    {
        "op_name": "exp",
        "family": "elementwise",
        "source_op": "linalg.generic(math.exp)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "max(2%, 1e-3)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch가 계산한 exp(x) (x는 -2~2 범위 정규분포)",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 허용 오차 이내로 일치. FpUnaryOp::Exp 단독으로 구현.",
        "meaning_fail": "오차가 허용 범위를 초과. Exp 커널 로직 확인 필요.",
        "detail": "생성된 커널 체인: narrow_split → vector_fp_unary(Exp) → widen_concat. sqrt와 동일한 구조.",
    },
    {
        "op_name": "sigmoid",
        "family": "elementwise",
        "source_op": "linalg.generic(arith.negf+math.exp+arith.addf+arith.divf)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "max(2%, 1e-3)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch가 계산한 sigmoid(x) = 1/(1+exp(-x))",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 허용 오차 이내로 일치. FpUnaryOp::Sigmoid 단독으로 구현. negf+exp+addf+divf 복합 패턴을 단일 하드웨어 op으로 대체.",
        "meaning_fail": "오차가 허용 범위를 초과. Sigmoid 커널 로직 확인 필요.",
        "detail": "SiLU(x) = x*sigmoid(x)의 sigmoid 부분. 복합 linalg.generic 블록을 FpUnaryOp::Sigmoid 단독으로 교체.",
    },
    {
        "op_name": "tanh",
        "family": "elementwise",
        "source_op": "linalg.generic(math.tanh)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "max(2%, 1e-3)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch가 계산한 tanh(x)",
        "meaning_pass": "FpUnaryOp::Tanh 단독 구현 — PASS",
        "meaning_fail": "오차 초과",
        "detail": "sqrt/exp와 동일한 단항 unary 패턴",
    },
    {
        "op_name": "sin",
        "family": "elementwise",
        "source_op": "linalg.generic(math.sin)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "max(2%, 1e-3)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch가 계산한 sin(x)",
        "meaning_pass": "FpUnaryOp::Sin 단독 구현 — PASS",
        "meaning_fail": "오차 초과",
        "detail": "sqrt/exp와 동일한 단항 unary 패턴",
    },
    {
        "op_name": "cos",
        "family": "elementwise",
        "source_op": "linalg.generic(math.cos)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "max(2%, 1e-3)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch가 계산한 cos(x)",
        "meaning_pass": "FpUnaryOp::Cos 단독 구현 — PASS",
        "meaning_fail": "오차 초과",
        "detail": "sqrt/exp와 동일한 단항 unary 패턴",
    },
    {
        "op_name": "pow2",
        "family": "elementwise",
        "source_op": "linalg.generic(math.powf, 지수=2.0)",
        "target_op": "rngd.elementwise",
        "dtype": "f32",
        "tolerance": "max(2%, 1e-3)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch가 계산한 x^2 (지수가 정확히 2.0인지 IR에서 실제 값 검증 후 매치)",
        "meaning_pass": "PyTorch 출력과 RNGD 시뮬레이터 출력이 허용 오차 이내로 일치. 하드웨어에 Pow가 직접 없어 MulF(x,x)로 구현했는데도 정확함을 확인.",
        "meaning_fail": "오차가 허용 범위를 초과. 자기 자신을 VRF로 적재하는 로직 확인 필요.",
        "detail": "생성된 커널: 같은 입력을 메인 스트림 + VRF로 두 번 적재 → vector_fp_binary(MulF, self_vrf). RMSNorm의 variance=mean(x^2) 계산에 필요한 첫 조각.",
    },
    {
        "op_name": "batch_gemm",
        "family": "contraction",
        "source_op": "linalg.batch_matmul",
        "target_op": "rngd.batch_gemm",
        "dtype": "bf16",
        "tolerance": "max(5%, 1.0)",
        "experimental": 0,
        "status": "done",
        "compares": "bf16 양자화 입력 기준 PyTorch 계산값 (원본 f32 결과 아님)",
        "meaning_pass": "bf16 양자화 입력 기준, PyTorch 계산값과 RNGD 시뮬레이터 출력이 허용 오차 이내로 일치. 잔차는 bf16 반올림 범위.",
        "meaning_fail": "오차가 bf16 반올림 범위를 초과. 행렬곱 커널 로직 결함 가능성.",
        "detail": "생성된 커널: TRF 적재 → contract_outer → contract_packet → contract_time → contract_lane",
    },
    {
        "op_name": "gemm",
        "family": "contraction",
        "source_op": "linalg.matmul",
        "target_op": "rngd.gemm",
        "dtype": "bf16",
        "tolerance": "max(5%, 1.0)",
        "experimental": 0,
        "status": "done",
        "compares": "bf16 양자화 입력 기준 PyTorch 계산값 (원본 f32 결과 아님)",
        "meaning_pass": "bf16 양자화 입력 기준, PyTorch 계산값과 RNGD 시뮬레이터 출력이 허용 오차 이내로 일치. 잔차는 bf16 반올림 범위.",
        "meaning_fail": "오차가 bf16 반올림 범위를 초과. 행렬곱 커널 로직 결함 가능성.",
        "detail": "batch_gemm과 동일 커널 구조를 V=1로 재사용 (TRF 적재 → contract_outer/packet/time/lane)",
    },
    {
        "op_name": "dot_product",
        "family": "contraction",
        "source_op": "linalg.dot",
        "target_op": "rngd.dot_product",
        "dtype": "bf16",
        "tolerance": "max(5%, 1.0)",
        "experimental": 0,
        "status": "done",
        "compares": "bf16 양자화 입력 기준 PyTorch 계산값 (원본 f32 결과 아님)",
        "meaning_pass": "bf16 양자화 입력 기준, PyTorch 계산값과 RNGD 시뮬레이터 출력이 허용 오차 이내로 일치. 잔차는 bf16 반올림 범위.",
        "meaning_fail": "오차가 bf16 반올림 범위를 초과. 내적 커널 로직 결함 가능성.",
        "detail": (
            "batch_gemm과는 별도의 전용 커널 구조 사용 (M=1,N=1 degenerate case가 하드웨어 "
            "제약(Time::SIZE가 Lane::SIZE(8)로 안 나눠짐)에 걸려 재사용 불가). "
            "고정 2048 버퍼 + 벡터 전체 reduction 방식."
        ),
    },
]

# evidence_type: 'static'  = 소스 코드 grep으로 확인 (실행 안 함)
#               'runtime' = cargo furiosa-opt test 실행 결과(panic or pass)
EVIDENCE = [
    ("add",         "static",  "FpBinaryOp::AddF 존재",                        "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:380"),
    ("sub",         "static",  "FpBinaryOp::SubF 존재",                        "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:382"),
    ("mul",         "static",  "FpBinaryOp::MulF(FpMulAlu) 존재",              "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:384"),
    ("div",         "static",  "FpBinaryOp::DivF 존재",                        "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:388"),
    ("rsqrt",       "static",  "FpUnaryOp::Sqrt 존재 (Rsqrt는 없음)",          "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:329"),
    ("rsqrt",       "static",  "vector_fp_unary(op: FpUnaryOp) 시그니처",      "furiosa-opt-std-0.3.0/src/engine/vector/tensor/vector_tensor.rs:1296"),
    ("rsqrt",       "static",  "BinaryArgMode::Mode10 = op(operand0, mainstream) — 분자/분모 순서 반전용", "furiosa-opt-std-0.3.0/src/engine/vector/op/arg_mode.rs:40"),
    ("sqrt",         "static",  "FpUnaryOp::Sqrt 존재 — rsqrt 구현 시 이미 확인됨",       "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:329"),
    ("sqrt",         "runtime", "vector_fp_unary(Sqrt) 단독으로 정합성 검증 통과",          "cargo furiosa-opt test: test_log_sqrt.txt — PASS"),
    ("exp",          "static",  "FpUnaryOp::Exp 존재 — op/mod.rs:324 확인",               "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:324"),
    ("exp",          "runtime", "vector_fp_unary(Exp) 단독으로 정합성 검증 통과",           "cargo furiosa-opt test: test_log_exp.txt — PASS"),
    ("sigmoid",      "static",  "FpUnaryOp::Sigmoid 존재 — op/mod.rs:330 확인",             "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:330"),
    ("sigmoid",      "runtime", "vector_fp_unary(Sigmoid) 단독으로 정합성 검증 통과",        "cargo furiosa-opt test: test_log_sigmoid.txt — PASS"),
    ("tanh",         "static",  "FpUnaryOp::Tanh 존재 확인",                               "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs"),
    ("tanh",         "runtime", "vector_fp_unary(Tanh) 정합성 검증 통과",                  "cargo furiosa-opt test: test_log_tanh.txt — PASS"),
    ("sin",          "static",  "FpUnaryOp::Sin 존재 확인",                                "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs"),
    ("sin",          "runtime", "vector_fp_unary(Sin) 정합성 검증 통과",                   "cargo furiosa-opt test: test_log_sin.txt — PASS"),
    ("cos",          "static",  "FpUnaryOp::Cos 존재 확인",                                "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs"),
    ("cos",          "runtime", "vector_fp_unary(Cos) 정합성 검증 통과",                   "cargo furiosa-opt test: test_log_cos.txt — PASS"),
    ("gemv",         "static",  "gemv_kernel.rs 존재 — Contraction Engine 패턴 확인",         "rngd-tcp-kernel-dev/src/kernel/gemv_kernel.rs"),
    ("gemv",         "runtime", "linalg.matvec → rngd.gemv 정합성 검증 통과 (I=256, J=2048)", "cargo furiosa-opt test: test_log_gemv.txt — PASS"),
    ("pow2",        "static",  "FpBinaryOp::MulF 존재 (Pow는 없음, MulF(x,x)로 조합)", "furiosa-opt-std-0.3.0/src/engine/vector/op/mod.rs:384"),
    ("pow2",        "runtime", "지수 값(2.0)을 IR의 arith.constant에서 FloatAttr로 직접 읽어 검증", "본 채팅 기록 — RMSNorm IR 조사"),
    ("batch_gemm",  "runtime", "linalg.batch_matmul ← torch.bmm 실측 확인",   "inspect 스크립트 실행 결과 (본 채팅 기록)"),
    ("gemm",        "runtime", "linalg.matmul(2D) ← torch.matmul 실측 확인",  "inspect_matmul_2d.py 실행 결과 (본 채팅 기록)"),
    ("dot_product", "runtime", "linalg.dot ← torch.matmul(1D,1D) 실측 확인 (torch.dot 자체는 미지원)", "inspect_dot_product_v2.py 실행 결과 (본 채팅 기록)"),
    ("dot_product", "runtime", "contract_lane의 OutPacket::SIZE는 정확히 8이어야 함", "cargo panic: furiosa-opt-std-0.3.0/src/engine/contraction/lane.rs:87"),
    ("dot_product", "runtime", "Lane::SIZE(8)이 Time::SIZE를 나눠야 함 (M=1처럼 작으면 깨짐)", "cargo panic: furiosa-opt-std-0.3.0/src/engine/contraction/collect.rs:146"),
]

HARDWARE_CONSTRAINTS = [
    {
        "description": "Slice 크기는 64/128/192/256 중 하나여야 함. 임의 크기를 넣으면 시뮬레이터가 즉시 panic.",
        "discovery_source": "cargo panic 메시지: 'Fetch: Slice size must be one of 64/128/192/256, got 16' — furiosa-opt-std-0.3.0/src/engine/fetch.rs:59",
    },
    {
        "description": "contract_lane의 OutPacket::SIZE는 정확히 8이어야 함. dot_product처럼 출력 1개인 경우에도 8-lane 공간에 패딩 필요.",
        "discovery_source": "cargo panic 메시지: 'contract_lane: OutPacket::SIZE must be 8, got 1' — furiosa-opt-std-0.3.0/src/engine/contraction/lane.rs:87",
    },
    {
        "description": "Lane::SIZE(8)이 Time::SIZE를 나눌 수 있어야 함. M=1처럼 극단적으로 작은 차원에서 위반됨.",
        "discovery_source": "cargo panic 메시지: 'Lane::SIZE (8) does not divide Time::SIZE (2)' — furiosa-opt-std-0.3.0/src/engine/contraction/collect.rs:146",
    },
    {
        "description": "나눗셈(DivF)에서 BinaryArgMode::Mode10이 op(operand0, mainstream) — 즉 분모/분자 순서가 일반적 직관과 반대.",
        "discovery_source": "runtime 실측: rsqrt 커널 출력이 기댓값의 정확한 역수로 나오는 것을 관찰 후 API 소스 재조사로 확인",
    },
]

# 연산 ↔ 하드웨어 제약 다대다 링크 (constraint는 위 리스트의 0-based 인덱스)
OP_CONSTRAINT_LINKS = [
    ("batch_gemm",  0),  # Slice 제약
    ("gemm",        0),  # Slice 제약
    ("dot_product", 0),  # Slice 제약
    ("dot_product", 1),  # OutPacket::SIZE = 8
    ("dot_product", 2),  # Lane::SIZE divides Time::SIZE
    ("rsqrt",       3),  # Mode10 나눗셈 순서
    ("div",         3),  # Mode10 나눗셈 순서 (DivF 자체는 Mode00 사용, 참고용)
]

ROADMAP = [
    {
        "op_name": "gemv",
        "family": "gemv",
        "source_op": "linalg.matvec",
        "target_op": "rngd.gemv",
        "dtype": "bf16",
        "tolerance": "max(2%, 0.5)",
        "experimental": 1,
        "status": "done",
        "compares": "PyTorch torch.mv(A, x) — A:[256,2048] bf16, x:[2048] bf16",
        "meaning_pass": "gemv_kernel.rs 구조(Contraction Engine) 그대로 재사용. PASS",
        "meaning_fail": "오차 초과. gemv 커널 로직 확인 필요.",
        "detail": "I=256, J=2048. dot_product와 동일한 Contraction Engine 패턴. linalg.matvec → rngd.gemv.",
    },
    ("rngd.reduce (mean)",  "설계중",
     "구조적 제약 발견: A가 8의 배수이면 m![A%8]=m![0]이 no-op이 되어 IntraSliceReduce 파이프라인 구성 불가. "
     "InterFirst(VectorInitTensor→vector_inter_slice_reduce) 경로는 Way8 유지로 commit까지 가능하나 Slice 내 원소를 모두 합산하지 못함(각 Packet lane 독립 합산). "
     "InterFirst 이후 IntraSliceReduce 추가 불가(Way8에서 vector_intra_slice_reduce 없음). "
     "결론: A가 8의 배수인 경우 완전한 reduce 구현 방법을 API로 찾지 못함. FuriosaAI에 문의 필요 — 핵심 질문: 8의 배수 크기 텐서에서 전체 원소를 scalar로 reduce하는 올바른 파이프라인 패턴은?"),
    ("브로드캐스트 곱셈",   "완료",
     "RMSNorm의 weight*x, rsqrt_val*x처럼 shape이 다른 두 텐서의 곱. "
     "check_elementwise에서 indexing_maps를 검사하지 않아 arith.mulf가 있으면 무조건 rngd.elementwise(mul)로 처리됨. "
     "2D×scalar, 2D×1D 브로드캐스트 모두 llama_rmsnorm에서 실제 동작 확인."),
    ("gemv (행렬-벡터)",    "미착수",
     "rngd-tcp-kernel-dev에 이미 검증된 base 템플릿(gemv_kernel.rs) 존재 — 재사용 가능성 높음."),
    ("rngd.transpose",      "미착수", "원래 설계 문서의 7개 확정 RNGD op 중 하나."),
    ("rngd.fill",           "미착수", "지금은 DCE로 지워버리는 대상 — 별도 op으로 다룰 필요가 있는지 재검토 필요."),
    ("rngd.conv2d",         "미착수", "원래 설계 문서의 7개 확정 RNGD op 중 하나. 아직 손도 안 댐."),
    ("rngd.elementwise (silu)", "완료",
     "SiLU(x) = x * sigmoid(x). sigmoid 블록(negf+exp+addf+divf)을 rngd.elementwise(sigmoid)로, "
     "mulf 블록을 rngd.elementwise(mul)로 각각 재작성. 두 단계 조합으로 완전 지원. "
     "e2e_pipeline.py에서 sigmoid 복합 패턴 감지 로직 추가 완료."),
    ("rngd.elementwise (log)", "설계중",
     "FpUnaryOp::Log는 존재하나 시뮬레이터 검증 실패. PyTorch ln(x)와 출력 불일치 및 NaN 발생. "
     "시뮬레이터의 Log 구현이 자연로그가 아닐 가능성 — FuriosaAI 확인 필요."),
    ("V>256 타일링 전략",   "미착수",
     "지금 Family B는 V≤256 가정(Slice=m![V # 256])에서만 검증됨. 배치 축이 256 넘는 경우 미구현."),
]

# ──────────────────────────────────────────────────────────────────────────────
# 초기화 로직
# ──────────────────────────────────────────────────────────────────────────────

def init(db_path: Path, reset: bool = False) -> None:
    if reset and db_path.exists():
        db_path.unlink()
        print(f"기존 DB 삭제: {db_path}")

    con = sqlite3.connect(db_path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)

    # ops
    con.executemany(
        """INSERT OR IGNORE INTO ops
           (op_name, family, source_op, target_op, dtype, tolerance,
            experimental, status, compares, meaning_pass, meaning_fail, detail)
           VALUES (:op_name, :family, :source_op, :target_op, :dtype, :tolerance,
                   :experimental, :status, :compares, :meaning_pass, :meaning_fail, :detail)""",
        OPS,
    )
    print(f"ops: {len(OPS)}개 삽입")

    # evidence
    con.executemany(
        "INSERT OR IGNORE INTO evidence (op_name, evidence_type, claim, source) VALUES (?,?,?,?)",
        EVIDENCE,
    )
    print(f"evidence: {len(EVIDENCE)}개 삽입")

    # hardware_constraints — INSERT OR IGNORE가 안 되므로 이미 있는지 확인
    con.execute("DELETE FROM hardware_constraints")
    con.execute("DELETE FROM op_constraint_link")
    rows = [(c["description"], c["discovery_source"]) for c in HARDWARE_CONSTRAINTS]
    con.executemany(
        "INSERT INTO hardware_constraints (description, discovery_source) VALUES (?,?)", rows
    )
    print(f"hardware_constraints: {len(rows)}개 삽입")

    # constraint_id는 rowid 순서 = 삽입 순서와 동일하므로 1-based index로 매핑
    for op_name, idx in OP_CONSTRAINT_LINKS:
        con.execute(
            "INSERT OR IGNORE INTO op_constraint_link (op_name, constraint_id) VALUES (?,?)",
            (op_name, idx + 1),
        )
    print(f"op_constraint_link: {len(OP_CONSTRAINT_LINKS)}개 삽입")

    # roadmap
    con.execute("DELETE FROM roadmap")
    con.executemany(
        "INSERT INTO roadmap (name, status, note) VALUES (?,?,?)", ROADMAP
    )
    print(f"roadmap: {len(ROADMAP)}개 삽입")

    con.commit()
    con.close()
    print(f"\n완료: {db_path}")


def verify(db_path: Path) -> None:
    """삽입 결과를 간단히 출력해서 눈으로 확인."""
    con = sqlite3.connect(db_path)
    for table in ("ops", "evidence", "hardware_constraints", "op_constraint_link", "roadmap"):
        (cnt,) = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        print(f"  {table}: {cnt}행")

    print("\n--- ops ---")
    for row in con.execute("SELECT op_name, family, dtype, status FROM ops ORDER BY family, op_name"):
        print(" ", row)

    print("\n--- op ↔ constraint ---")
    for row in con.execute(
            """SELECT l.op_name, c.description
               FROM op_constraint_link l
                        JOIN hardware_constraints c ON l.constraint_id = c.id
               ORDER BY l.op_name"""
    ):
        print(f"  {row[0]:15s} → {row[1][:60]}...")

    con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="DB를 삭제하고 처음부터 다시 생성")
    parser.add_argument("--db", default=str(DB_PATH), help="DB 파일 경로")
    args = parser.parse_args()

    init(Path(args.db), reset=args.reset)
    print("\n=== 검증 ===")
    verify(DB_PATH)
    