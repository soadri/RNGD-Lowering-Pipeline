# RNGD Lowering Pipeline

표준 MLIR linalg IR을 FuriosaAI RNGD NPU의 vISA Rust 커널로 낮춰(lower)주는 규칙을
자동 생성·검증하는 개발/테스트 환경이다.

**CI 리포트 (GitHub Pages):** https://soadri.github.io/RNGD-Lowering-Pipeline/

---

## 현재 구현 범위

### 지원 연산 (17종, 전부 PASS)

| 연산 | Family | linalg → RNGD | dtype |
|---|---|---|---|
| `add` | Elementwise (A) | `linalg.generic(arith.addf)` → `rngd.elementwise` | f32 |
| `sub` | Elementwise (A) | `linalg.generic(arith.subf)` → `rngd.elementwise` | f32 |
| `mul` | Elementwise (A) | `linalg.generic(arith.mulf)` → `rngd.elementwise` | f32 |
| `div` | Elementwise (A) | `linalg.generic(arith.divf)` → `rngd.elementwise` | f32 |
| `rsqrt` ⚗️ | Elementwise (A) | `linalg.generic(math.rsqrt)` → `rngd.elementwise` | f32 |
| `sqrt` | Elementwise (A) | `linalg.generic(math.sqrt)` → `rngd.elementwise` | f32 |
| `exp` | Elementwise (A) | `linalg.generic(math.exp)` → `rngd.elementwise` | f32 |
| `sigmoid` ⚗️ | Elementwise (A) | `linalg.generic(negf+exp+addf+divf)` → `rngd.elementwise` | f32 |
| `tanh` | Elementwise (A) | `linalg.generic(math.tanh)` → `rngd.elementwise` | f32 |
| `sin` | Elementwise (A) | `linalg.generic(math.sin)` → `rngd.elementwise` | f32 |
| `cos` | Elementwise (A) | `linalg.generic(math.cos)` → `rngd.elementwise` | f32 |
| `pow2` ⚗️ | Elementwise (A) | `linalg.generic(math.powf, 지수=2.0)` → `rngd.elementwise` | f32 |
| `batch_gemm` | Contraction (B) | `linalg.batch_matmul` → `rngd.batch_gemm` | bf16 |
| `gemm` | Contraction (B) | `linalg.matmul` → `rngd.gemm` | bf16 |
| `dot_product` | Contraction (B) | `linalg.dot` → `rngd.dot_product` | bf16 |
| `gemv` | Contraction (B) | `linalg.matvec` → `rngd.gemv` | bf16 |
| `transpose` | Transpose | `linalg.generic(identity)` → `rngd.transpose` | f32 |

> ⚗️ 하드웨어에 직접 해당 연산이 없어 다른 연산들의 조합으로 구현한 케이스.

### 파이프라인 3단계

```
PyTorch 모델
    ↓  torch_mlir.compile()
linalg IR  (표준, 하드웨어 무관)
    ↓  e2e_pipeline.py / rewrite_to_rngd()
RNGD IR    (rngd.* ops, KETI 내부 placeholder)
    ↓  코드 생성 + cargo furiosa-opt test
Rust vISA 커널  (FuriosaAI 기능 시뮬레이터로 정합성 검증)
```

### 검증 현황

- **Family A (f32):** PyTorch 독립 계산값과 허용 오차 `1e-4` 이내 일치 확인
- **Family B (bf16):** bf16 양자화 기준 허용 오차 `max(2%, 0.5)` 이내 일치 확인
- 검증 환경: FuriosaAI `furiosa-opt-std v0.3.0` 기능 시뮬레이터 (실리콘 아님)

### 모델 커버리지 (전체 100%)

| 모델 | 커버리지 | 설명 |
|---|---|---|
| `example_linear` | 🟢 100% | 단순 Linear 레이어 |
| `example_rmsnorm` | 🟢 100% | RMSNorm (reduce 제외) |
| `llama_rmsnorm` | 🟢 100% | Llama-3.1 RMSNorm 구조 |
| `llama_ffn` | 🟢 100% | Llama-3.1 FFN (SiLU 포함) |
| `llama_ffn_real` | 🟢 100% | Llama-3.1 FFN 실제 비율 (dim=256, hidden=896) |
| `simple_mlp` | 🟢 100% | reduce-free 2-layer MLP |
| `transformer_no_softmax` | 🟢 100% | Transformer Attention+FFN (softmax 제외) |

### 아직 미구현

- `rngd.reduce (mean)` — RMSNorm의 `mean(x²)`, Softmax의 핵심 블로커. 시뮬레이터 구조적 제약 발견, FuriosaAI 문의 필요
- `rngd.elementwise (log)` — 시뮬레이터 출력 불일치 및 NaN 발생, FuriosaAI 확인 필요
- `rngd.elementwise (erf)` — 시뮬레이터 미구현 (`not yet implemented`)
- `rngd.conv2d` — 미착수

---

## CI/CD

push 또는 PR이 생성되면 GitHub Actions가 자동으로 실행된다.  
서버를 self-hosted runner로 등록해 실제 환경(`.venv`)을 그대로 사용한다.

### Job 구성

| Job | 트리거 | 역할 |
|---|---|---|
| `validate-registry` | push / PR | DB 재생성 + 변환 규칙 evidence 무결성 검증 |
| `coverage-analysis` | push / PR | 모델 → linalg IR → 지원/미지원 op 분류 |
| `kernel-test` | coverage 후 | 지원 op 커널 빌드 + 정합성 검증 |
| `build-report` | main push만 | 분석 결과 → GitHub Pages HTML 대시보드 배포 |
| `pr-comment` | PR만 | 커버리지 요약을 PR에 자동 코멘트 |

---

## 모델 커버리지 자동 측정 방법

`models/` 디렉토리에 모델 파일을 추가하고 push하면 된다.

### 1. 모델 파일 작성

```python
# models/my_model.py
import torch

def get_model() -> torch.nn.Module:
    """분석할 PyTorch 모델을 반환한다."""
    return MyModel()

def get_sample_inputs() -> tuple:
    """모델 forward에 넣을 샘플 입력을 반환한다."""
    return (torch.randn(1, 64),)
```

**주의사항:**
- `get_model()`과 `get_sample_inputs()` 두 함수 모두 반드시 있어야 한다.
- `get_model()`은 `torch.nn.Module`을 반환해야 한다.
- `get_sample_inputs()`는 tuple을 반환해야 한다 (원소가 1개여도 `(x,)` 형태로).
- 모델이 `torch_mlir.compile()`로 컴파일 가능해야 한다.

### 2. `ci.yml`에 모델 등록

`.github/workflows/ci.yml`의 `coverage-analysis` job의 `matrix.model`에 추가한다.

```yaml
strategy:
  matrix:
    model:
      - models/example_linear.py
      - models/my_model.py      # ← 추가
```

### 3. 커밋 & push

```bash
git add models/my_model.py .github/workflows/ci.yml
git commit -m "feat: add my_model coverage analysis"
git push
```

---

## 로컬 실행

```bash
# 환경 활성화
cd ~/rngd-mlir-pipeline
source .venv/bin/activate

# 특정 연산 실행 및 테스트
python e2e_pipeline.py add
python deploy_and_test.py add

# 모델 커버리지 직접 분석
python ci_analyze.py --model models/llama_ffn.py --out-dir ci_out/llama_ffn

# 웹 대시보드 (SSH 터널 후 http://localhost:8000)
sudo systemctl restart rngd-dashboard
```

---

## 프로젝트 구조

```
rngd-mlir-pipeline/
├── e2e_pipeline.py        IR 감지·재작성 + 커널/참조데이터 코드 생성 (핵심)
├── deploy_and_test.py     배치 + cargo furiosa-opt test 실행
├── dashboard_app.py       FastAPI 백엔드
├── dashboard_index.html   웹 대시보드 프론트엔드
├── ci_analyze.py          모델 커버리지 분석 스크립트
├── build_pages.py         GitHub Pages HTML 빌드
├── init_registry.py       변환 규칙 DB 초기화
├── registry.py            DB 읽기 헬퍼
├── models/                CI 분석 대상 모델 파일들
└── .github/workflows/
    └── ci.yml             GitHub Actions 워크플로우

rngd-tcp-kernel-dev/       (별도 레포)
└── src/
    ├── kernel/            생성된 vISA 커널 (17종)
    ├── pilot_e2e_{op}.rs  테스트 파일
    └── reference_data_*   PyTorch 실측 기반 참조값
```

---

## 환경

- **서버:** Ubuntu 24.04
- **Python:** 3.11 + `.venv` (torch-mlir 기반)
- **Rust:** furiosa-opt-std v0.3.0 (FuriosaAI 기능 시뮬레이터)
- **CI:** GitHub Actions (self-hosted runner)