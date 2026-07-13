# RNGD Lowering Pipeline

KETI K-Cloud 컨소시엄(FuriosaAI · Nota AI · POSTECH) 과제용 bridge 컴파일러.  
표준 MLIR linalg IR을 FuriosaAI RNGD NPU의 vISA Rust 커널로 낮춰(lower)주는 규칙을
자동 생성·검증하는 개발/테스트 환경이다.

**CI 리포트 (GitHub Pages):** https://soadri.github.io/RNGD-Lowering-Pipeline/

---

## 현재 구현 범위

### 지원 연산 (9종, 전부 PASS)

| 연산 | Family | linalg → RNGD | dtype |
|---|---|---|---|
| `add` | Elementwise (A) | `linalg.generic(arith.addf)` → `rngd.elementwise` | f32 |
| `sub` | Elementwise (A) | `linalg.generic(arith.subf)` → `rngd.elementwise` | f32 |
| `mul` | Elementwise (A) | `linalg.generic(arith.mulf)` → `rngd.elementwise` | f32 |
| `div` | Elementwise (A) | `linalg.generic(arith.divf)` → `rngd.elementwise` | f32 |
| `rsqrt` ⚗️ | Elementwise (A) | `linalg.generic(math.rsqrt)` → `rngd.elementwise` | f32 |
| `pow2` ⚗️ | Elementwise (A) | `linalg.generic(math.powf, 지수=2.0)` → `rngd.elementwise` | f32 |
| `batch_gemm` | Contraction (B) | `linalg.batch_matmul` → `rngd.batch_gemm` | bf16 |
| `gemm` | Contraction (B) | `linalg.matmul` → `rngd.gemm` | bf16 |
| `dot_product` | Contraction (B) | `linalg.dot` → `rngd.dot_product` | bf16 |

> ⚗️ 하드웨어에 직접 해당 연산이 없어 다른 연산들의 조합으로 구현한 케이스.

### 파이프라인 3단계

```
PyTorch 모델
    ↓  torch_mlir.compile()
linalg IR  (표준, 하드웨어 무관)
    ↓  e2e_pipeline.py / rewrite_to_rngd()
RNGD IR    (rngd.* ops, KETI 내부 placeholder)
    ↓  코드 생성 + cargo furiosa-opt test
Rust vISA 커널  (FuriosaAI 시뮬레이터로 정합성 검증)
```

### 검증 현황

- **Family A (f32):** PyTorch 독립 계산값과 허용 오차 `1e-4` 이내 일치 확인
- **Family B (bf16):** bf16 양자화 기준 허용 오차 `max(5%, 1.0)` 이내 일치 확인
- 검증 환경: FuriosaAI `furiosa-opt-std v0.3.0` 시뮬레이터 (기능적 인터프리터, 실리콘 아님)

### 아직 미구현

- `rngd.reduce (mean)` — RMSNorm의 `mean(x²)` 계산에 필요, 설계 진행 중
- 브로드캐스트 곱셈, gemv, transpose, conv2d 등

---

## CI/CD

push 또는 PR이 생성되면 GitHub Actions가 자동으로 실행된다.  
k-cloud 서버를 self-hosted runner로 등록해 서버의 실제 환경(`.venv`)을 그대로 사용한다.

### Job 구성

| Job | 트리거 | 역할 |
|---|---|---|
| `validate-registry` | push / PR | DB 재생성 + 변환 규칙 evidence 무결성 검증 |
| `coverage-analysis` | push / PR | 모델 → linalg IR → 지원/미지원 op 분류 |
| `build-report` | main push만 | 분석 결과 → GitHub Pages HTML 대시보드 배포 |
| `pr-comment` | PR만 | 커버리지 요약을 PR에 자동 코멘트 |

---

## 모델 커버리지 자동 측정 방법

파이프라인이 특정 PyTorch 모델을 얼마나 지원하는지 자동으로 측정할 수 있다.  
`models/` 디렉토리에 모델 파일을 추가하고 push하면 된다.

### 1. 모델 파일 작성

`models/` 아래에 `.py` 파일을 하나 만든다. 규격은 다음과 같다.

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
- 모델이 `torch_mlir.compile()`로 컴파일 가능해야 한다 (동적 제어 흐름 등은 제한될 수 있음).

### 2. `ci.yml`에 모델 등록

`.github/workflows/ci.yml`의 `coverage-analysis` job의 `matrix.model`에 추가한다.

```yaml
strategy:
  matrix:
    model:
      - models/example_linear.py
      - models/example_rmsnorm.py
      - models/my_model.py      # ← 추가
```

### 3. 커밋 & push

```bash
git add models/my_model.py .github/workflows/ci.yml
git commit -m "feat: add my_model coverage analysis"
git push
```

push하면 CI가 자동으로 실행된다.

---

## 결과 예시

### GitHub Actions 로그

```
[1/3] 모델 컴파일 중: models/example_rmsnorm.py
  → linalg IR 저장: ci_out/example_rmsnorm/model_ir.mlir
[2/3] op 분석 중
[3/3] Markdown 리포트 생성 중
  → 리포트 저장: ci_out/example_rmsnorm/coverage_report.md

============================================================
  커버리지: 57.1%  (4/7개 op 지원)
  미지원 op (3개):
    · linalg.generic (reduce)
    · math.sqrt
    · tensor.collapse_shape
============================================================
```

### PR 자동 코멘트

PR을 열면 다음과 같은 코멘트가 자동으로 달린다.

```
## 🔍 RNGD Lowering 커버리지 분석

| 모델            | 커버리지    | 지원 | 미지원 |
|----------------|------------|------|--------|
| example_linear  | 🟢 100%   | 2개  | 0개    |
| example_rmsnorm | 🟡 57.1%  | 4개  | 3개    |

<details><summary>미지원 op (3개)</summary>

- `linalg.generic (reduce)`
- `math.sqrt`
- `tensor.collapse_shape`

</details>
```

### GitHub Pages 리포트

https://soadri.github.io/RNGD-Lowering-Pipeline/

main push마다 자동으로 갱신된다. 모델별 커버리지 카드와 지원 연산 레지스트리 전체가 표시된다.

---

## 로컬 실행

```bash
# 환경 활성화
cd ~/rngd-mlir-pipeline
source .venv/bin/activate

# 특정 연산 실행
python e2e_pipeline.py add
python deploy_and_test.py add

# 전체 연산 실행
python e2e_pipeline.py
# 연산별로 deploy_and_test.py 실행

# 모델 커버리지 직접 분석
python ci_analyze.py --model models/example_rmsnorm.py --out-dir ci_out/rmsnorm

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
    ├── kernel/            생성된 vISA 커널 (9종)
    ├── pilot_e2e_{op}.rs  테스트 파일
    └── reference_data_*   PyTorch 실측 기반 참조값
```

---

## 환경

- **서버:** Naver Cloud k-cloud (Ubuntu 24.04)
- **Python:** 3.11 + `.venv` (torch 2.3.0.dev20240122, torch_mlir 20240127.1096)
- **Rust:** furiosa-opt-std v0.3.0 (FuriosaAI vISA SDK)
- **CI:** GitHub Actions (self-hosted runner on k-cloud)

---

*KETI RNGD NPU Compiler · K-Cloud 컨소시엄*