# RNGD Lowering Pipeline

표준 MLIR linalg IR을 FuriosaAI RNGD NPU의 vISA Rust 커널로 낮춰(lower)주는 규칙을
자동 생성·검증하는 개발/테스트 환경이다.

**CI 리포트 (GitHub Pages):** https://soadri.github.io/RNGD-Lowering-Pipeline/  
**Agent 실험 현황:** https://soadri.github.io/RNGD-Lowering-Pipeline/agent.html

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

### 워크플로우 구성 (3개)

**`ci.yml` — 기존 모델 검증**

| Job | 트리거 | 역할 |
|---|---|---|
| `validate-registry` | push / PR | DB 재생성 + 변환 규칙 evidence 무결성 검증 |
| `coverage-analysis` | push / PR | 모델 → linalg IR → 지원/미지원 op 분류 |
| `kernel-test` | coverage 후 | 지원 op 커널 빌드 + 정합성 검증 |
| `build-report` | main push만 | 분석 결과 → GitHub Pages HTML 대시보드 배포 |
| `pr-comment` | PR만 | 커버리지 요약을 PR에 자동 코멘트 |

**`agent_ci.yml` — 자동 실험 모델 검증**

| Job | 트리거 | 역할 |
|---|---|---|
| `agent-coverage` | `models/agent/agent_*.py` push | 자동 생성 모델 커버리지 분석 |
| `agent-report` | agent-coverage 후 | agent.html 빌드 + GitHub Pages 배포 |

**`upstream_track.yml` — PyTorch 버전 추적**

| Job | 트리거 | 역할 |
|---|---|---|
| `track-upstream` | 매주 월요일 09:00 KST | PyTorch op 패턴 스냅샷 생성 + 변경 감지 시 GitHub Issue 생성 |

---

## 🤖 자동 연산 조합 실험 (RNGD Agent)

**Agent 결과 페이지:** https://soadri.github.io/RNGD-Lowering-Pipeline/agent.html

RNGD Agent가 지원 연산 17종을 자동으로 조합하여 2~4 레이어 모델을 생성하고
CI 파이프라인을 통해 커버리지를 검증합니다.

### 실험 규모

| 레이어 | 패턴 | 조합 수 |
|---|---|---|
| 2-layer | Contraction → Elementwise | 48가지 |
| 3-layer | Contraction → Elementwise → Contraction | 144가지 |
| 4-layer | Contraction → Elementwise → Contraction → Elementwise | 1,728가지 |
| **전체** | | **1,920가지** |

> `batch_gemm`은 첫 번째 레이어에만 허용 (중간 위치에서 shape 불일치 발생)  
> 예상 소요 시간: CI 3분 + 쿨다운 60초 × 1,920가지 ≈ 112시간 (약 4.7일)

### ⚠️ 주의사항

- **실험 진행 중** `models/agent/` 디렉토리에 자동 생성된 모델 파일이 지속적으로 push됩니다.
- **`agent_ci.yml` 워크플로우**가 자동으로 트리거됩니다 (기존 `ci.yml`과 별개).
- 실험 결과는 `experiment_log.db`에 누적 저장됩니다.

### Agent 실행 / 중단

```bash
# 시작
sudo systemctl start rngd-agent

# 상태 확인
sudo systemctl status rngd-agent

# 로그 실시간 확인
tail -f ~/rngd-mlir-pipeline/agent_run.log

# 즉시 중단 (방법 1 — systemd)
sudo systemctl stop rngd-agent

# 즉시 중단 (방법 2 — STOP 파일)
touch ~/rngd-mlir-pipeline/AGENT_STOP
```

### 실험 결과 확인

- **GitHub Pages:** https://soadri.github.io/RNGD-Lowering-Pipeline/agent.html
- **로컬 DB:** `experiment_log.db` (SQLite)
- **로그:** `agent_run.log`

---

## 🔍 PyTorch 버전 추적

매주 월요일 자동으로 PyTorch 버전을 확인하고 linalg IR op 패턴 스냅샷을 저장합니다.
PyTorch 버전 업그레이드 시 `e2e_pipeline.py`의 패턴 감지 로직 수정 필요 여부를 자동 감지합니다.

```
snapshots/
  pytorch_2_3_0_dev20240122_cpu.json  ← op 패턴 스냅샷 (버전별)
upstream_versions.json                 ← 버전 이력
```

변경 감지 시 GitHub Issue가 자동 생성됩니다.

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

# PyTorch 버전 추적 수동 실행
python upstream_tracker.py

# 웹 대시보드 (SSH 터널 후 http://localhost:8000)
sudo systemctl restart rngd-dashboard
```

---

## 프로젝트 구조

```
rngd-mlir-pipeline/
├── e2e_pipeline.py          IR 감지·재작성 + 커널/참조데이터 코드 생성 (핵심)
├── deploy_and_test.py       배치 + cargo furiosa-opt test 실행
├── ci_analyze.py            모델 커버리지 분석 스크립트
├── build_pages.py           GitHub Pages HTML 빌드 (index.html)
├── build_agent_pages.py     GitHub Pages HTML 빌드 (agent.html)
├── init_registry.py         변환 규칙 DB 초기화
├── registry.py              DB 읽기 헬퍼
├── dashboard_app.py         FastAPI 백엔드
├── dashboard_index.html     웹 대시보드 프론트엔드
├── agent.py                 자동 실험 에이전트 메인
├── agent_db.py              실험 이력 DB (SQLite)
├── agent_strategy.py        연산 조합 생성 전략
├── agent_model_gen.py       연산 조합 → PyTorch 모델 코드 생성
├── agent_github.py          GitHub API 클라이언트 (CI 상태 polling)
├── upstream_tracker.py      PyTorch 버전 추적 + IR op 패턴 비교
├── models/                  CI 분석 대상 모델 파일들
│   └── agent/               자동 실험 생성 모델 (agent_*.py)
├── snapshots/               PyTorch 버전별 op 패턴 스냅샷
├── upstream_versions.json   PyTorch 버전 이력
├── experiment_log.json      자동 실험 결과 (Pages 빌드용)
└── .github/workflows/
    ├── ci.yml               기존 모델 CI
    ├── agent_ci.yml         자동 실험 모델 CI
    └── upstream_track.yml   PyTorch 버전 추적 (주 1회)

rngd-tcp-kernel-dev/         (별도 레포)
└── src/
    ├── kernel/              생성된 vISA 커널 (17종)
    ├── pilot_e2e_{op}.rs    테스트 파일
    └── reference_data_*     PyTorch 실측 기반 참조값
```

---

## 환경

- **서버:** Ubuntu 24.04
- **Python:** 3.11 + `.venv` (torch-mlir 기반)
- **Rust:** furiosa-opt-std v0.3.0 (FuriosaAI 기능 시뮬레이터)
- **CI:** GitHub Actions (self-hosted runner)