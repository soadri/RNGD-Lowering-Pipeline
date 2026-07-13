"""
KETI 내부 팀원용 파이프라인 대시보드 — FastAPI 백엔드 (registry.py 연동 버전).

변경 사항:
- SUPPORTED_OPS, OP_GROUPS, OP_METADATA, ROADMAP을 rngd_registry.db에서 읽음
- 하드코딩 딕셔너리 제거 (METHOD_NOTE, FAMILY_NOTE, LOWERING_NOTE, CAVEATS는 변경 빈도가
  낮아 그대로 유지 — 필요 시 추후 DB로 이관)
- /api/registry, /api/registry/search, /api/registry/{op_name},
  /api/constraints 엔드포인트 추가
"""

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import registry  # rngd_registry.db 읽기 전용 헬퍼

PIPELINE_DIR = Path.home() / "rngd-mlir-pipeline"
KERNEL_PROJECT_DIR = Path.home() / "rngd-tcp-kernel-dev"
HISTORY_FILE = PIPELINE_DIR / "run_history.json"
HISTORY_MAX_PER_OP = 20

# 변경 빈도가 낮아 계속 코드로 유지하는 상수
METHOD_NOTE = (
    "정합성 검증(correctness check): PyTorch가 독립적으로 계산한 값과 RNGD 시뮬레이터가 "
    "별도로 계산한 값을 비교한다. 자체 재계산값과 비교하는 순환 검증이 아니다."
)

CAVEATS = [
    "rngd.elementwise / rngd.batch_gemm / rngd.gemm은 FuriosaAI의 공식 dialect가 아니라 "
    "KETI 내부 placeholder다. attribute 스키마는 미확정이며 추후 전면 교체될 수 있다.",
    "Family B(batch_gemm/gemm)는 배치 축 V≤256 가정(하드웨어 Slice 제약 대응)에서만 검증됐다. "
    "V>256인 경우의 타일링 전략은 아직 구현/검증되지 않았다.",
    "Family A는 4×4 텐서 실측값만 실제 검증 대상 (커널 자체는 2048 고정 크기 버퍼).",
    "Family B 테스트 shape은 고정값(batch_gemm: 32×32×32×8, gemm: 32×32×8)이며, "
    "임의 크기로 일반화되지 않는다.",
    "Family B의 값 비교는 원본 f32 결과가 아니라 'bf16으로 양자화한 뒤 계산한 현실적 정답'과 "
    "비교한다 — 정밀도 손실은 의도된 것이며 버그가 아니다.",
    "여기서 '시뮬레이터'는 vISA 명령어의 계산 의미론(실제 값)만 재현하는 기능적 "
    "인터프리터다. 사이클 정확도나 실제 RNGD 실리콘과의 동작 일치는 보증하지 않는다 "
    "— 지금까지의 '정합성 검증'은 전부 이 기능 시뮬레이터 기준이다.",
]

FAMILY_NOTE = (
    "Family A (Elementwise) — reduction 없는 원소별 연산. linalg.generic + iterator_types 전부 "
    "parallel. 예: add, sub, mul.\n"
    "Family B (Contraction) — reduction(축소)을 포함하는 연산. named op(linalg.matmul, "
    "linalg.batch_matmul) 기반. 예: 행렬곱. Contraction Engine(TRF, contract_outer 등) 사용."
)

LOWERING_NOTE = {
    "elementwise": (
        "Lowering: linalg IR(고수준) → RNGD IR(타겟 전용) → Rust 커널(vISA) 순으로 낮춰가는 컴파일 단계.\n"
        "이 케이스에서는 linalg.generic(단일 산술 연산)이 rngd.elementwise 하나로 치환된다. "
        "tensor.empty, linalg.yield 등 보일러플레이트는 제거되고 연산 종류(add/sub/mul)만 attribute로 남는다."
    ),
    "contraction": (
        "Lowering: linalg IR(고수준) → RNGD IR(타겟 전용) → Rust 커널(vISA) 순으로 낮춰가는 컴파일 단계.\n"
        "이 케이스에서는 linalg.matmul 계열(+ 선행 linalg.fill)이 rngd.gemm/rngd.batch_gemm 하나로 치환된다. "
        "출력 0-초기화(linalg.fill)는 흡수되어 사라진다."
    ),
}

app = FastAPI()


class RunRequest(BaseModel):
    op: str


# ─────────────────────────────────────────────────────────────────────
# 기존 엔드포인트 (registry.py로 교체)
# ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "dashboard_index.html"
    return html_path.read_text(encoding="utf-8")


@app.get("/api/ops")
def list_ops():
    return {
        "ops":          registry.load_supported_ops(),
        "meta":         registry.load_op_metadata(),
        "groups":       registry.load_op_groups(),
        "roadmap":      registry.load_roadmap(),
        "method_note":  METHOD_NOTE,
        "lowering_note": LOWERING_NOTE,
        "family_note":  FAMILY_NOTE,
        "caveats":      CAVEATS,
    }


# ─────────────────────────────────────────────────────────────────────
# 레지스트리 조회/검색 엔드포인트 (신규)
# ─────────────────────────────────────────────────────────────────────

@app.get("/api/registry")
def registry_list(
        family: Optional[str] = Query(None, description="elementwise | contraction"),
        dtype: Optional[str] = Query(None, description="f32 | bf16"),
        status: Optional[str] = Query(None, description="done | wip | broken"),
        experimental: Optional[bool] = Query(None),
):
    """
    전체 변환 규칙 목록. 쿼리 파라미터로 필터 가능.
    예: /api/registry?family=elementwise&dtype=f32
    """
    import sqlite3
    con = sqlite3.connect(registry.DB_PATH)
    con.row_factory = sqlite3.Row

    conditions, params = [], []
    if family:
        conditions.append("o.family = ?"); params.append(family)
    if dtype:
        conditions.append("o.dtype = ?"); params.append(dtype)
    if status:
        conditions.append("o.status = ?"); params.append(status)
    if experimental is not None:
        conditions.append("o.experimental = ?"); params.append(1 if experimental else 0)

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    ops = con.execute(
        f"""SELECT o.*, GROUP_CONCAT(c.description, '||') AS constraint_descs
            FROM ops o
            LEFT JOIN op_constraint_link l ON l.op_name = o.op_name
            LEFT JOIN hardware_constraints c ON c.id = l.constraint_id
            {where}
            GROUP BY o.op_name
            ORDER BY o.family, o.op_name""",
        params,
    ).fetchall()

    ev_rows = con.execute(
        "SELECT op_name, evidence_type, claim, source FROM evidence"
    ).fetchall()
    con.close()

    ev_map: dict[str, list] = {}
    for ev in ev_rows:
        ev_map.setdefault(ev["op_name"], []).append({
            "type": ev["evidence_type"], "claim": ev["claim"], "source": ev["source"]
        })

    result = []
    for row in ops:
        d = dict(row)
        d["experimental"] = bool(d["experimental"])
        d["constraints"] = [s for s in (d.pop("constraint_descs") or "").split("||") if s]
        d["evidence"] = ev_map.get(d["op_name"], [])
        result.append(d)

    return {"total": len(result), "ops": result}


@app.get("/api/registry/search")
def registry_search(q: str = Query(..., min_length=1, description="검색어")):
    """
    op_name, source_op, target_op, detail, evidence.claim을 대상으로 전문 검색.
    예: /api/registry/search?q=rsqrt
        /api/registry/search?q=Mode10
        /api/registry/search?q=bf16
    """
    import sqlite3
    con = sqlite3.connect(registry.DB_PATH)
    con.row_factory = sqlite3.Row
    like = f"%{q}%"

    # ops 테이블 검색
    op_rows = con.execute(
        """SELECT DISTINCT o.op_name, o.family, o.source_op, o.target_op,
                           o.dtype, o.tolerance, o.experimental, o.status, o.detail
           FROM ops o
           WHERE o.op_name LIKE ?
              OR o.source_op LIKE ?
              OR o.target_op LIKE ?
              OR o.detail LIKE ?
              OR o.compares LIKE ?
           ORDER BY o.family, o.op_name""",
        [like] * 5,
        ).fetchall()

    # evidence 검색 — claim/source에서 히트한 것도 포함
    ev_hits = con.execute(
        """SELECT DISTINCT e.op_name, o.family, o.source_op, o.target_op,
                           o.dtype, o.tolerance, o.experimental, o.status, o.detail
           FROM evidence e
                    JOIN ops o ON o.op_name = e.op_name
           WHERE e.claim LIKE ? OR e.source LIKE ?""",
        [like, like],
    ).fetchall()

    # 결과 합치기 (중복 제거)
    seen = set()
    merged = []
    for row in list(op_rows) + list(ev_hits):
        if row["op_name"] not in seen:
            seen.add(row["op_name"])
            merged.append(dict(row))

    # 각 op의 evidence 매칭 여부 표시
    all_evidence = con.execute(
        "SELECT op_name, evidence_type, claim, source FROM evidence"
    ).fetchall()
    con.close()

    ev_map: dict[str, list] = {}
    for ev in all_evidence:
        ev_map.setdefault(ev["op_name"], []).append({
            "type": ev["evidence_type"],
            "claim": ev["claim"],
            "source": ev["source"],
            "matched": q.lower() in (ev["claim"] + ev["source"]).lower(),
        })

    for item in merged:
        item["experimental"] = bool(item["experimental"])
        item["evidence"] = ev_map.get(item["op_name"], [])

    return {"query": q, "total": len(merged), "ops": merged}


@app.get("/api/registry/{op_name}")
def registry_detail(op_name: str):
    """특정 연산의 상세 정보 — 연결된 하드웨어 제약까지 포함."""
    import sqlite3
    con = sqlite3.connect(registry.DB_PATH)
    con.row_factory = sqlite3.Row

    row = con.execute("SELECT * FROM ops WHERE op_name = ?", (op_name,)).fetchone()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"op '{op_name}' not found in registry")

    evidence = con.execute(
        "SELECT evidence_type, claim, source FROM evidence WHERE op_name = ? ORDER BY id",
        (op_name,),
    ).fetchall()

    constraints = con.execute(
        """SELECT c.id, c.description, c.discovery_source
           FROM op_constraint_link l
                    JOIN hardware_constraints c ON c.id = l.constraint_id
           WHERE l.op_name = ?
           ORDER BY c.id""",
        (op_name,),
    ).fetchall()
    con.close()

    d = dict(row)
    d["experimental"] = bool(d["experimental"])
    d["evidence"] = [dict(e) for e in evidence]
    d["constraints"] = [dict(c) for c in constraints]
    return d


@app.get("/api/constraints")
def list_constraints():
    """하드웨어 제약 전체 목록 — 각 제약에 영향받는 연산 목록 포함."""
    import sqlite3
    con = sqlite3.connect(registry.DB_PATH)
    con.row_factory = sqlite3.Row

    constraints = con.execute(
        "SELECT id, description, discovery_source FROM hardware_constraints ORDER BY id"
    ).fetchall()
    links = con.execute(
        "SELECT constraint_id, op_name FROM op_constraint_link ORDER BY constraint_id"
    ).fetchall()
    con.close()

    link_map: dict[int, list] = {}
    for lnk in links:
        link_map.setdefault(lnk["constraint_id"], []).append(lnk["op_name"])

    result = []
    for c in constraints:
        d = dict(c)
        d["affected_ops"] = link_map.get(d["id"], [])
        result.append(d)
    return {"total": len(result), "constraints": result}


# ─────────────────────────────────────────────────────────────────────
# 유틸리티
# ─────────────────────────────────────────────────────────────────────

def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _load_history() -> dict:
    if not HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _append_history(op: str, ok: bool, duration: float | None) -> dict:
    history = _load_history()
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "ok": ok,
        "duration": duration,
    }
    op_history = history.setdefault(op, [])
    op_history.append(entry)
    history[op] = op_history[-HISTORY_MAX_PER_OP:]
    try:
        HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass
    return entry


_DURATION_RE = re.compile(r"finished in ([\d.]+)s")


def _parse_duration(log_text: str):
    m = _DURATION_RE.search(log_text)
    return float(m.group(1)) if m else None


@app.get("/api/history")
def get_history():
    return _load_history()


def _read_artifact(prefix: str, suffix: str) -> str:
    path = PIPELINE_DIR / "generated" / f"{prefix}_{suffix}"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _stream_run(op: str):
    supported = registry.load_supported_ops()
    if op not in supported:
        yield _sse({"type": "done", "ok": False, "error": f"지원하지 않는 연산: {op}"})
        return

    yield _sse({"type": "stage", "step": 1, "total": 3, "text": "코드 생성 중 (e2e_pipeline.py)"})
    proc = subprocess.Popen(
        ["python", "e2e_pipeline.py", op], cwd=str(PIPELINE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        yield _sse({"type": "log", "text": line.rstrip()})
    proc.wait()
    if proc.returncode != 0:
        yield _sse({"type": "done", "ok": False, "error": "코드 생성 단계 실패 (위 로그 참고)"})
        return

    prefix = f"e2e_{op}"
    yield _sse({
        "type": "artifacts",
        "ir_before": _read_artifact(prefix, "ir_before_marked.mlir"),
        "ir_after":  _read_artifact(prefix, "ir_after_marked.mlir"),
        "ir_diff":   _read_artifact(prefix, "ir_diff.txt"),
        "kernel_rs": _read_artifact(prefix, "kernel.rs"),
    })

    yield _sse({"type": "stage", "step": 2, "total": 3, "text": "배치 및 빌드/테스트 실행 중 (수 초~수 분 소요)"})
    proc2 = subprocess.Popen(
        ["python", "deploy_and_test.py", op], cwd=str(PIPELINE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc2.stdout:
        yield _sse({"type": "log", "text": line.rstrip()})
    proc2.wait()

    yield _sse({"type": "stage", "step": 3, "total": 3, "text": "결과 정리 중"})
    log_path = KERNEL_PROJECT_DIR / f"test_log_{op}.txt"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    log_lines = log_text.splitlines()

    compare_start = next(
        (i for i, l in enumerate(log_lines)
         if l.strip().startswith("===") and l.strip() not in ("=== STDOUT ===", "=== STDERR ===")),
        None,
    )
    if compare_start is not None:
        stderr_marker = next(
            (i for i in range(compare_start, len(log_lines)) if log_lines[i].strip() == "=== STDERR ==="),
            len(log_lines),
        )
        preview_end = min(compare_start + 10, stderr_marker)
        preview = log_lines[compare_start:preview_end]
    else:
        preview = []

    passed = any("test result: ok" in l for l in log_lines)
    duration = _parse_duration(log_text)
    history_entry = _append_history(op, passed, duration)

    yield _sse({
        "type": "done", "ok": passed, "op": op,
        "preview": preview,
        "meta": registry.load_op_metadata().get(op),
        "history_entry": history_entry,
    })


@app.post("/api/run_stream")
def run_stream(req: RunRequest):
    return StreamingResponse(_stream_run(req.op), media_type="text/event-stream")


def _stream_run_all():
    supported = registry.load_supported_ops()
    yield _sse({"type": "all_stage", "text": f"코드 생성 중 (연산 {len(supported)}개 전부)"})
    proc = subprocess.Popen(
        ["python", "e2e_pipeline.py"], cwd=str(PIPELINE_DIR),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    for line in proc.stdout:
        yield _sse({"type": "log", "text": line.rstrip()})
    proc.wait()
    if proc.returncode != 0:
        yield _sse({"type": "all_done", "ok": False, "error": "코드 생성 단계 실패 (위 로그 참고)"})
        return

    results = {}
    for op in supported:
        yield _sse({"type": "op_start", "op": op})
        proc2 = subprocess.Popen(
            ["python", "deploy_and_test.py", op], cwd=str(PIPELINE_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc2.stdout:
            yield _sse({"type": "log", "text": line.rstrip()})
        proc2.wait()

        log_path = KERNEL_PROJECT_DIR / f"test_log_{op}.txt"
        log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
        passed = "test result: ok" in log_text
        results[op] = passed
        duration = _parse_duration(log_text)
        history_entry = _append_history(op, passed, duration)
        yield _sse({"type": "op_done", "op": op, "ok": passed, "history_entry": history_entry})

    yield _sse({"type": "all_done", "ok": all(results.values()), "results": results})


@app.post("/api/run_all_stream")
def run_all_stream():
    return StreamingResponse(_stream_run_all(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
