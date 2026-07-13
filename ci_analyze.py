"""
ci_analyze.py
=============
PyTorch 모델 파일을 받아서:
  1. linalg IR로 컴파일
  2. 등장한 op 목록 추출
  3. rngd_registry.db의 지원 연산과 대조해 커버리지 계산
  4. 결과를 JSON + Markdown으로 출력

사용법:
    python ci_analyze.py --model path/to/model.py [--out-dir ./ci_out]

모델 파일 규격:
    - `get_model()` 함수와 `get_sample_inputs()` 함수를 export해야 한다.
    - get_model() → torch.nn.Module
    - get_sample_inputs() → tuple of tensors

예시 모델 파일:
    import torch
    def get_model():
        return torch.nn.Linear(64, 32)
    def get_sample_inputs():
        return (torch.randn(1, 64),)
"""

import argparse
import importlib.util
import json
import re
import sqlite3
import sys
from pathlib import Path
from datetime import datetime, timezone

import torch
import torch_mlir

# rngd_registry.db 경로 (CI에서는 --db로 override 가능)
DEFAULT_DB = Path(__file__).parent / "rngd_registry.db"


def load_model_module(model_path: str):
    spec = importlib.util.spec_from_file_location("user_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def extract_ops_from_ir(ir_text: str) -> list[str]:
    """linalg IR 텍스트에서 op 이름 목록을 추출한다."""
    pattern = re.compile(r'\b(linalg\.\w+|arith\.\w+|math\.\w+|tensor\.\w+|func\.\w+)\b')
    found = pattern.findall(ir_text)
    # 보일러플레이트 제외
    exclude = {
        "func.func", "func.return",
        "tensor.empty", "linalg.yield",
        "arith.constant", "arith.index_cast", "arith.index_castui",
        # linalg.fill은 DCE 대상, linalg.generic은 내부 op(arith.*, math.*)으로 분해됨
        "linalg.fill", "linalg.generic",
    }
    return sorted(set(found) - exclude)


def load_supported_ops(db_path: Path) -> dict[str, dict]:
    """
    DB에서 지원 연산 정보를 읽어 반환.
    {source_op_inner: {op_name, family, status}} 형태.
    예: {"arith.addf": {"op_name": "add", "family": "elementwise", ...}}
    """
    if not db_path.exists():
        print(f"[경고] DB 없음: {db_path} — init_registry.py를 먼저 실행하세요", file=sys.stderr)
        return {}
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT op_name, family, source_op, target_op, status FROM ops").fetchall()
    con.close()

    result = {}
    for row in rows:
        src = row["source_op"]
        # "linalg.generic(arith.addf)" → "arith.addf"
        # "linalg.batch_matmul" → "linalg.batch_matmul"
        if "(" in src:
            inner = src.split("(")[-1].rstrip(")")
            # "math.powf, 지수=2.0" 같은 경우 쉼표 앞부분만 사용
            inner = inner.split(",")[0].strip()
        else:
            inner = src.split(",")[0].strip()
        result[inner] = dict(row)
    return result


def analyze(model_path: str, db_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 모델 로드
    mod = load_model_module(model_path)
    model = mod.get_model().eval()
    inputs = mod.get_sample_inputs()

    print(f"[1/3] 모델 컴파일 중: {model_path}")
    try:
        compiled = torch_mlir.compile(model, inputs, output_type="linalg-on-tensors")
        ir_text = str(compiled)
    except Exception as e:
        print(f"[오류] torch_mlir 컴파일 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # IR 저장
    ir_path = out_dir / "model_ir.mlir"
    ir_path.write_text(ir_text, encoding="utf-8")
    print(f"  → linalg IR 저장: {ir_path}")

    # op 분석
    print("[2/3] op 분석 중")
    found_ops = extract_ops_from_ir(ir_text)
    supported_map = load_supported_ops(db_path)

    # 각 op에 대해 지원 여부 판단
    op_results = []
    for op in found_ops:
        if op in supported_map:
            info = supported_map[op]
            op_results.append({
                "op": op,
                "supported": True,
                "rngd_op": info["target_op"],
                "family": info["family"],
                "status": info["status"],
            })
        else:
            op_results.append({
                "op": op,
                "supported": False,
                "rngd_op": None,
                "family": None,
                "status": "unsupported",
            })

    supported = [r for r in op_results if r["supported"]]
    unsupported = [r for r in op_results if not r["supported"]]
    total = len(op_results)
    coverage = round(len(supported) / total * 100, 1) if total > 0 else 0.0

    result = {
        "model": model_path,
        "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_ops": total,
        "supported_count": len(supported),
        "unsupported_count": len(unsupported),
        "coverage_pct": coverage,
        "ops": op_results,
    }

    # JSON 저장
    json_path = out_dir / "coverage.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # Markdown 리포트 생성
    print("[3/3] Markdown 리포트 생성 중")
    md = _build_markdown(result)
    md_path = out_dir / "coverage_report.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"  → 리포트 저장: {md_path}")

    # 터미널 요약 출력
    print(f"\n{'='*60}")
    print(f"  커버리지: {coverage}%  ({len(supported)}/{total}개 op 지원)")
    if unsupported:
        print(f"  미지원 op ({len(unsupported)}개):")
        for r in unsupported:
            print(f"    · {r['op']}")
    print(f"{'='*60}\n")

    return result


def _build_markdown(result: dict) -> str:
    cov = result["coverage_pct"]
    supported = [r for r in result["ops"] if r["supported"]]
    unsupported = [r for r in result["ops"] if not r["supported"]]

    # 커버리지에 따라 이모지/색상 결정
    if cov >= 80:
        badge = "🟢"
    elif cov >= 50:
        badge = "🟡"
    else:
        badge = "🔴"

    lines = [
        f"# RNGD Lowering 커버리지 리포트 {badge}",
        f"",
        f"| 항목 | 값 |",
        f"|---|---|",
        f"| 모델 | `{result['model']}` |",
        f"| 분석 시각 | {result['analyzed_at']} |",
        f"| 전체 op 수 | {result['total_ops']} |",
        f"| 지원 op 수 | {result['supported_count']} |",
        f"| 미지원 op 수 | {result['unsupported_count']} |",
        f"| **커버리지** | **{cov}%** |",
        f"",
    ]

    if supported:
        lines += [
            f"## ✅ 지원되는 op ({len(supported)}개)",
            f"",
            f"| linalg op | RNGD op | Family |",
            f"|---|---|---|",
        ]
        for r in supported:
            lines.append(f"| `{r['op']}` | `{r['rngd_op']}` | {r['family']} |")
        lines.append("")

    if unsupported:
        lines += [
            f"## ❌ 미지원 op ({len(unsupported)}개)",
            f"",
            f"> 아래 op들은 아직 변환 규칙이 없습니다.",
            f"",
        ]
        for r in unsupported:
            lines.append(f"- `{r['op']}`")
        lines.append("")

    lines += [
        "---",
        "*이 리포트는 RNGD CI 파이프라인에 의해 자동 생성됩니다.*",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="PyTorch 모델 파일 경로")
    parser.add_argument("--out-dir", default="ci_out", help="결과 저장 디렉토리")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="rngd_registry.db 경로")
    args = parser.parse_args()

    analyze(
        model_path=args.model,
        db_path=Path(args.db),
        out_dir=Path(args.out_dir),
    )
    