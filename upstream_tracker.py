"""
upstream_tracker.py — PyTorch 버전 추적 + IR op 패턴 비교
주 1회 실행 (GitHub Actions schedule)

IR 전체 저장 대신 op 패턴 목록(JSON)만 저장
→ 파일 크기 수 KB 수준
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

SNAPSHOT_DIR = Path("snapshots")
VERSION_FILE = Path("upstream_versions.json")
MODELS = [
    "models/example_linear.py",
    "models/example_rmsnorm.py",
    "models/llama_rmsnorm.py",
    "models/llama_ffn.py",
    "models/simple_mlp.py",
    "models/llama_ffn_real.py",
    "models/transformer_no_softmax.py",
]

OP_PATTERN = re.compile(r'\b(linalg\.\w+|arith\.\w+|math\.\w+|tensor\.\w+)\b')

EXCLUDE_OPS = {
    "func.func", "func.return", "tensor.empty", "linalg.yield",
    "arith.constant", "arith.index_cast", "arith.index_castui",
    "linalg.fill", "linalg.generic", "arith.truncf", "arith.negf",
    "tensor.collapse_shape", "tensor.expand_shape",
}


def get_pytorch_version() -> str:
    import torch
    return torch.__version__


def extract_ops(ir_text: str) -> dict:
    """IR에서 op 목록과 등장 횟수 추출"""
    ops = OP_PATTERN.findall(ir_text)
    result = {}
    for op in ops:
        if op not in EXCLUDE_OPS:
            result[op] = result.get(op, 0) + 1
    return dict(sorted(result.items()))


def generate_op_snapshot(pytorch_ver: str) -> dict:
    """현재 PyTorch 버전으로 모든 모델의 op 패턴 추출"""
    import torch_mlir
    from ci_analyze import load_model_module

    safe_ver = pytorch_ver.replace("+", "_").replace(".", "_")
    snapshot = {
        "pytorch_version": pytorch_ver,
        "created_at": datetime.now().isoformat(),
        "models": {}
    }

    print(f"op 패턴 스냅샷 생성 (PyTorch {pytorch_ver})")
    for model_path in MODELS:
        name = Path(model_path).stem
        try:
            mod = load_model_module(model_path)
            model = mod.get_model().eval()
            inputs = mod.get_sample_inputs()
            compiled = torch_mlir.compile(
                model, inputs, output_type='linalg-on-tensors'
            )
            ops = extract_ops(str(compiled))
            snapshot["models"][name] = {"status": "ok", "ops": ops}
            print(f"  ✅ {name}: {list(ops.keys())}")
        except Exception as e:
            snapshot["models"][name] = {"status": "error", "error": str(e)[:100]}
            print(f"  ❌ {name}: {e}")

    return snapshot, safe_ver


def compare_snapshots(prev: dict, curr: dict) -> list[dict]:
    """두 스냅샷 op 패턴 비교"""
    changes = []
    all_models = set(list(prev["models"].keys()) + list(curr["models"].keys()))

    for name in sorted(all_models):
        prev_model = prev["models"].get(name, {})
        curr_model = curr["models"].get(name, {})

        if prev_model.get("status") != "ok" or curr_model.get("status") != "ok":
            continue

        prev_ops = set(prev_model.get("ops", {}).keys())
        curr_ops  = set(curr_model.get("ops", {}).keys())

        added   = sorted(curr_ops - prev_ops)
        removed = sorted(prev_ops - curr_ops)

        if added or removed:
            changes.append({
                "model":   name,
                "type":    "op_change",
                "added":   added,
                "removed": removed,
            })

    return changes


def run():
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    pytorch_ver = get_pytorch_version()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"PyTorch 버전: {pytorch_ver}")
    print(f"실행 시각: {now}")

    # 버전 히스토리 로드
    history = json.loads(VERSION_FILE.read_text()) if VERSION_FILE.exists() else []
    prev_ver = history[-1].get("pytorch") if history else None
    version_changed = (prev_ver != pytorch_ver)

    if prev_ver:
        if version_changed:
            print(f"\n⚠️  PyTorch 버전 변경: {prev_ver} → {pytorch_ver}")
        else:
            print(f"\n✅ PyTorch 버전 동일: {pytorch_ver}")
    else:
        print("\n첫 실행 — 베이스라인 스냅샷 생성")

    # op 패턴 스냅샷 생성
    curr_snapshot, safe_ver = generate_op_snapshot(pytorch_ver)

    # 스냅샷 파일 저장 (JSON, 수 KB)
    snap_file = SNAPSHOT_DIR / f"pytorch_{safe_ver}.json"
    snap_file.write_text(
        json.dumps(curr_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n스냅샷 저장: {snap_file} ({snap_file.stat().st_size} bytes)")

    # 이전 스냅샷과 비교
    snap_files = sorted(SNAPSHOT_DIR.glob("pytorch_*.json"))
    ir_changes = []
    if len(snap_files) >= 2:
        prev_snap = json.loads(snap_files[-2].read_text())
        ir_changes = compare_snapshots(prev_snap, curr_snapshot)
        if ir_changes:
            print(f"\n⚠️  op 패턴 변경 감지 ({len(ir_changes)}개 모델):")
            for c in ir_changes:
                print(f"  [{c['model']}]")
                if c["added"]:   print(f"    추가: {c['added']}")
                if c["removed"]: print(f"    제거: {c['removed']}")
            print("\n→ e2e_pipeline.py 패턴 감지 로직 검토 필요")
        else:
            print("\n✅ op 패턴 변경 없음 — e2e_pipeline.py 수정 불필요")
    else:
        print("\n베이스라인 저장 완료 — 다음 실행 시부터 비교")

    # 버전 이력 저장
    record = {
        "pytorch":    pytorch_ver,
        "checked_at": datetime.now().isoformat(),
        "snap_file":  str(snap_file),
        "ir_changes": ir_changes,
    }
    history.append(record)
    VERSION_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"버전 이력 저장: {len(history)}개")

    if version_changed or ir_changes:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    run()
