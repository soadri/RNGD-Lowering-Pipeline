"""
upstream_tracker.py — PyTorch 버전 추적 + IR 패턴 스냅샷 비교
주 1회 실행 (GitHub Actions schedule)

목적:
  PyTorch 버전이 올라갔을 때 linalg IR 패턴이 바뀌는지 감지
  → e2e_pipeline.py 패턴 감지 로직 수정 필요 여부 판단
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path

SNAPSHOT_DIR = Path("snapshots")
VERSION_FILE  = Path("upstream_versions.json")
MODELS = [
    "models/example_linear.py",
    "models/example_rmsnorm.py",
    "models/llama_rmsnorm.py",
    "models/llama_ffn.py",
    "models/simple_mlp.py",
    "models/llama_ffn_real.py",
    "models/transformer_no_softmax.py",
]


def get_pytorch_version() -> str:
    import torch
    return torch.__version__


def generate_ir_snapshot(pytorch_ver: str) -> Path:
    """현재 PyTorch 버전으로 모든 모델의 linalg IR 생성"""
    import torch_mlir
    from ci_analyze import load_model_module

    # 버전 문자열에서 파일명에 쓸 수 없는 문자 제거
    safe_ver = pytorch_ver.replace("+", "_").replace(".", "_")
    snap_dir = SNAPSHOT_DIR / f"pytorch_{safe_ver}"
    snap_dir.mkdir(parents=True, exist_ok=True)

    print(f"IR 스냅샷 생성: {snap_dir}")
    for model_path in MODELS:
        try:
            mod    = load_model_module(model_path)
            model  = mod.get_model().eval()
            inputs = mod.get_sample_inputs()
            compiled = torch_mlir.compile(
                model, inputs, output_type='linalg-on-tensors'
            )
            name = Path(model_path).stem
            (snap_dir / f"{name}.mlir").write_text(str(compiled), encoding="utf-8")
            print(f"  ✅ {name}.mlir")
        except Exception as e:
            print(f"  ❌ {model_path}: {e}")

    return snap_dir


def compare_snapshots(prev_dir: Path, curr_dir: Path) -> list[dict]:
    """두 스냅샷 비교 — op 패턴 변경 감지"""
    changes = []
    for curr_file in sorted(curr_dir.glob("*.mlir")):
        name      = curr_file.stem
        prev_file = prev_dir / curr_file.name

        if not prev_file.exists():
            changes.append({"model": name, "type": "new"})
            continue

        prev_ops = set(re.findall(
            r'\b(linalg\.\w+|arith\.\w+|math\.\w+)\b',
            prev_file.read_text()
        ))
        curr_ops = set(re.findall(
            r'\b(linalg\.\w+|arith\.\w+|math\.\w+)\b',
            curr_file.read_text()
        ))

        added   = sorted(curr_ops - prev_ops)
        removed = sorted(prev_ops - curr_ops)

        if added or removed:
            changes.append({
                "model":   name,
                "type":    "op_change",
                "added":   added,
                "removed": removed,
            })
        elif curr_file.read_text() != prev_file.read_text():
            changes.append({
                "model": name,
                "type":  "minor_change",
            })

    return changes


def run():
    SNAPSHOT_DIR.mkdir(exist_ok=True)

    pytorch_ver = get_pytorch_version()
    print(f"PyTorch 버전: {pytorch_ver}")
    print(f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 버전 히스토리 로드
    history = json.loads(VERSION_FILE.read_text()) if VERSION_FILE.exists() else []

    # 이전 버전과 비교
    prev_ver = history[-1].get("pytorch") if history else None
    version_changed = (prev_ver != pytorch_ver)

    if prev_ver:
        if version_changed:
            print(f"\n⚠️  PyTorch 버전 변경 감지: {prev_ver} → {pytorch_ver}")
        else:
            print(f"\n✅ PyTorch 버전 동일: {pytorch_ver}")
    else:
        print("\n첫 실행 — 베이스라인 스냅샷 생성")

    # IR 스냅샷 생성
    curr_snap = generate_ir_snapshot(pytorch_ver)

    # 이전 스냅샷과 비교
    snap_dirs = sorted(SNAPSHOT_DIR.glob("pytorch_*"))
    ir_changes = []
    if len(snap_dirs) >= 2:
        prev_snap  = snap_dirs[-2]
        ir_changes = compare_snapshots(prev_snap, curr_snap)
        if ir_changes:
            print(f"\n⚠️  IR 패턴 변경 감지 ({len(ir_changes)}개 모델):")
            for c in ir_changes:
                if c["type"] == "op_change":
                    print(f"  [{c['model']}]")
                    if c["added"]:   print(f"    추가된 op: {c['added']}")
                    if c["removed"]: print(f"    제거된 op: {c['removed']}")
                else:
                    print(f"  [{c['model']}] 구조 변경 (op 종류 동일)")
        else:
            print("\n✅ IR 패턴 변경 없음 — e2e_pipeline.py 수정 불필요")

    # 기록 저장
    record = {
        "pytorch":    pytorch_ver,
        "checked_at": datetime.now().isoformat(),
        "ir_changes": ir_changes,
    }
    history.append(record)
    VERSION_FILE.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"\n버전 이력 저장: {len(history)}개")

    # 변경 감지 시 exit 1 (CI 알림용)
    if version_changed or ir_changes:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    run()
