"""
diagnose_priority.py — "어떤 연산을 구현하면 커버리지가 얼마나 오르나?"
역방향 탐색: 미구현 연산 → 커버리지 향상 기여도 분석

실행: python diagnose_priority.py [--model models/xxx.py] [--all]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# ci_analyze.py의 분석 함수 재사용
sys.path.insert(0, str(Path(__file__).parent))


def analyze_model(model_path: str, db_path: str = "rngd_registry.db") -> dict:
    """단일 모델 커버리지 분석"""
    from ci_analyze import analyze
    import io, contextlib, tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = analyze(model_path, out_dir=Path(tmp), db_path=Path(db_path))
    return result


def get_all_models() -> list[str]:
    """models/ 디렉토리의 기존 모델 목록 (agent 제외)"""
    return sorted([
        str(p) for p in Path("models").glob("*.py")
        if not p.name.startswith("agent_") and not p.name.startswith("__")
    ])


def run(model_paths: list[str], db_path: str = "rngd_registry.db"):
    print("=" * 60)
    print("  RNGD 연산 구현 우선순위 분석")
    print("=" * 60)

    # 모델별 분석
    model_results = {}
    for path in model_paths:
        name = Path(path).stem
        try:
            result = analyze_model(path, db_path)
            model_results[name] = result
            pct = result.get("coverage_pct", 0)
            print(f"  ✅ {name}: {pct:.1f}%")
        except Exception as e:
            print(f"  ❌ {name}: {e}")

    if not model_results:
        print("분석 가능한 모델 없음")
        return

    print()

    # 미지원 op 수집
    # unsupported_op → {model: count}
    unsupported = defaultdict(lambda: defaultdict(int))
    total_ops_per_model = {}

    for name, result in model_results.items():
        ops = result.get("ops", [])
        total = len(ops)
        total_ops_per_model[name] = total
        for op in ops:
            if not op.get("supported"):
                unsupported[op["op"]][name] += op.get("count", 1)

    if not unsupported:
        print("🎉 모든 모델의 모든 연산이 지원됩니다!")
        return

    # 각 미지원 op 구현 시 커버리지 향상 계산
    print("=" * 60)
    print("  미구현 연산 구현 시 커버리지 향상 예측")
    print("=" * 60)

    improvements = []
    for op, model_counts in unsupported.items():
        affected_models = list(model_counts.keys())
        
        # 각 모델에서 이 op가 커버리지에 미치는 영향
        model_gains = {}
        for m, cnt in model_counts.items():
            total = total_ops_per_model.get(m, 1)
            gain = (cnt / total) * 100
            model_gains[m] = gain

        total_gain = sum(model_gains.values()) / len(model_results)
        max_gain   = max(model_gains.values())

        improvements.append({
            "op":              op,
            "affected_models": affected_models,
            "model_gains":     model_gains,
            "avg_gain":        total_gain,
            "max_gain":        max_gain,
        })

    # 평균 커버리지 향상 기준 정렬
    improvements.sort(key=lambda x: x["avg_gain"], reverse=True)

    print(f"\n{'순위':<4} {'미구현 연산':<35} {'영향 모델':<6} {'평균 향상':<10} {'최대 향상'}")
    print("-" * 75)
    for i, imp in enumerate(improvements, 1):
        print(f"  {i:<3} {imp['op']:<35} "
              f"{len(imp['affected_models'])}개 모델   "
              f"+{imp['avg_gain']:.1f}%      "
              f"+{imp['max_gain']:.1f}%")

    print()

    # 상세 분석 (상위 3개)
    print("=" * 60)
    print("  상위 3개 연산 상세 분석")
    print("=" * 60)
    for imp in improvements[:3]:
        print(f"\n🔧 [{imp['op']}] 구현 시:")
        for m, gain in sorted(imp["model_gains"].items(),
                               key=lambda x: x[1], reverse=True):
            curr_pct = model_results[m].get("coverage_pct", 0)
            new_pct  = min(curr_pct + gain, 100.0)
            print(f"   {m:<35} {curr_pct:.1f}% → {new_pct:.1f}%  (+{gain:.1f}%)")

    # 전체 현황 요약
    print()
    print("=" * 60)
    print("  현재 커버리지 요약")
    print("=" * 60)
    for name, result in sorted(model_results.items(),
                                key=lambda x: x[1].get("coverage_pct", 0)):
        pct  = result.get("coverage_pct", 0)
        icon = "🟢" if pct == 100 else "🟡" if pct >= 50 else "🔴"
        print(f"  {icon} {name:<35} {pct:.1f}%")

    avg = sum(r.get("coverage_pct", 0) for r in model_results.values()) / len(model_results)
    print(f"\n  전체 평균 커버리지: {avg:.1f}%")
    print()

    # JSON 출력 (다른 도구에서 활용 가능)
    output = {
        "summary": {
            "total_models":   len(model_results),
            "avg_coverage":   avg,
            "unsupported_ops": len(unsupported),
        },
        "priority": [
            {
                "rank":            i + 1,
                "op":              imp["op"],
                "affected_models": imp["affected_models"],
                "avg_gain":        round(imp["avg_gain"], 2),
                "max_gain":        round(imp["max_gain"], 2),
                "model_gains":     {k: round(v, 2) for k, v in imp["model_gains"].items()},
            }
            for i, imp in enumerate(improvements)
        ],
        "models": {
            name: {"coverage_pct": r.get("coverage_pct", 0)}
            for name, r in model_results.items()
        }
    }
    out_path = Path("ci_out/priority_report.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"  JSON 저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="연산 구현 우선순위 분석")
    parser.add_argument("--model", nargs="+", help="분석할 모델 경로")
    parser.add_argument("--all",   action="store_true", help="models/ 전체 분석")
    parser.add_argument("--db",    default="rngd_registry.db", help="레지스트리 DB 경로")
    args = parser.parse_args()

    if args.all or not args.model:
        models = get_all_models()
    else:
        models = args.model

    print(f"분석 대상: {len(models)}개 모델\n")
    run(models, args.db)
