"""
agent.py — RNGD 자동 실험 에이전트
실행: python agent.py [--max N] [--cooldown 60] [--dry-run]
즉시 중단: sudo systemctl stop rngd-agent  또는  touch AGENT_STOP
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from agent_db import init_db, insert_experiment, update_experiment, \
                     get_tried_combos, set_state, get_state
from agent_github import wait_for_run
from agent_model_gen import save_model, validate_model
from agent_strategy import all_combos, combo_id

PIPELINE_DIR = Path(__file__).parent
MODELS_DIR   = PIPELINE_DIR / "models" / "agent"
STOP_FILE    = PIPELINE_DIR / "AGENT_STOP"
LOG_FILE     = PIPELINE_DIR / "agent_run.log"

# 종료 플래그 (signal 수신 시 True)
_stop_requested = False

def _handle_signal(signum, frame):
    global _stop_requested
    _stop_requested = True
    log(f"⚠️  종료 신호 수신 (signal {signum}) — 현재 단계 완료 후 중단")

signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT,  _handle_signal)

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def should_stop() -> bool:
    """중단 조건 확인"""
    if _stop_requested:
        return True
    if STOP_FILE.exists():
        log("🛑 AGENT_STOP 파일 감지")
        STOP_FILE.unlink(missing_ok=True)
        return True
    return False

def git_commit_push(model_path: Path, combo: str) -> str | None:
    """모델 파일을 git add/commit/push하고 commit SHA 반환"""
    try:
        subprocess.run(["git", "add", str(model_path)],
                       cwd=PIPELINE_DIR, check=True, capture_output=True)
        msg = f"agent: add model {model_path.name} [{combo}]"
        subprocess.run(["git", "commit", "-m", msg],
                       cwd=PIPELINE_DIR, check=True, capture_output=True)
        result = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=PIPELINE_DIR, check=True,
                                capture_output=True, text=True)
        sha = result.stdout.strip()
        subprocess.run(["git", "push"],
                       cwd=PIPELINE_DIR, check=True, capture_output=True)
        return sha
    except subprocess.CalledProcessError as e:
        log(f"  ❌ git 오류: {e.stderr.decode() if e.stderr else ''}")
        return None

def run_single_experiment(ops: list, dry_run: bool) -> str:
    """
    단일 실험 실행. 반환값: "success" | "fail" | "error" | "skip"
    예외가 발생해도 잡아서 "error" 반환 — 절대 죽지 않음
    """
    cid        = combo_id(ops)
    model_file = f"models/agent_{cid}.py"

    try:
        # DB 등록
        if not insert_experiment(cid, ops, model_file):
            return "skip"

        update_experiment(cid, status="running")

        # 1. 모델 유효성 검증
        log("  🔍 모델 유효성 검증 중...")
        valid, err = validate_model(ops)
        if not valid:
            log(f"  ❌ 컴파일 불가: {err[:100]}")
            update_experiment(cid, status="error", error_msg=err[:200],
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return "error"

        if dry_run:
            log("  🧪 [DRY-RUN] 완료 처리")
            update_experiment(cid, status="success", coverage_pct=100.0,
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return "success"

        # 2. 모델 파일 저장
        model_path = save_model(ops, MODELS_DIR)
        log(f"  📝 모델 저장: {model_path.name}")

        # 3. git push
        log("  📤 git push 중...")
        sha = git_commit_push(model_path, cid)
        if not sha:
            update_experiment(cid, status="error", error_msg="git push 실패",
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return "error"

        update_experiment(cid, commit_sha=sha)
        log(f"  ✅ push 완료 (SHA: {sha[:8]})")

        # 4. CI 완료 대기 (중단 신호 체크 포함)
        result = wait_for_run(sha, timeout=600, poll=30)
        ci_status = result["status"]
        run_id    = result.get("run_id", 0)
        finished  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if ci_status == "success":
            log("  ✅ CI PASS")
            update_experiment(cid, status="success", coverage_pct=100.0,
                ci_run_id=run_id, finished_at=finished)
            return "success"
        elif ci_status == "timeout":
            log("  ⏱️  CI 타임아웃")
            update_experiment(cid, status="error", error_msg="CI timeout",
                ci_run_id=run_id, finished_at=finished)
            return "error"
        else:
            log(f"  ❌ CI FAIL: {ci_status}")
            update_experiment(cid, status="fail", ci_run_id=run_id,
                finished_at=finished)
            return "fail"

    except Exception as e:
        # 어떤 예외가 발생해도 잡아서 기록 후 계속
        err_msg = f"{type(e).__name__}: {str(e)[:150]}"
        log(f"  💥 예외 발생 (계속 진행): {err_msg}")
        try:
            update_experiment(cid, status="error", error_msg=err_msg,
                finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass
        return "error"


def run_agent(max_experiments: int, cooldown: int, dry_run: bool):
    init_db()
    log(f"🤖 RNGD Agent 시작 (max={max_experiments}, cooldown={cooldown}s, dry_run={dry_run})")

    combos    = all_combos()
    tried     = get_tried_combos()
    remaining = [c for c in combos if combo_id(c) not in tried]

    log(f"전체 조합: {len(combos)}개, 시도됨: {len(tried)}개, 남은: {len(remaining)}개")

    stats = {"success": 0, "fail": 0, "error": 0, "skip": 0}
    count = 0

    for ops in remaining:
        if should_stop():
            log("🛑 중단 요청 — 종료합니다")
            break

        if count >= max_experiments:
            log(f"🏁 최대 실험 수({max_experiments}) 도달 — 종료")
            break

        cid = combo_id(ops)
        count += 1

        log(f"\n{'='*60}")
        log(f"실험 {count}: {' → '.join(ops)}")

        result = run_single_experiment(ops, dry_run)
        stats[result] = stats.get(result, 0) + 1

        set_state("last_run", datetime.now().isoformat())
        set_state("stats", json.dumps(stats))

        if should_stop():
            log("🛑 중단 요청 — 종료합니다")
            break

        if result != "skip":
            log(f"  ⏸️  쿨다운 {cooldown}초 대기...")
            # 쿨다운 중에도 중단 신호 체크 (10초마다)
            for _ in range(cooldown // 10):
                if should_stop():
                    break
                time.sleep(10)

    log(f"\n🏁 Agent 완료 — 총 {count}개 실험")
    log(f"   성공: {stats.get('success',0)}, 실패: {stats.get('fail',0)}, "
        f"오류: {stats.get('error',0)}, 건너뜀: {stats.get('skip',0)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RNGD 자동 실험 에이전트")
    parser.add_argument("--max",      type=int, default=9999, help="최대 실험 수")
    parser.add_argument("--cooldown", type=int, default=60,   help="쿨다운 초 (기본: 60)")
    parser.add_argument("--dry-run",  action="store_true",    help="실제 push 없이 동작 확인")
    args = parser.parse_args()

    run_agent(args.max, args.cooldown, args.dry_run)
