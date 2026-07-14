"""
build_agent_pages.py — Agent 실험 결과 → GitHub Pages agent.html 빌드
기존 build_pages.py가 생성한 pages_out/index.html과 병존
"""
import json
import pathlib
import sys
from datetime import datetime

def build(artifacts_dir: str, combo_id: str, commit_sha: str):
    artifacts = pathlib.Path(artifacts_dir)
    out_dir   = pathlib.Path("pages_out")
    out_dir.mkdir(exist_ok=True)

    # 기존 index.html 유지 (build_pages.py가 이미 생성했을 것)
    # agent_log.json 읽기
    log_path = pathlib.Path("agent_log.json")
    experiments = []
    if log_path.exists():
        experiments = json.loads(log_path.read_text(encoding="utf-8"))

    # 이번 실험 커버리지 읽기
    coverage = {}
    for p in artifacts.glob("**/coverage.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        coverage = data
        break

    # 통계
    total   = len(experiments)
    success = sum(1 for e in experiments if e["status"] == "success")
    fail    = sum(1 for e in experiments if e["status"] == "fail")
    error   = sum(1 for e in experiments if e["status"] == "error")
    running = sum(1 for e in experiments if e["status"] == "running")

    # 최근 실험 50개
    recent = sorted(experiments, key=lambda e: e.get("id", 0), reverse=True)[:50]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = ""
    for e in recent:
        status = e["status"]
        icon = {"success": "✅", "fail": "❌", "error": "⚠️", "running": "⏳", "pending": "🔵"}.get(status, "❓")
        ops_str = " → ".join(json.loads(e["ops"])) if isinstance(e["ops"], str) else " → ".join(e["ops"])
        sha = e.get("commit_sha", "")[:8]
        finished = e.get("finished_at", "-")
        rows += f"""
        <tr>
          <td>{e.get('id','')}</td>
          <td><code>{ops_str}</code></td>
          <td>{icon} {status}</td>
          <td>{e.get('coverage_pct', 0):.0f}%</td>
          <td><code>{sha}</code></td>
          <td>{finished}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="60">
  <title>RNGD Agent 실험 현황</title>
  <style>
    body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }}
    h1 {{ color: #1a1a2e; }}
    .nav {{ margin-bottom: 20px; }}
    .nav a {{ margin-right: 15px; color: #0366d6; text-decoration: none; font-weight: bold; }}
    .stats {{ display: flex; gap: 15px; margin: 20px 0; flex-wrap: wrap; }}
    .stat {{ background: white; border-radius: 8px; padding: 15px 25px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); min-width: 100px; }}
    .stat .num {{ font-size: 2em; font-weight: bold; }}
    .stat .label {{ color: #666; font-size: 0.85em; }}
    .success {{ color: #28a745; }}
    .fail    {{ color: #dc3545; }}
    .error   {{ color: #fd7e14; }}
    .running {{ color: #007bff; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    th {{ background: #1a1a2e; color: white; padding: 10px 15px; text-align: left; }}
    td {{ padding: 8px 15px; border-bottom: 1px solid #eee; }}
    tr:hover {{ background: #f8f9fa; }}
    code {{ background: #f1f1f1; padding: 2px 6px; border-radius: 3px; font-size: 0.85em; }}
    .progress {{ background: #e9ecef; border-radius: 10px; height: 20px; margin: 10px 0; }}
    .progress-bar {{ background: #28a745; border-radius: 10px; height: 20px; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8em; font-weight: bold; transition: width 0.3s; }}
    .stop-btn {{ background: #dc3545; color: white; border: none; padding: 10px 25px; border-radius: 6px; cursor: pointer; font-size: 1em; margin: 10px 0; }}
    .stop-btn:hover {{ background: #c82333; }}
    .updated {{ color: #666; font-size: 0.85em; margin-top: 10px; }}
  </style>
</head>
<body>
  <div class="nav">
    <a href="index.html">← 메인 대시보드</a>
    <a href="agent.html">🤖 Agent 실험 현황</a>
  </div>

  <h1>🤖 RNGD Agent 자동 실험 현황</h1>

  <div class="stats">
    <div class="stat"><div class="num">{total}</div><div class="label">전체 실험</div></div>
    <div class="stat"><div class="num success">{success}</div><div class="label">✅ 성공</div></div>
    <div class="stat"><div class="num fail">{fail}</div><div class="label">❌ 실패</div></div>
    <div class="stat"><div class="num error">{error}</div><div class="label">⚠️ 오류</div></div>
    <div class="stat"><div class="num running">{running}</div><div class="label">⏳ 진행중</div></div>
    <div class="stat"><div class="num">{2544}</div><div class="label">전체 목표</div></div>
  </div>

  <div class="progress">
    <div class="progress-bar" style="width: {min(total/2544*100, 100):.1f}%">
      {total/2544*100:.1f}%
    </div>
  </div>

  <p>
    현재 실험 중인 조합: <strong>{combo_id.replace('-', ' → ')}</strong><br>
    Commit: <code>{commit_sha[:8]}</code>
  </p>

  <h2>최근 실험 결과 (최신 50개)</h2>
  <table>
    <thead>
      <tr><th>#</th><th>연산 조합</th><th>상태</th><th>커버리지</th><th>Commit</th><th>완료 시각</th></tr>
    </thead>
    <tbody>
      {rows if rows else '<tr><td colspan="6" style="text-align:center">아직 실험 없음</td></tr>'}
    </tbody>
  </table>

  <p class="updated">자동 새로고침: 60초마다 | 마지막 빌드: {now}</p>
</body>
</html>"""

    (out_dir / "agent.html").write_text(html, encoding="utf-8")
    print(f"agent.html 빌드 완료 → {out_dir / 'agent.html'}")

    # 기존 index.html도 함께 배포 (없으면 빈 페이지)
    index = out_dir / "index.html"
    if not index.exists():
        index.write_text('<meta http-equiv="refresh" content="0;url=agent.html">')


if __name__ == "__main__":
    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else "all_artifacts"
    combo_id_arg  = sys.argv[2] if len(sys.argv) > 2 else "unknown"
    commit_sha    = sys.argv[3] if len(sys.argv) > 3 else "unknown"
    build(artifacts_dir, combo_id_arg, commit_sha)
