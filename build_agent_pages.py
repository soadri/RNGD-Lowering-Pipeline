"""
build_agent_pages.py — Agent 실험 결과 → GitHub Pages agent.html 빌드
"""
import json
import pathlib
import sys
from datetime import datetime


def build(artifacts_dir: str, combo_id: str = "", commit_sha: str = ""):
    artifacts = pathlib.Path(artifacts_dir)
    out_dir   = pathlib.Path("pages_out")
    out_dir.mkdir(exist_ok=True)

    # agent_log.json 읽기
    log_path = pathlib.Path("agent_log.json")
    experiments = []
    if log_path.exists():
        try:
            experiments = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            experiments = []

    # 통계
    total   = len(experiments)
    success = sum(1 for e in experiments if e["status"] == "success")
    fail    = sum(1 for e in experiments if e["status"] == "fail")
    error   = sum(1 for e in experiments if e["status"] == "error")
    pending = sum(1 for e in experiments if e["status"] in ("running", "pending"))

    from agent_strategy import gen_2layer, gen_3layer, gen_4layer
    T2, T3, T4 = len(gen_2layer()), len(gen_3layer()), len(gen_4layer())
    TARGET  = T2 + T3 + T4
    pct     = min(total / TARGET * 100, 100) if TARGET > 0 else 0
    now     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 최근 50개
    recent = sorted(experiments, key=lambda e: e.get("id", 0), reverse=True)[:50]
    rows   = ""
    for e in recent:
        status   = e["status"]
        icon     = {"success": "✅", "fail": "❌", "error": "⚠️",
                    "running": "⏳", "pending": "🔵"}.get(status, "❓")
        ops_raw  = e.get("ops", "[]")
        ops_list = json.loads(ops_raw) if isinstance(ops_raw, str) else ops_raw
        ops_str  = " → ".join(ops_list)
        sha      = e.get("commit_sha", "")[:8] or "-"
        finished = e.get("finished_at", "-") or "-"
        err      = e.get("error_msg", "") or ""
        err_td   = f'<span title="{err}" style="color:#999;font-size:0.8em">{err[:40]}{"…" if len(err)>40 else ""}</span>' if err else ""
        rows += f"""
        <tr>
          <td style="text-align:center">{e.get('id','')}</td>
          <td><code>{ops_str}</code></td>
          <td style="text-align:center">{icon} {status}</td>
          <td style="text-align:center">{e.get('coverage_pct',0):.0f}%</td>
          <td style="text-align:center"><code>{sha}</code></td>
          <td>{finished}</td>
          <td>{err_td}</td>
        </tr>"""

    # 2-layer / 3-layer / 4-layer 성공률
    def layer_stats(n):
        exps = [e for e in experiments if len(json.loads(e["ops"]) if isinstance(e["ops"],str) else e["ops"]) == n]
        s = sum(1 for e in exps if e["status"] == "success")
        return len(exps), s

    t2, s2 = layer_stats(2)
    t3, s3 = layer_stats(3)
    t4, s4 = layer_stats(4)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="60">
  <title>RNGD Agent 자동 실험</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #f0f2f5; color: #1a1a2e; }}
    .header {{ background: #1a1a2e; color: white; padding: 20px 30px; }}
    .header h1 {{ font-size: 1.5em; margin-bottom: 5px; }}
    .header p  {{ color: #aaa; font-size: 0.9em; }}
    .nav {{ background: white; padding: 10px 30px; border-bottom: 1px solid #eee; }}
    .nav a {{ margin-right: 20px; color: #0366d6; text-decoration: none; font-size: 0.9em; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
    .card {{ background: white; border-radius: 10px; padding: 20px;
             box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 20px; }}
    .card h2 {{ font-size: 1.1em; margin-bottom: 15px; color: #1a1a2e;
                border-left: 4px solid #0366d6; padding-left: 10px; }}
    .stats {{ display: flex; gap: 12px; flex-wrap: wrap; }}
    .stat {{ background: #f8f9fa; border-radius: 8px; padding: 12px 20px;
             text-align: center; min-width: 100px; flex: 1; }}
    .stat .num {{ font-size: 1.8em; font-weight: bold; }}
    .stat .lbl {{ color: #666; font-size: 0.8em; margin-top: 2px; }}
    .green  {{ color: #28a745; }}
    .red    {{ color: #dc3545; }}
    .orange {{ color: #fd7e14; }}
    .blue   {{ color: #007bff; }}
    .progress-wrap {{ margin: 10px 0; }}
    .progress {{ background: #e9ecef; border-radius: 20px; height: 24px; overflow: hidden; }}
    .progress-bar {{ background: linear-gradient(90deg, #28a745, #20c997);
                     height: 24px; border-radius: 20px;
                     display: flex; align-items: center; justify-content: center;
                     color: white; font-size: 0.85em; font-weight: bold;
                     transition: width 0.5s; min-width: 40px; }}
    .layer-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }}
    .layer-card {{ background: #f8f9fa; border-radius: 8px; padding: 15px; text-align: center; }}
    .layer-card .title {{ font-weight: bold; margin-bottom: 8px; color: #555; }}
    .layer-card .nums {{ font-size: 1.3em; font-weight: bold; }}
    .notice {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
               padding: 15px; margin-bottom: 20px; }}
    .notice strong {{ color: #856404; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
    th {{ background: #1a1a2e; color: white; padding: 10px 12px; text-align: left; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #f0f0f0; }}
    tr:hover {{ background: #f8f9fa; }}
    code {{ background: #f1f3f5; padding: 2px 5px; border-radius: 3px; font-size: 0.85em; }}
    .updated {{ color: #999; font-size: 0.8em; text-align: right; margin-top: 10px; }}
    .desc {{ line-height: 1.7; color: #444; }}
    .desc li {{ margin: 5px 0 5px 20px; }}
    .combo-example {{ background: #f8f9fa; border-radius: 6px; padding: 10px 15px;
                      font-family: monospace; font-size: 0.9em; margin: 5px 0; color: #333; }}
  </style>
</head>
<body>

<div class="header">
  <h1>🤖 RNGD Agent — 자동 연산 조합 실험</h1>
  <p>RNGD NPU 컴파일러 파이프라인의 지원 연산 조합을 자동으로 생성하고 검증합니다</p>
</div>

<div class="nav">

  <a href="agent.html">🤖 Agent 실험 현황</a>
</div>

<div class="container">

  <!-- 주의사항 -->
  <div class="notice">
    <strong>⚠️ 자동 실험 진행 중</strong><br>
    현재 RNGD Agent가 자동으로 연산 조합 모델을 생성하고 CI 파이프라인을 통해 검증하고 있습니다.
    <code>models/agent/</code> 디렉토리에 자동 생성된 모델 파일이 지속적으로 push됩니다.
    실험을 중단하려면 관리자에게 문의하거나 <code>AGENT_STOP</code> 파일을 생성하세요.
  </div>

  <!-- 실험 목적 -->
  <div class="card">
    <h2>📋 실험 목적</h2>
    <div class="desc">
      <p>RNGD NPU 컴파일러 파이프라인이 현재 지원하는 17종의 연산을 다양한 방식으로 조합하여
      실제 AI 모델에서 사용 가능한 연산 패턴을 체계적으로 검증합니다.</p>
      <br>
      <p><strong>검증 목표:</strong></p>
      <ul>
        <li>지원 연산들의 조합이 linalg IR → RNGD IR 변환 파이프라인을 통과하는지 확인</li>
        <li>커버리지 100% 달성 가능한 모델 패턴 데이터베이스 구축</li>
        <li>향후 FuriosaAI RNGD 하드웨어 실행을 위한 검증된 연산 조합 목록 확보</li>
      </ul>
    </div>
  </div>

  <!-- 실험 방식 -->
  <div class="card">
    <h2>🔬 실험 방식</h2>
    <div class="desc">
      <p>아래 두 카테고리의 연산을 조합하여 2~4 레이어 모델을 자동 생성합니다.</p>
      <br>
      <p><strong>Contraction 연산 (4종):</strong>
        <code>gemm</code> · <code>batch_gemm</code> · <code>gemv</code> · <code>dot_product</code>
      </p>
      <p><strong>Elementwise 연산 (12종 + 1종):</strong>
        <code>add</code> · <code>sub</code> · <code>mul</code> · <code>div</code> ·
        <code>rsqrt</code> · <code>sqrt</code> · <code>exp</code> · <code>sigmoid</code> ·
        <code>tanh</code> · <code>sin</code> · <code>cos</code> · <code>pow2</code>
      </p>
      <br>
      <p><strong>조합 패턴:</strong></p>
      <div class="combo-example">2-layer: [Contraction] → [Elementwise]</div>
      <div class="combo-example">3-layer: [Contraction] → [Elementwise] → [Contraction]</div>
      <div class="combo-example">4-layer: [Contraction] → [Elementwise] → [Contraction] → [Elementwise]</div>
      <br>
      <p>각 실험은 모델 자동 생성 → torch_mlir 컴파일 검증 → git push → GitHub Actions CI →
      커버리지 분석 순으로 진행됩니다. 실험 간 60초 쿨다운을 둡니다.</p>
    </div>
  </div>

  <!-- 진행 현황 -->
  <div class="card">
    <h2>📊 전체 진행 현황</h2>
    <div class="progress-wrap">
      <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
        <span style="font-size:0.9em; color:#555">진행률</span>
        <span style="font-size:0.9em; font-weight:bold">{total} / {TARGET}개</span>
      </div>
      <div class="progress">
        <div class="progress-bar" style="width:{pct:.1f}%">{pct:.1f}%</div>
      </div>
    </div>
    <div class="stats" style="margin-top:15px">
      <div class="stat"><div class="num">{total}</div><div class="lbl">시도</div></div>
      <div class="stat"><div class="num green">{success}</div><div class="lbl">✅ 성공</div></div>
      <div class="stat"><div class="num red">{fail}</div><div class="lbl">❌ 실패</div></div>
      <div class="stat"><div class="num orange">{error}</div><div class="lbl">⚠️ 오류</div></div>
      <div class="stat"><div class="num blue">{pending}</div><div class="lbl">⏳ 진행중</div></div>
      <div class="stat"><div class="num">{TARGET - total}</div><div class="lbl">남은 조합</div></div>
    </div>
  </div>

  <!-- 레이어별 현황 -->
  <div class="card">
    <h2>📈 레이어별 현황</h2>
    <div class="layer-grid">
      <div class="layer-card">
        <div class="title">2-layer (목표 {T2}개)</div>
        <div class="nums green">{s2}</div>
        <div style="color:#999;font-size:0.85em">성공 / {t2}개 시도</div>
      </div>
      <div class="layer-card">
        <div class="title">3-layer (목표 {T3}개)</div>
        <div class="nums green">{s3}</div>
        <div style="color:#999;font-size:0.85em">성공 / {t3}개 시도</div>
      </div>
      <div class="layer-card">
        <div class="title">4-layer (목표 {T4}개)</div>
        <div class="nums green">{s4}</div>
        <div style="color:#999;font-size:0.85em">성공 / {t4}개 시도</div>
      </div>
    </div>
  </div>

  <!-- 2-layer 히트맵 -->
  <div class="card">
    <h2>🗺️ 2-layer 조합 히트맵</h2>
    <p style="color:#666;font-size:0.9em;margin-bottom:12px">
      Contraction × Elementwise 조합 결과 (✅ 성공 / ❌ 실패 / ⚠️ 오류 / — 미시도)
    </p>
    {heatmap_html}
  </div>

  <!-- 최근 실험 결과 -->
  <div class="card">
    <h2>🔬 최근 실험 결과 (최신 50개)</h2>
    <table>
      <thead>
        <tr>
          <th style="width:50px">#</th>
          <th>연산 조합</th>
          <th style="width:120px">상태</th>
          <th style="width:80px">커버리지</th>
          <th style="width:90px">Commit</th>
          <th style="width:150px">완료 시각</th>
          <th>비고</th>
        </tr>
      </thead>
      <tbody>
        {rows if rows else '<tr><td colspan="7" style="text-align:center;padding:30px;color:#999">아직 실험 결과가 없습니다</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="updated">자동 새로고침: 60초마다 | 마지막 빌드: {now}</div>
</div>
</body>
</html>"""

    (out_dir / "agent.html").write_text(html, encoding="utf-8")
    print(f"agent.html 빌드 완료 → pages_out/agent.html")

    # index.html은 build_pages.py가 관리 — 건드리지 않음


if __name__ == "__main__":
    artifacts_dir = sys.argv[1] if len(sys.argv) > 1 else "all_artifacts"
    combo_id_arg  = sys.argv[2] if len(sys.argv) > 2 else ""
    commit_sha    = sys.argv[3] if len(sys.argv) > 3 else ""
    build(artifacts_dir, combo_id_arg, commit_sha)
