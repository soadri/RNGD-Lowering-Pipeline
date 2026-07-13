import json, pathlib, datetime, sys

artifacts_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path("all_artifacts")
out_dir = pathlib.Path("pages_out")
out_dir.mkdir(exist_ok=True)

coverages = []
for p in sorted(artifacts_dir.glob("**/coverage.json")):
    with open(p) as f:
        coverages.append(json.load(f))

snapshot_path = artifacts_dir / "registry-snapshot/registry_snapshot.json"
registry_ops = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else []

now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
commit = pathlib.Path("COMMIT_SHA").read_text().strip()[:7] if pathlib.Path("COMMIT_SHA").exists() else "unknown"
commit_full = pathlib.Path("COMMIT_SHA").read_text().strip() if pathlib.Path("COMMIT_SHA").exists() else "unknown"

# ── history.json 업데이트 ─────────────────────────────────────────
history_path = out_dir / "history.json"
history = json.loads(history_path.read_text()) if history_path.exists() else []
history.append({
    "commit": commit_full,
    "commit_short": commit,
    "generated_at": now,
    "models": [
        {
            "name": pathlib.Path(c["model"]).stem,
            "coverage_pct": c["coverage_pct"],
            "supported": c["supported_count"],
            "total": c["total_ops"],
        }
        for c in coverages
    ],
})
# 최근 50개만 유지
history = history[-50:]
history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"history.json 업데이트: {len(history)}개 커밋 이력")

# ── 헬퍼 ─────────────────────────────────────────────────────────
def pct_style(pct):
    if pct >= 80:   return "#1e5c3f", "#e2f0e9"
    elif pct >= 50: return "#9a5b13", "#faeeda"
    else:           return "#a3342a", "#f8ece9"

def op_tag(op):
    return f'<code style="margin:2px;display:inline-block;background:#f0ede6;padding:1px 6px;border-radius:4px;font-size:11px">{op}</code>'

# ── 모델별 카드 (아코디언) ────────────────────────────────────────
cards_html = ""
for i, cov in enumerate(coverages):
    model_name = pathlib.Path(cov["model"]).stem
    pct = cov["coverage_pct"]
    color, bg = pct_style(pct)
    sup   = [r for r in cov["ops"] if r["supported"]]
    unsup = [r for r in cov["ops"] if not r["supported"]]

    unsup_tags = "".join(op_tag(r["op"]) for r in unsup) or '<span style="color:#aaa">없음</span>'

    sup_rows = "".join(
        f'<tr>'
        f'<td style="padding:6px 8px;font-family:monospace;font-size:12px;border-bottom:1px solid #eee">{r["op"]}</td>'
        f'<td style="padding:6px 8px;font-size:12px;border-bottom:1px solid #eee">{r.get("rngd_op","")}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee">'
        f'<span style="background:{"#e2f0e9" if r.get("family")=="elementwise" else "#eee8fb"};'
        f'color:{"#1e5c3f" if r.get("family")=="elementwise" else "#5a3d9e"};'
        f'padding:1px 6px;border-radius:4px;font-size:11px;font-weight:bold">{r.get("family","")}</span>'
        f'</td></tr>'
        for r in sup
    )
    unsup_rows = "".join(
        f'<tr><td style="padding:6px 8px;font-family:monospace;font-size:12px;'
        f'border-bottom:1px solid #eee;color:#a3342a">{r["op"]}</td></tr>'
        for r in unsup
    )

    cards_html += f"""
    <div style="background:#fff;border:1px solid #ddd;border-radius:10px;margin-bottom:12px;overflow:hidden">
      <div onclick="toggle({i})"
           style="display:flex;justify-content:space-between;align-items:center;
                  padding:16px 20px;cursor:pointer;user-select:none">
        <div>
          <strong style="font-size:15px">{model_name}</strong>
          <span style="font-size:12px;color:#888;margin-left:10px">
            지원 {cov["supported_count"]}개 / 전체 {cov["total_ops"]}개 op
          </span>
        </div>
        <div style="display:flex;align-items:center;gap:10px">
          <span style="background:{bg};color:{color};padding:3px 10px;
                       border-radius:16px;font-weight:bold;font-size:13px">{pct}%</span>
          <span id="arrow-{i}" style="color:#888;font-size:12px">▼</span>
        </div>
      </div>
      <div id="detail-{i}" style="display:none;border-top:1px solid #eee;padding:16px 20px">
        <div style="margin-bottom:12px">
          <div style="font-size:11px;color:#888;text-transform:uppercase;
                      letter-spacing:.05em;margin-bottom:6px">미지원 op</div>
          {unsup_tags}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
          <div>
            <div style="font-size:11px;color:#888;text-transform:uppercase;
                        letter-spacing:.05em;margin-bottom:6px">✅ 지원 ({len(sup)}개)</div>
            <table style="width:100%;border-collapse:collapse;font-size:12px">
              <tr>
                <th style="text-align:left;padding:4px 8px;font-size:10px;
                           color:#888;border-bottom:1px solid #ddd">linalg op</th>
                <th style="text-align:left;padding:4px 8px;font-size:10px;
                           color:#888;border-bottom:1px solid #ddd">RNGD op</th>
                <th style="text-align:left;padding:4px 8px;font-size:10px;
                           color:#888;border-bottom:1px solid #ddd">Family</th>
              </tr>
              {sup_rows or '<tr><td colspan="3" style="padding:6px 8px;color:#aaa">없음</td></tr>'}
            </table>
          </div>
          <div>
            <div style="font-size:11px;color:#888;text-transform:uppercase;
                        letter-spacing:.05em;margin-bottom:6px">❌ 미지원 ({len(unsup)}개)</div>
            <table style="width:100%;border-collapse:collapse;font-size:12px">
              <tr>
                <th style="text-align:left;padding:4px 8px;font-size:10px;
                           color:#888;border-bottom:1px solid #ddd">linalg op</th>
              </tr>
              {unsup_rows or '<tr><td style="padding:6px 8px;color:#aaa">없음</td></tr>'}
            </table>
          </div>
        </div>
      </div>
    </div>"""

# ── 레지스트리 테이블 ─────────────────────────────────────────────
reg_rows = ""
for op in registry_ops:
    fc  = "#1e5c3f" if op["family"] == "elementwise" else "#5a3d9e"
    fbg = "#e2f0e9" if op["family"] == "elementwise" else "#eee8fb"
    exp = '<span style="font-size:9px;background:#faeeda;color:#9a5b13;padding:1px 5px;border-radius:6px;font-weight:bold;margin-left:4px">EXP</span>' if op.get("experimental") else ""
    reg_rows += f"""<tr>
      <td style="padding:8px 10px;border-bottom:1px solid #eee;font-weight:600">{op["op_name"]}{exp}</td>
      <td style="padding:8px 10px;border-bottom:1px solid #eee">
        <span style="background:{fbg};color:{fc};padding:2px 8px;border-radius:6px;
                     font-size:11px;font-weight:bold">{op["family"]}</span></td>
      <td style="padding:8px 10px;border-bottom:1px solid #eee;
                 font-size:12px;font-family:monospace">{op["source_op"]} → {op["target_op"]}</td>
      <td style="padding:8px 10px;border-bottom:1px solid #eee">
        <code style="font-size:11px">{op["dtype"]}</code></td>
      <td style="padding:8px 10px;border-bottom:1px solid #eee;font-size:12px">{op["tolerance"]}</td>
    </tr>"""

# ── 전체 index.html ───────────────────────────────────────────────
index_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RNGD Lowering CI 리포트</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Noto Sans KR", sans-serif;
         background:#fafaf9;color:#1c1c1a;max-width:900px;margin:40px auto;padding:0 20px }}
  h1 {{ font-size:22px;margin-bottom:4px }}
  h2 {{ font-size:16px;margin:32px 0 12px;padding-bottom:8px;border-bottom:1px solid #ddd }}
  .sub {{ color:#888;font-size:13px;margin-bottom:32px }}
  table {{ width:100%;border-collapse:collapse;font-size:13px }}
  th {{ text-align:left;padding:8px 10px;font-size:11px;color:#888;text-transform:uppercase;
        letter-spacing:.05em;border-bottom:2px solid #1c1c1a;font-weight:600 }}
</style>
</head>
<body>
<h1>RNGD Lowering CI 리포트</h1>
<div class="sub">생성: {now} · commit: {commit}</div>

<h2>모델별 커버리지 ({len(coverages)}개 모델)</h2>
{cards_html if cards_html else '<p style="color:#aaa">분석 결과 없음</p>'}

<h2>지원 연산 레지스트리 ({len(registry_ops)}개)</h2>
<table>
  <tr><th>연산</th><th>Family</th><th>변환 규칙</th><th>dtype</th><th>허용 오차</th></tr>
  {reg_rows}
</table>

<script>
function toggle(i) {{
  const d = document.getElementById('detail-' + i);
  const a = document.getElementById('arrow-' + i);
  const open = d.style.display !== 'none';
  d.style.display = open ? 'none' : 'block';
  a.textContent = open ? '▼' : '▲';
}}
</script>
</body>
</html>"""

(out_dir / "index.html").write_text(index_html, encoding="utf-8")
print(f"Pages 빌드 완료: {out_dir}/index.html")
