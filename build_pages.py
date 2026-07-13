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

# 커널 테스트 결과 로드
kernel_results = {}
kr_path = artifacts_dir / "kernel-results/kernel_results.json"
if kr_path.exists():
    kernel_results = json.loads(kr_path.read_text())
    print(f"커널 결과 로드: {len(kernel_results)}개 연산")

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
    "kernel_results": kernel_results,
})
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

def kernel_badge(op_name):
    if not kernel_results:
        return '<span style="color:#aaa;font-size:11px">-</span>'
    result = kernel_results.get(op_name)
    if result == "PASS":
        return '<span style="background:#e2f0e9;color:#1e5c3f;padding:1px 7px;border-radius:6px;font-size:11px;font-weight:bold">PASS</span>'
    elif result == "FAIL":
        return '<span style="background:#f8ece9;color:#a3342a;padding:1px 7px;border-radius:6px;font-size:11px;font-weight:bold">FAIL</span>'
    return '<span style="color:#aaa;font-size:11px">-</span>'

# ── 커널 테스트 요약 ──────────────────────────────────────────────
kernel_summary_html = ""
if kernel_results:
    passed = [op for op, r in kernel_results.items() if r == "PASS"]
    failed = [op for op, r in kernel_results.items() if r != "PASS"]
    total = len(kernel_results)
    summary_color = "#1e5c3f" if not failed else "#a3342a"
    summary_bg = "#e2f0e9" if not failed else "#f8ece9"
    kernel_summary_html = f"""
    <div style="background:{summary_bg};border:1px solid {summary_color};border-radius:10px;
                padding:16px 20px;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center">
      <div>
        <strong style="color:{summary_color};font-size:15px">커널 정합성 검증</strong>
        <span style="font-size:13px;color:#666;margin-left:10px">
          {len(passed)}/{total}개 PASS
          {"· 모두 통과 ✅" if not failed else f"· FAIL: {', '.join(failed)}"}
        </span>
      </div>
    </div>"""

# ── 모델별 카드 (아코디언) ────────────────────────────────────────
cards_html = ""
for i, cov in enumerate(coverages):
    model_name = pathlib.Path(cov["model"]).stem
    pct = cov["coverage_pct"]
    color, bg = pct_style(pct)
    sup   = [r for r in cov["ops"] if r["supported"]]
    unsup = [r for r in cov["ops"] if not r["supported"]]

    unsup_tags = "".join(op_tag(r["op"]) for r in unsup) or '<span style="color:#aaa">없음</span>'

    # 지원 op 테이블 — 커널 검증 결과 컬럼 추가
    sup_rows = "".join(
        f'<tr>'
        f'<td style="padding:6px 8px;font-family:monospace;font-size:12px;border-bottom:1px solid #eee">{r["op"]}</td>'
        f'<td style="padding:6px 8px;font-size:12px;border-bottom:1px solid #eee">{r.get("rngd_op","")}</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee">'
        f'<span style="background:{"#e2f0e9" if r.get("family")=="elementwise" else "#eee8fb"};'
        f'color:{"#1e5c3f" if r.get("family")=="elementwise" else "#5a3d9e"};'
        f'padding:1px 6px;border-radius:4px;font-size:11px;font-weight:bold">{r.get("family","")}</span>'
        f'</td>'
        f'<td style="padding:6px 8px;border-bottom:1px solid #eee">{kernel_badge(r.get("op_name",""))}</td>'
        f'</tr>'
        for r in sup
    )
    unsup_rows = "".join(
        f'<tr><td style="padding:6px 8px;font-family:monospace;font-size:12px;'
        f'border-bottom:1px solid #eee;color:#a3342a">{r["op"]}</td></tr>'
        for r in unsup
    )

    # 이 모델의 커널 검증 통과 여부
    model_kernel_ops = [r.get("op_name") for r in sup if r.get("op_name")]
    model_passed = all(kernel_results.get(op) == "PASS" for op in model_kernel_ops if op in kernel_results)
    model_has_kernel = any(op in kernel_results for op in model_kernel_ops)
    kernel_indicator = ""
    if model_has_kernel:
        if model_passed:
            kernel_indicator = '<span style="font-size:11px;background:#e2f0e9;color:#1e5c3f;padding:1px 7px;border-radius:6px;margin-left:8px">커널 ✅</span>'
        else:
            kernel_indicator = '<span style="font-size:11px;background:#f8ece9;color:#a3342a;padding:1px 7px;border-radius:6px;margin-left:8px">커널 ❌</span>'

    cards_html += f"""
    <div style="background:#fff;border:1px solid #ddd;border-radius:10px;margin-bottom:12px;overflow:hidden">
      <div onclick="toggle({i})"
           style="display:flex;justify-content:space-between;align-items:center;
                  padding:16px 20px;cursor:pointer;user-select:none">
        <div>
          <strong style="font-size:15px">{model_name}</strong>
          {kernel_indicator}
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
        <div style="display:grid;grid-template-columns:2fr 1fr;gap:16px">
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
                <th style="text-align:left;padding:4px 8px;font-size:10px;
                           color:#888;border-bottom:1px solid #ddd">커널 검증</th>
              </tr>
              {sup_rows or '<tr><td colspan="4" style="padding:6px 8px;color:#aaa">없음</td></tr>'}
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
    kr  = kernel_badge(op["op_name"])
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
      <td style="padding:8px 10px;border-bottom:1px solid #eee">{kr}</td>
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

{kernel_summary_html}

<h2>모델별 커버리지 ({len(coverages)}개 모델)</h2>
{cards_html if cards_html else '<p style="color:#aaa">분석 결과 없음</p>'}

<h2>지원 연산 레지스트리 ({len(registry_ops)}개)</h2>
<table>
  <tr>
    <th>연산</th><th>Family</th><th>변환 규칙</th>
    <th>dtype</th><th>허용 오차</th><th>커널 검증</th>
  </tr>
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
