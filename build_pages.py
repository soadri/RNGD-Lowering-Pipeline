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

cards_html = ""
for cov in coverages:
    pct = cov["coverage_pct"]
    color = "#1e5c3f" if pct >= 80 else "#9a5b13" if pct >= 50 else "#a3342a"
    bg = "#e2f0e9" if pct >= 80 else "#faeeda" if pct >= 50 else "#f8ece9"
    model_name = pathlib.Path(cov["model"]).stem
    unsup = [r["op"] for r in cov["ops"] if not r["supported"]]
    unsup_html = "".join(
        f'<code style="margin:2px;display:inline-block;background:#f0ede6;padding:1px 6px;border-radius:4px;font-size:11px">{op}</code>'
        for op in unsup
    ) or '<span style="color:#aaa">없음</span>'
    cards_html += f"""
    <div style="background:#fff;border:1px solid #ddd;border-radius:10px;padding:20px 24px;margin-bottom:16px">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <strong style="font-size:16px">{model_name}</strong>
        <span style="background:{bg};color:{color};padding:4px 12px;border-radius:20px;font-weight:bold;font-size:14px">{pct}%</span>
      </div>
      <div style="font-size:13px;color:#666;margin-bottom:8px">지원 {cov['supported_count']}개 / 전체 {cov['total_ops']}개 op</div>
      <div style="font-size:12px;color:#999;margin-bottom:4px">미지원 op</div>
      <div>{unsup_html}</div>
    </div>"""

reg_rows = ""
for op in registry_ops:
    fc = "#1e5c3f" if op["family"] == "elementwise" else "#5a3d9e"
    fbg = "#e2f0e9" if op["family"] == "elementwise" else "#eee8fb"
    exp = '<span style="font-size:9px;background:#faeeda;color:#9a5b13;padding:1px 5px;border-radius:6px;font-weight:bold;margin-left:4px">EXP</span>' if op.get("experimental") else ""
    reg_rows += f"""<tr>
      <td style="padding:8px 10px;border-bottom:1px solid #eee;font-weight:600">{op["op_name"]}{exp}</td>
      <td style="padding:8px 10px;border-bottom:1px solid #eee"><span style="background:{fbg};color:{fc};padding:2px 8px;border-radius:6px;font-size:11px;font-weight:bold">{op["family"]}</span></td>
      <td style="padding:8px 10px;border-bottom:1px solid #eee;font-size:12px;font-family:monospace">{op["source_op"]} → {op["target_op"]}</td>
      <td style="padding:8px 10px;border-bottom:1px solid #eee"><code style="font-size:11px">{op["dtype"]}</code></td>
      <td style="padding:8px 10px;border-bottom:1px solid #eee;font-size:12px">{op["tolerance"]}</td>
    </tr>"""

body_cards = cards_html if cards_html else '<p style="color:#aaa">분석 결과 없음</p>'

html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RNGD Lowering CI 리포트</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", "Noto Sans KR", sans-serif;
         background:#fafaf9;color:#1c1c1a;max-width:900px;margin:40px auto;padding:0 20px }}
  h1 {{ font-size:22px;margin-bottom:4px }}
  .sub {{ color:#888;font-size:13px;margin-bottom:32px }}
  h2 {{ font-size:16px;margin:32px 0 12px;padding-bottom:8px;border-bottom:1px solid #ddd }}
  table {{ width:100%;border-collapse:collapse;font-size:13px }}
  th {{ text-align:left;padding:8px 10px;font-size:11px;color:#888;text-transform:uppercase;
        letter-spacing:.05em;border-bottom:2px solid #1c1c1a;font-weight:600 }}
</style>
</head>
<body>
<h1>RNGD Lowering CI 리포트</h1>
<div class="sub">생성: {now} · commit: {commit}</div>
<h2>모델별 커버리지</h2>
{body_cards}
<h2>지원 연산 레지스트리 ({len(registry_ops)}개)</h2>
<table>
  <tr><th>연산</th><th>Family</th><th>변환 규칙</th><th>dtype</th><th>허용 오차</th></tr>
  {reg_rows}
</table>
</body>
</html>"""

(out_dir / "index.html").write_text(html, encoding="utf-8")
print(f"Pages 빌드 완료: {out_dir / 'index.html'}")
