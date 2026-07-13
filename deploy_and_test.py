"""
e2e_pipeline.py가 생성한 파일을 rngd-tcp-kernel-dev에 자동 배치하고
cargo furiosa-opt test까지 실행한다. 전체 출력은 로그 파일로 저장하고,
터미널에는 핵심 요약만 보여준다.

사용법:
  python deploy_and_test.py add sub batch_gemm gemm
"""

import subprocess
import shutil
import sys
from pathlib import Path

PIPELINE_DIR = Path.home() / "rngd-mlir-pipeline"
GENERATED_DIR = PIPELINE_DIR / "generated"
KERNEL_PROJECT_DIR = Path.home() / "rngd-tcp-kernel-dev"


def deploy_and_test(op_name: str, run_test: bool = True):
    prefix = f"e2e_{op_name}"
    kernel_src = GENERATED_DIR / f"{prefix}_kernel.rs"
    host_test_src = GENERATED_DIR / f"{prefix}_host_test.rs"
    reference_src = GENERATED_DIR / f"{prefix}_reference_data.rs"

    for f in (kernel_src, host_test_src, reference_src):
        if not f.exists():
            raise FileNotFoundError(
                f"생성된 파일이 없습니다: {f}\n"
                f"(먼저 python e2e_pipeline.py 를 실행해서 '{op_name}' 산출물을 만들어두세요)"
            )

    kernel_dst = KERNEL_PROJECT_DIR / "src" / "kernel" / f"pilot_e2e_{op_name}_kernel.rs"
    host_test_dst = KERNEL_PROJECT_DIR / "src" / f"pilot_e2e_{op_name}.rs"
    reference_dst = KERNEL_PROJECT_DIR / "src" / f"reference_data_e2e_{op_name}.rs"

    shutil.copy(kernel_src, kernel_dst)
    shutil.copy(host_test_src, host_test_dst)
    shutil.copy(reference_src, reference_dst)
    print(f"[배치] {kernel_dst}")
    print(f"[배치] {host_test_dst}")
    print(f"[배치] {reference_dst}")

    mod_rs = KERNEL_PROJECT_DIR / "src" / "kernel" / "mod.rs"
    mod_line = f"pub mod pilot_e2e_{op_name}_kernel;"
    mod_content = mod_rs.read_text()
    if mod_line not in mod_content:
        with mod_rs.open("a") as f:
            f.write(mod_line + "\n")
        print(f"[mod.rs] 등록: {mod_line}")
    else:
        print("[mod.rs] 이미 등록됨 — 건너뜀")

    cargo_toml = KERNEL_PROJECT_DIR / "Cargo.toml"
    bin_name = f"pilot_e2e_{op_name}"
    cargo_content = cargo_toml.read_text()
    if f'name = "{bin_name}"' not in cargo_content:
        with cargo_toml.open("a") as f:
            f.write(f'\n[[bin]]\nname = "{bin_name}"\npath = "src/pilot_e2e_{op_name}.rs"\n')
        print(f"[Cargo.toml] 등록: {bin_name}")
    else:
        print("[Cargo.toml] 이미 등록됨 — 건너뜀")

    if not run_test:
        return None

    log_path = KERNEL_PROJECT_DIR / f"test_log_{op_name}.txt"
    print(f"\n=== cargo furiosa-opt test --bin {bin_name} 실행 (전체 로그: {log_path}) ===\n")
    result = subprocess.run(
        ["cargo", "furiosa-opt", "test", "--release", "--bin", bin_name, "--", "--nocapture"],
        cwd=str(KERNEL_PROJECT_DIR),
        capture_output=True,
        text=True,
    )

    with log_path.open("w") as f:
        f.write("=== STDOUT ===\n")
        f.write(result.stdout)
        f.write("\n=== STDERR ===\n")
        f.write(result.stderr)

    stdout_lines = result.stdout.splitlines()
    summary_lines = [
        line for line in stdout_lines
        if ("test result:" in line or "mismatch" in line or "panicked" in line)
    ]
    compare_start = next(
        (i for i, l in enumerate(stdout_lines)
         if l.strip().startswith("===") and "STDOUT" not in l and "STDERR" not in l),
        None,
    )
    preview_lines = stdout_lines[compare_start:compare_start + 6] if compare_start is not None else []

    print("--- 값 비교 미리보기 (앞부분) ---")
    for line in preview_lines:
        print(" ", line)
    print("--- 요약 라인 ---")
    for line in summary_lines:
        print(" ", line)

    if result.returncode != 0 and not summary_lines:
        print("--- STDERR 마지막 20줄 ---")
        for line in result.stderr.splitlines()[-20:]:
            print(" ", line)

    print(f"\n(전체 로그는 {log_path} 에서 확인 가능: cat {log_path} 또는 less {log_path})")

    return result.returncode == 0


if __name__ == "__main__":
    op_names = sys.argv[1:] if len(sys.argv) > 1 else ["add", "sub"]

    summary = {}
    for op in op_names:
        print(f"\n{'#'*70}\n# {op}\n{'#'*70}")
        try:
            passed = deploy_and_test(op)
            summary[op] = "PASS" if passed else "FAIL"
        except Exception as e:
            summary[op] = f"ERROR: {e}"

    print("\n\n=== 요약 ===")
    for op, status in summary.items():
        marker = "PASS" if status == "PASS" else "FAIL"
        print(f"  [{marker}] {op}: {status}")
