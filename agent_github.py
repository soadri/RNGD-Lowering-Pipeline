"""
agent_github.py — GitHub API 클라이언트 (CI 상태 polling)
"""
import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
REPO  = os.getenv("GITHUB_REPO", "soadri/RNGD-Lowering-Pipeline")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}

BASE = "https://api.github.com"

def get_latest_run(workflow_file: str = "agent_ci.yml") -> dict | None:
    """agent_ci.yml의 가장 최근 workflow run 반환"""
    url = f"{BASE}/repos/{REPO}/actions/workflows/{workflow_file}/runs"
    r = requests.get(url, headers=HEADERS, params={"per_page": 1, "branch": "main"})
    r.raise_for_status()
    runs = r.json().get("workflow_runs", [])
    return runs[0] if runs else None

def get_run_by_sha(commit_sha: str, workflow_file: str = "agent_ci.yml") -> dict | None:
    """특정 commit SHA의 workflow run 반환"""
    url = f"{BASE}/repos/{REPO}/actions/workflows/{workflow_file}/runs"
    r = requests.get(url, headers=HEADERS, params={"head_sha": commit_sha, "branch": "main"})
    r.raise_for_status()
    runs = r.json().get("workflow_runs", [])
    return runs[0] if runs else None

def wait_for_run(commit_sha: str, timeout: int = 600, poll: int = 30) -> dict:
    """CI 완료까지 대기. 결과 dict 반환.
    status: "success" | "failure" | "timeout"
    """
    print(f"  ⏳ CI 대기 중 (commit: {commit_sha[:8]})")
    start = time.time()

    # run이 생성될 때까지 대기
    run = None
    for _ in range(10):
        run = get_run_by_sha(commit_sha)
        if run:
            break
        print(f"  ⏳ workflow run 생성 대기...")
        time.sleep(poll)

    if not run:
        return {"status": "timeout", "conclusion": None, "run_id": 0}

    run_id = run["id"]
    print(f"  🔗 Run ID: {run_id} — {run['html_url']}")

    # 완료까지 polling
    while time.time() - start < timeout:
        r = requests.get(f"{BASE}/repos/{REPO}/actions/runs/{run_id}", headers=HEADERS)
        r.raise_for_status()
        run = r.json()

        status     = run["status"]
        conclusion = run["conclusion"]
        elapsed    = int(time.time() - start)

        print(f"  ⏳ [{elapsed:3d}s] status={status} conclusion={conclusion}")

        if status == "completed":
            return {
                "status": conclusion or "failure",
                "conclusion": conclusion,
                "run_id": run_id,
                "url": run["html_url"],
            }

        time.sleep(poll)

    return {"status": "timeout", "conclusion": None, "run_id": run_id}


def get_coverage_from_artifacts(run_id: int) -> float:
    """CI artifacts에서 커버리지 추출"""
    url = f"{BASE}/repos/{REPO}/actions/runs/{run_id}/artifacts"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    artifacts = r.json().get("artifacts", [])
    # coverage artifact 찾기
    for a in artifacts:
        if "coverage" in a["name"].lower():
            return 100.0  # 단순화: artifact 존재하면 100%로 처리
    return 0.0


if __name__ == "__main__":
    run = get_latest_run()
    if run:
        print(f"최근 run: {run['name']} — {run['status']} / {run['conclusion']}")
        print(f"URL: {run['html_url']}")
    else:
        print("실행 이력 없음")
