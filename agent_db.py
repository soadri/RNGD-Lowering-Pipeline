"""
agent_db.py — 실험 이력 DB (SQLite)
"""
import sqlite3
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import json

DB_PATH = Path(__file__).parent / "experiment_log.db"

@dataclass
class Experiment:
    id: int
    combo_id: str          # "gemm-sigmoid-gemm"
    ops: list              # ["gemm", "sigmoid", "gemm"]
    model_file: str        # "models/agent_gemm_sigmoid_gemm.py"
    status: str            # "pending" | "running" | "success" | "fail" | "error"
    coverage_pct: float
    commit_sha: str
    ci_run_id: int
    error_msg: str
    created_at: str
    finished_at: str

def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            combo_id     TEXT UNIQUE NOT NULL,
            ops          TEXT NOT NULL,        -- JSON list
            model_file   TEXT NOT NULL,
            status       TEXT NOT NULL DEFAULT 'pending',
            coverage_pct REAL DEFAULT 0.0,
            commit_sha   TEXT DEFAULT '',
            ci_run_id    INTEGER DEFAULT 0,
            error_msg    TEXT DEFAULT '',
            created_at   TEXT DEFAULT (datetime('now', 'localtime')),
            finished_at  TEXT DEFAULT ''
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS agent_state (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    con.commit()
    con.close()

def get_con():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con

def insert_experiment(combo_id: str, ops: list, model_file: str) -> bool:
    """이미 있으면 False, 새로 삽입하면 True"""
    con = get_con()
    try:
        con.execute(
            "INSERT INTO experiments (combo_id, ops, model_file) VALUES (?, ?, ?)",
            (combo_id, json.dumps(ops), model_file)
        )
        con.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        con.close()

def update_experiment(combo_id: str, **kwargs):
    con = get_con()
    sets = ", ".join(f"{k}=?" for k in kwargs)
    vals = list(kwargs.values()) + [combo_id]
    con.execute(f"UPDATE experiments SET {sets} WHERE combo_id=?", vals)
    con.commit()
    con.close()

def get_all_experiments():
    con = get_con()
    rows = con.execute("SELECT * FROM experiments ORDER BY id").fetchall()
    con.close()
    return [dict(r) for r in rows]

def get_tried_combos():
    con = get_con()
    rows = con.execute("SELECT combo_id FROM experiments").fetchall()
    con.close()
    return {r["combo_id"] for r in rows}

def set_state(key: str, value: str):
    con = get_con()
    con.execute("INSERT OR REPLACE INTO agent_state (key, value) VALUES (?, ?)", (key, value))
    con.commit()
    con.close()

def get_state(key: str, default: str = "") -> str:
    con = get_con()
    row = con.execute("SELECT value FROM agent_state WHERE key=?", (key,)).fetchone()
    con.close()
    return row["value"] if row else default

if __name__ == "__main__":
    init_db()
    print(f"DB 초기화 완료: {DB_PATH}")
