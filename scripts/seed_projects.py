"""一次性数据播种：把模块 3 的闭环数据搬进 Web 界面的多级目录。

运行：python scripts/seed_projects.py
（Streamlit 应用本身不会在启动时改数据，播种是独立、显式的一步。）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app import (  # noqa: E402
    HISTORY_NAME,
    PROJECTS_ROOT,
    create_project,
    history_columns,
    space_for,
)


def seed_demo() -> None:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

    # ① 有预设的主项目：Cu / SiO2_H2O2_Glycine，灌入 34 条闭环真实数据
    ref = create_project("Cu", "SiO2", "H2O2", "Glycine")
    space, blind = space_for(ref)
    payload = json.loads(
        (ROOT / "data" / "closed_loop_n34.json").read_text(encoding="utf-8")
    )
    X = payload["recipes"]
    Y = payload["responses"]
    rounds = {r["round_index"]: r for r in payload["extra"].get("rounds", [])}
    costs = {}
    for meta in payload["extra"].get("rounds", []):
        for k, c in enumerate(meta.get("cost_cny") or []):
            costs[25 + int(meta["round_index"]) * 3 + k] = float(c)

    rows = []
    for i, (x, y) in enumerate(zip(X, Y)):
        row = {n: float(x[j]) for j, n in enumerate(space.names)}
        for j, n in enumerate(space.objective_names):
            row[n] = float(y[j])
        for c in space.covariates:
            row[c.name] = None
        row.update({
            "experiment_id": f"E{i:03d}",
            "round_id": 0 if i < 25 else 1 + (i - 25) // 3,
            "batch_id": "B00" if i < 25 else f"B{1 + (i - 25) // 3:02d}",
            "timestamp": "",
            "operator": "",
            "anomaly": "none",
            "failure": "none",
            "notes": "seeded from closed_loop_n34",
            "cost_cny": costs.get(i),
        })
        rows.append(row)

    df = pd.DataFrame(rows, columns=history_columns(space))
    df.to_csv(ref.history_path, index=False, encoding="utf-8-sig")
    print(f"[seed] {ref.rel_label}: {len(df)} 行, 盲跑={blind}")

    # ② 无预设的第二个体系：W / Al2O3_APS_CitricAcid（空表，用于演示隔离与盲跑警告）
    ref2 = create_project("W", "Al2O3", "APS", "CitricAcid")
    space2, blind2 = space_for(ref2)
    if not ref2.history_path.exists() or ref2.history_path.stat().st_size == 0:
        pd.DataFrame(columns=history_columns(space2)).to_csv(
            ref2.history_path, index=False, encoding="utf-8-sig"
        )
    print(f"[seed] {ref2.rel_label}: 空表, 盲跑={blind2}")

    for p in sorted(PROJECTS_ROOT.rglob(HISTORY_NAME)):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size} B")


if __name__ == "__main__":
    seed_demo()
