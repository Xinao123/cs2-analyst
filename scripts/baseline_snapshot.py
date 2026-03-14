#!/usr/bin/env python3
"""Capture a lightweight baseline snapshot for rollout guardrails."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from db.models import Database


def capture_baseline(config_path: str, output_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    db = Database(config["database"]["path"])
    stats = db.get_stats()

    with db.connect() as conn:
        row = conn.execute(
            """SELECT
                   COUNT(*) AS total,
                   SUM(CASE WHEN outcome_status='win' THEN 1 ELSE 0 END) AS wins,
                   SUM(CASE WHEN outcome_status='loss' THEN 1 ELSE 0 END) AS losses,
                   SUM(CASE WHEN outcome_status='pending' THEN 1 ELSE 0 END) AS pending
               FROM daily_top5_items
               WHERE datetime(COALESCE(resolved_at, match_date, '1970-01-01')) >= datetime('now', '-48 hours')"""
        ).fetchone()

    wins = int(row["wins"] or 0)
    losses = int(row["losses"] or 0)
    pending = int(row["pending"] or 0)
    resolved = wins + losses
    acc = (wins / resolved * 100.0) if resolved else 0.0

    payload = {
        "captured_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config_path": config_path,
        "stats": stats,
        "live_48h": {
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "resolved": resolved,
            "accuracy": round(acc, 2),
        },
        "avg_clv_30d": round(float(db.get_avg_clv(days=30)), 4),
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Capture baseline metrics snapshot")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="data/baseline_metrics.json")
    args = parser.parse_args()

    payload = capture_baseline(args.config, args.out)
    print("Baseline salvo em:", args.out)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
