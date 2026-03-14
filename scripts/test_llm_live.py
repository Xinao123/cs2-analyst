"""Manual smoke test for DeepSeek prompts with real local data.

Usage:
    python -m scripts.test_llm_live --config config.yaml
"""

import argparse
import logging

from ai.context import ContextCollector
from ai.llm import DeepSeekClient
from analysis.features import FeatureExtractor
from analysis.predictor import Predictor
from analysis.value import ValueDetector
from db.models import Database
from main import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="LLM live smoke test")
    parser.add_argument("--config", "-c", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    db = Database(config["database"]["path"])
    llm = DeepSeekClient(config)
    ctx = ContextCollector(db, config)
    feat = FeatureExtractor(db, config)
    pred = Predictor(config)
    value = ValueDetector(config)

    if not llm.is_available:
        print("LLM indisponivel. Verifique llm.enabled e DEEPSEEK_API_KEY.")
        return

    match = _pick_match(db)
    if not match:
        print("Nenhuma partida encontrada para teste.")
        return

    features = feat.extract(match)
    if not features:
        print("Sem features suficientes para a partida selecionada.")
        return

    prediction = pred.predict(features)
    if not prediction:
        print("Modelo nao retornou predicao para a partida selecionada.")
        return

    analysis = value.analyze(
        prediction=prediction,
        odds_team1=match.get("odds_team1"),
        odds_team2=match.get("odds_team2"),
        match=match,
    ) or {}

    context_text = ctx.collect(match)
    print("\n=== MATCH ===")
    print(f"{match.get('team1_name', '?')} vs {match.get('team2_name', '?')}")
    print("\n=== CONTEXT (preview) ===")
    print(context_text[:1200])

    llm_match = llm.generate_match_analysis(
        match=match,
        features=features,
        prediction=prediction,
        analysis=analysis,
        context_text=context_text,
    )
    print("\n=== LLM MATCH ANALYSIS ===")
    print(llm_match or "(vazio)")

    llm_top = llm.generate_top_picks_report(
        picks=[
            {
                "match": match,
                "features": features,
                "prediction": prediction,
                "analysis": analysis,
                "score": 0.0,
                "official_pick_winner_id": int(match.get("team1_id", 0) or 0),
                "best_vb": (analysis.get("value_bets") or [{}])[0],
            }
        ],
        total_candidates=1,
    )
    print("\n=== LLM TOP PICKS ===")
    print(llm_top or "(vazio)")

    llm_audit = llm.generate_audit_report(
        {
            "run_date": "manual",
            "wins": 1,
            "losses": 0,
            "pending": 0,
            "accuracy": 100.0,
            "items": [
                {
                    "outcome_status": "win",
                    "team1_name": match.get("team1_name", "?"),
                    "team2_name": match.get("team2_name", "?"),
                    "official_pick_winner_name": match.get("team1_name", "?"),
                }
            ],
            "found": True,
        }
    )
    print("\n=== LLM AUDIT ===")
    print(llm_audit or "(vazio)")

    print("\n=== COST ===")
    print(f"calls_this_cycle={llm.calls_this_cycle}")
    print(f"month_cost_usd={llm._month_cost_usd:.6f}")


def _pick_match(db: Database) -> dict | None:
    upcoming = db.get_upcoming_matches()
    if upcoming:
        return upcoming[0]
    preds = db.get_prediction_history(limit=1)
    if not preds:
        return None
    return db.get_match_result_by_id(int(preds[0].get("match_id", 0) or 0))


if __name__ == "__main__":
    main()
