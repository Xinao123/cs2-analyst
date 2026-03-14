import unittest

from ai.prompts import (
    SYSTEM_AUDIT,
    SYSTEM_MATCH_ANALYSIS,
    SYSTEM_TOP_PICKS,
    build_audit_prompt,
    build_match_analysis_prompt,
    build_top_picks_prompt,
)


class PromptBuildersTests(unittest.TestCase):
    def test_match_prompt_includes_context(self):
        prompt = build_match_analysis_prompt(
            match={"team1_name": "FURIA", "team2_name": "TYLOO", "event_name": "ESL", "best_of": 3},
            features={"ranking_diff": -10, "form_diff": 0.11, "h2h_matches": 3, "h2h_winrate_t1": 0.66},
            prediction={"team1_win_prob": 63.0, "team2_win_prob": 37.0, "confidence": 63.0, "predicted_winner": 1},
            analysis={"has_value": False, "odds_team1": 1.75, "odds_team2": 2.10},
            context_text="CONTEXTO: bloco de teste",
        )
        self.assertIn("CONTEXTO: bloco de teste", prompt)
        self.assertIn("FURIA vs TYLOO", prompt)

    def test_match_prompt_compact(self):
        prompt = build_match_analysis_prompt(
            match={"team1_name": "A", "team2_name": "B"},
            features={},
            prediction={},
            analysis={},
            context_text="ctx",
        )
        self.assertLess(len(prompt.split()), 850)

    def test_top_picks_prompt_format(self):
        out = build_top_picks_prompt(
            picks=[
                {
                    "match": {"team1_name": "A", "team2_name": "B", "best_of": 3},
                    "prediction": {"team1_win_prob": 60, "team2_win_prob": 40},
                    "best_vb": {"side": "team1", "odds": 1.9, "bookmaker": "pinnacle", "value_pct": 7.1},
                    "score": 81.2,
                }
            ],
            total_candidates=20,
        )
        self.assertIn("TOP 1 de 20 analisadas", out)
        self.assertIn("A vs B", out)

    def test_audit_prompt_format(self):
        out = build_audit_prompt(
            {
                "run_date": "2026-03-14",
                "wins": 2,
                "losses": 1,
                "pending": 0,
                "accuracy": 66.7,
                "items": [
                    {
                        "outcome_status": "win",
                        "team1_name": "A",
                        "team2_name": "B",
                        "official_pick_winner_name": "A",
                    }
                ],
            }
        )
        self.assertIn("Auditoria 2026-03-14", out)
        self.assertIn("win: A vs B pick:A", out)

    def test_system_prompts_are_short(self):
        for prompt in [SYSTEM_MATCH_ANALYSIS, SYSTEM_TOP_PICKS, SYSTEM_AUDIT]:
            self.assertLess(len(prompt.split()), 150)


if __name__ == "__main__":
    unittest.main()
