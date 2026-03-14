"""Prompt templates and compact builders for DeepSeek integration."""

SYSTEM_MATCH_ANALYSIS = (
    "Voce e um analista senior de CS2 focado em value betting. "
    "TAREFA: analise a partida usando apenas dados fornecidos e explique o racional do modelo. "
    "FORMATO: texto corrido em portugues BR, 120-180 palavras, sem markdown. "
    "REGRAS: mencione forma, ranking, H2H, map pool, contexto do evento e drivers de decisao; "
    "se houver value, explique desalinhamento da odd; nao invente dados."
)

SYSTEM_TOP_PICKS = (
    "Voce e um analista de CS2 que resume oportunidades de aposta. "
    "FORMATO: texto corrido em portugues BR, 80-120 palavras, sem markdown. "
    "REGRAS: destaque o melhor pick e os principais riscos."
)

SYSTEM_AUDIT = (
    "Voce e um analista de performance de apostas em CS2. "
    "FORMATO: texto corrido em portugues BR, 60-100 palavras, sem markdown. "
    "REGRAS: identifique padroes de erro e proponha uma acao objetiva."
)

SYSTEM_EXPLAINABILITY = (
    "Voce traduz feature importance de ML para linguagem clara. "
    "FORMATO: texto corrido em portugues BR, 60-80 palavras, sem markdown. "
    "REGRAS: explique apenas com base nas features fornecidas."
)

SYSTEM_ANOMALY_CHECK = (
    "Voce e um verificador de anomalias para picks de CS2. "
    "Retorne apenas: OK ou FLAG, seguido de no maximo 12 palavras de motivo. "
    "Flag somente para incoerencia evidente entre probabilidade, odd e contexto."
)


def build_match_analysis_prompt(
    match: dict,
    features: dict,
    prediction: dict,
    analysis: dict,
    context_text: str,
) -> str:
    team1 = str(match.get("team1_name", "Team1"))
    team2 = str(match.get("team2_name", "Team2"))
    event = str(match.get("event_name", "?"))
    best_of = int(match.get("best_of", 1) or 1)
    is_lan = "LAN" if bool(match.get("is_lan")) else "Online"

    p1 = float(prediction.get("team1_win_prob", 50.0))
    p2 = float(prediction.get("team2_win_prob", 50.0))
    conf = float(prediction.get("confidence", 50.0))
    model_winner = team1 if int(prediction.get("predicted_winner", 1)) == 1 else team2

    f_line = (
        f"rank_diff:{_safe_num(features.get('ranking_diff')):+.0f} "
        f"wr_diff:{_safe_num(features.get('winrate_diff')):+.3f} "
        f"form_diff:{_safe_num(features.get('form_diff')):+.3f} "
        f"h2h_wr:{_safe_num(features.get('h2h_winrate_t1'), 0.5):.2f}({int(_safe_num(features.get('h2h_matches')))}g) "
        f"streak:{int(_safe_num(features.get('team1_streak'))):+d}/{int(_safe_num(features.get('team2_streak'))):+d} "
        f"maps_diff:{_safe_num(features.get('strong_maps_diff')):+.0f} "
        f"rust:{_safe_num(features.get('team1_rust_days')):.1f}/{_safe_num(features.get('team2_rust_days')):.1f}d "
        f"vol:{_safe_num(features.get('team1_volatility')):.3f}/{_safe_num(features.get('team2_volatility')):.3f}"
    )

    driver_line = (
        f"Drivers: wr7d_diff={_safe_num(features.get('wr_diff_7d')):+.3f} "
        f"bo_wr_diff={_safe_num(features.get('bo_wr_diff')):+.3f} "
        f"venue_wr_diff={_safe_num(features.get('venue_wr_diff')):+.3f} "
        f"side_diff={_safe_num(features.get('side_strength_diff')):+.3f} "
        f"map_adv_diff={_safe_num(features.get('map_advantage_diff')):+.3f}"
    )

    value_str = "Sem value detectado"
    if analysis and analysis.get("has_value"):
        value_bets = analysis.get("value_bets", [])
        if value_bets:
            best = max(
                value_bets,
                key=lambda vb: (
                    float(vb.get("value_pct", 0.0)),
                    float(vb.get("expected_value", 0.0)),
                ),
            )
            value_str = (
                f"VALUE: {best.get('side', '?')} @{_safe_num(best.get('odds')):.2f} "
                f"({best.get('bookmaker', '?')}) +{_safe_num(best.get('value_pct')):.1f}%"
            )

    odds_team1 = _safe_num(analysis.get("odds_team1")) if analysis else 0.0
    odds_team2 = _safe_num(analysis.get("odds_team2")) if analysis else 0.0
    odds_str = ""
    if odds_team1 > 1 and odds_team2 > 1:
        odds_str = f"Odds: {team1}@{odds_team1:.2f} {team2}@{odds_team2:.2f}"

    return (
        f"{team1} vs {team2} | {event} | {is_lan} BO{best_of}\n"
        f"Modelo: {team1} {p1:.1f}% x {p2:.1f}% {team2} | Conf:{conf:.1f}% | Fav:{model_winner}\n"
        f"{odds_str}\n"
        f"{value_str}\n"
        f"Features: {f_line}\n"
        f"{driver_line}\n\n"
        f"{context_text}\n\n"
        "Gere a analise."
    )


def build_top_picks_prompt(picks: list[dict], total_candidates: int) -> str:
    lines = [f"TOP {len(picks)} de {int(total_candidates)} analisadas:"]
    for idx, pick in enumerate(picks, start=1):
        match = pick.get("match", {})
        pred = pick.get("prediction", {})
        vb = pick.get("best_vb", {}) or {}

        team1 = str(match.get("team1_name", "?"))
        team2 = str(match.get("team2_name", "?"))
        bo = int(match.get("best_of", 1) or 1)
        side = str(vb.get("side", "model"))
        odd = _safe_num(vb.get("odds"))
        value_pct = _safe_num(vb.get("value_pct"))
        bookmaker = str(vb.get("bookmaker", "-"))
        score = _safe_num(pick.get("score"))
        p1 = _safe_num(pred.get("team1_win_prob"), 50.0)
        p2 = _safe_num(pred.get("team2_win_prob"), 50.0)

        lines.append(
            f"{idx}. {team1} vs {team2} BO{bo} | "
            f"{p1:.0f}%x{p2:.0f}% | {side}@{odd:.2f}({bookmaker}) +{value_pct:.1f}% | sc:{score:.1f}"
        )

    lines.append("Resuma as oportunidades.")
    return "\n".join(lines)


def build_audit_prompt(summary: dict) -> str:
    wins = int(summary.get("wins", 0))
    losses = int(summary.get("losses", 0))
    pending = int(summary.get("pending", 0))
    accuracy = float(summary.get("accuracy", 0.0))
    run_date = str(summary.get("run_date", "?"))

    trend_7d = _safe_num(summary.get("trend_7d", 0.0))
    clv_30d = _safe_num(summary.get("avg_clv_30d", 0.0))

    lines = [
        f"Auditoria {run_date}: {wins}W {losses}L {pending}P | Acc:{accuracy:.1f}% | "
        f"Trend7d:{trend_7d:.1f}% | CLV30d:{clv_30d:+.2f}%"
    ]
    for item in summary.get("items", []) or []:
        status = str(item.get("outcome_status", "?"))
        team1 = str(item.get("team1_name", "?"))
        team2 = str(item.get("team2_name", "?"))
        pick = str(item.get("official_pick_winner_name", "?"))
        lines.append(f"{status}: {team1} vs {team2} pick:{pick}")
    lines.append("Analise e recomende.")
    return "\n".join(lines)


def build_explainability_prompt(
    match: dict,
    prediction: dict,
    top_features: list[tuple[str, float]],
) -> str:
    team1 = str(match.get("team1_name", "Team1"))
    team2 = str(match.get("team2_name", "Team2"))
    winner = team1 if int(prediction.get("predicted_winner", 1)) == 1 else team2
    p1 = float(prediction.get("team1_win_prob", 50.0))
    p2 = float(prediction.get("team2_win_prob", 50.0))
    feats = " | ".join(f"{name}:{weight:.4f}" for name, weight in top_features[:7])
    return (
        f"{team1} vs {team2} | Fav:{winner} ({p1:.1f}%x{p2:.1f}%)\n"
        f"Top features: {feats}\n"
        "Explique a predicao."
    )


def build_anomaly_prompt(
    match: dict,
    prediction: dict,
    analysis: dict,
) -> str:
    t1 = str(match.get("team1_name", "Team1"))
    t2 = str(match.get("team2_name", "Team2"))
    event = str(match.get("event_name", "?"))
    p1 = _safe_num(prediction.get("team1_win_prob"), 50.0)
    p2 = _safe_num(prediction.get("team2_win_prob"), 50.0)
    conf = _safe_num(prediction.get("confidence"), 50.0)
    o1 = _safe_num(analysis.get("odds_team1"), 0.0)
    o2 = _safe_num(analysis.get("odds_team2"), 0.0)

    best = None
    for vb in analysis.get("value_bets", []) if isinstance(analysis, dict) else []:
        if best is None or _safe_num(vb.get("value_pct")) > _safe_num(best.get("value_pct")):
            best = vb

    if best:
        vb_line = (
            f"best_value={best.get('side')} odd={_safe_num(best.get('odds')):.2f} "
            f"value={_safe_num(best.get('value_pct')):+.1f}% ev={_safe_num(best.get('expected_value')):+.2f}"
        )
    else:
        vb_line = "best_value=none"

    return (
        f"{t1} vs {t2} | {event}\n"
        f"probs={p1:.1f}/{p2:.1f} conf={conf:.1f}\n"
        f"odds={o1:.2f}/{o2:.2f}\n"
        f"{vb_line}\n"
        "Classifique anomalia em uma linha."
    )


def _safe_num(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
