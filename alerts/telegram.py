"""
Telegram Alerts - Envia analises e value bets via Telegram.
"""

import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class Notifier:
    """Notificacoes via Telegram."""

    def __init__(self, config: dict):
        tg = config.get("telegram", {})
        self.enabled = tg.get("enabled", False)
        self.bot_token = tg.get("bot_token", "")
        self.chat_id = tg.get("chat_id", "")

        if self.enabled and self.bot_token.startswith("SEU_"):
            self.enabled = False

    def _send(self, text: str):
        if not self.enabled:
            return
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
            if resp.status_code >= 400:
                logger.warning(
                    "[TG] Falha status=%s body=%s",
                    resp.status_code,
                    _safe_text(resp.text)[:500],
                )
        except Exception as e:
            logger.warning(f"[TG] Falha: {e}")

    def startup(self, stats: dict):
        now = datetime.now().strftime("%d/%m/%Y %H:%M")
        self._send(
            f"🎮 <b>CS2 Analyst Bot Iniciado</b>\n"
            f"{'━' * 26}\n"
            f"📅 {now}\n"
            f"👥 {stats.get('teams', 0)} times no banco\n"
            f"🎯 {stats.get('completed_matches', 0)} partidas historicas\n"
            f"📋 {stats.get('upcoming_matches', 0)} partidas futuras\n"
            f"🤖 Analisando..."
        )

    def value_bet_alert(self, report: str, match: dict):
        """Envia alerta de value bet."""
        now = datetime.now().strftime("%H:%M")
        self._send(
            f"🔔 <b>VALUE BET DETECTADO</b> - {now}\n"
            f"{'━' * 30}\n\n"
            f"{_html_escape(report)}"
        )

    def prediction_alert(self, report: str, match: dict):
        """Envia predicao sem value (informativo)."""
        self._send(
            f"📊 <b>Analise de partida</b>\n"
            f"{'━' * 30}\n\n"
            f"{_html_escape(report)}"
        )

    def top_picks_alert(
        self,
        picks: list[dict],
        total_candidates: int,
        requested_top: int,
        candidates_with_odds: int = 0,
    ):
        """Envia ranking consolidado das melhores oportunidades."""
        now = datetime.now().strftime("%d/%m %H:%M")
        if not picks:
            self._send(
                "\n".join(
                    [
                        f"🏆 <b>Top {requested_top} apostas do ciclo</b>",
                        f"{'━' * 30}",
                        f"📅 {now}",
                        f"🔎 Candidatas: {total_candidates} (com odds: {candidates_with_odds})",
                        "❌ Sem oportunidades de value neste ciclo",
                    ]
                )
            )
            return

        lines = [
            f"🏆 <b>Top {len(picks)} apostas do ciclo</b>",
            f"{'━' * 30}",
            f"📅 {now}",
            f"🔎 Candidatas: {total_candidates} (com odds: {candidates_with_odds}, top solicitado: {requested_top})",
            "",
        ]

        for idx, item in enumerate(picks, start=1):
            match = item.get("match", {})
            pred = item.get("prediction", {})
            analysis = item.get("analysis", {})
            score = float(item.get("score", 0.0))

            t1 = _safe_text(match.get("team1_name", "Team 1"))
            t2 = _safe_text(match.get("team2_name", "Team 2"))
            event = _safe_text(match.get("event_name", ""))
            when = _format_short_datetime(match.get("date"))
            p1 = float(pred.get("team1_win_prob", 50.0))
            p2 = float(pred.get("team2_win_prob", 50.0))
            conf = float(pred.get("confidence", 50.0))
            side = t1 if pred.get("predicted_winner") == 1 else t2

            best_vb = item.get("best_vb") or {}
            if not best_vb:
                value_bets = analysis.get("value_bets", [])
                if value_bets:
                    best_vb = max(value_bets, key=lambda vb: float(vb.get("value_pct", 0.0)))

            odd = float(best_vb.get("odds", 0.0))
            value_pct = float(best_vb.get("value_pct", 0.0))
            ev = float(best_vb.get("expected_value", 0.0))
            bookmaker = _safe_text(best_vb.get("bookmaker", "N/D"))

            lines.extend(
                [
                    f"{idx}. <b>{_html_escape(t1)} vs {_html_escape(t2)}</b>",
                    f"   🗓 {_html_escape(when)}",
                    f"   🏆 {_html_escape(event) if event else '-'}",
                    f"   🎯 Pick: {_html_escape(side)} ({p1:.1f}% x {p2:.1f}%, conf {conf:.1f}%)",
                    f"   💵 Odd: {odd:.2f} ({_html_escape(bookmaker)})",
                    f"   📈 Value: +{value_pct:.1f}% | EV: R$ {ev:.2f}",
                    f"   📌 Score: {score:.2f}",
                    "",
                ]
            )

        self._send("\n".join(lines).rstrip())

    def daily_summary(self, summary: dict):
        """Resumo diario de performance."""
        total = summary.get("total_predictions", 0)
        correct = summary.get("correct", 0)
        accuracy = (correct / total * 100) if total > 0 else 0
        profit = summary.get("total_profit", 0)
        roi = summary.get("roi", 0)

        self._send(
            f"📈 <b>Resumo do dia</b>\n"
            f"{'━' * 26}\n"
            f"🎯 Acerto: {correct}/{total} ({accuracy:.0f}%)\n"
            f"💰 Lucro: R$ {profit:+.2f}\n"
            f"📊 ROI: {roi:+.1f}%\n"
            f"🏦 Banca: R$ {summary.get('bankroll', 0):.2f}"
        )

    def model_trained(self, metrics: dict):
        """Notifica sobre treinamento do modelo."""
        self._send(
            f"🧠 <b>Modelo treinado</b>\n"
            f"{'━' * 26}\n"
            f"📊 Modelo: {metrics.get('model', 'N/A')}\n"
            f"📏 Amostras: {metrics.get('samples', 0)}\n"
            f"🎯 CV Accuracy: {metrics.get('cv_accuracy', 0):.1f}% "
            f"(±{metrics.get('cv_std', 0):.1f}%)\n"
            f"📈 Train Accuracy: {metrics.get('train_accuracy', 0):.1f}%\n"
            f"🔝 Top features: {', '.join(f[0] for f in metrics.get('top_features', [])[:5])}"
        )

    def error(self, message: str):
        self._send(f"❌ <b>Erro:</b> {message}")


def _html_escape(text: str) -> str:
    """Escapa caracteres HTML mantendo tags que ja temos."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _safe_text(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _format_short_datetime(value) -> str:
    text = _safe_text(value)
    if not text:
        return "Data indefinida"

    parsed = None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo:
            parsed = parsed.astimezone().replace(tzinfo=None)
    except ValueError:
        pass

    if parsed is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue

    if parsed is None:
        return text
    return parsed.strftime("%d/%m %H:%M")
