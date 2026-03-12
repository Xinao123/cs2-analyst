"""
Telegram Alerts — Envia análises e value bets via Telegram.
"""

import logging
from datetime import datetime

import requests

logger = logging.getLogger(__name__)


class Notifier:
    """Notificações via Telegram."""

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
            requests.post(
                url,
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
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
            f"🎯 {stats.get('completed_matches', 0)} partidas históricas\n"
            f"📋 {stats.get('upcoming_matches', 0)} partidas futuras\n"
            f"🤖 Analisando..."
        )

    def value_bet_alert(self, report: str, match: dict):
        """Envia alerta de value bet."""
        now = datetime.now().strftime("%H:%M")
        self._send(
            f"🔔 <b>VALUE BET DETECTADO</b> — {now}\n"
            f"{'━' * 30}\n\n"
            f"{_html_escape(report)}"
        )

    def prediction_alert(self, report: str, match: dict):
        """Envia predição sem value (informativo)."""
        self._send(
            f"📊 <b>Análise de partida</b>\n"
            f"{'━' * 30}\n\n"
            f"{_html_escape(report)}"
        )

    def daily_summary(self, summary: dict):
        """Resumo diário de performance."""
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
    """Escapa caracteres HTML mantendo tags que já temos."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
