#!/bin/bash
# ============================================================
# CS2 Analyst Bot — Setup
# ============================================================

echo "🎮 CS2 Analyst Bot — Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

mkdir -p data

if [ ! -f data/config.yaml ]; then
    cp config.yaml data/config.yaml
    echo "✅ Config copiado para data/config.yaml"
else
    echo "📝 data/config.yaml já existe"
fi

if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "✅ .env criado a partir de .env.example"
else
    echo "📝 .env já existe (ou .env.example ausente)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " PASSO A PASSO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Edite data/config.yaml:"
echo "     - Telegram: bot_token e chat_id"
echo "     - Ajuste min_value_pct e min_confidence"
echo ""
echo "  2. Edite .env (persistente no servidor):"
echo "     - PANDASCORE_API_TOKEN"
echo "     - ODDSPAPI_API_KEY"
echo "     - DEEPSEEK_API_KEY"
echo "     - TELEGRAM_BOT_TOKEN"
echo "     - TELEGRAM_CHAT_ID"
echo ""
echo "  3. Instale dependências:"
echo "     pip install -r requirements.txt"
echo ""
echo "  4. Colete dados históricos (roda ~5-10 min):"
echo "     python -m scripts.backfill"
echo ""
echo "  5. Treine o modelo:"
echo "     python -m scripts.backtest"
echo ""
echo "  6. Rode o bot:"
echo "     python main.py --once     # testa uma vez"
echo "     python main.py            # roda contínuo"
echo ""
echo "  Ou com Docker:"
echo "     docker compose up -d"
echo "     docker logs -f cs2-analyst"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " COMANDOS ÚTEIS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  python main.py --stats       # Stats do banco"
echo "  python main.py --train       # Re-treina modelo"
echo "  python main.py --once        # Analisa e sai"
echo "  cat data/cs2_analyst.db      # Banco SQLite"
echo ""
