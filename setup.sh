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

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " PASSO A PASSO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  1. Edite data/config.yaml:"
echo "     - Telegram: bot_token e chat_id"
echo "     - Ajuste min_value_pct e min_confidence"
echo ""
echo "  2. Instale dependências:"
echo "     pip install -r requirements.txt"
echo ""
echo "  3. Colete dados históricos (roda ~5-10 min):"
echo "     python -m scripts.backfill"
echo ""
echo "  4. Treine o modelo:"
echo "     python -m scripts.backtest"
echo ""
echo "  5. Rode o bot:"
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
