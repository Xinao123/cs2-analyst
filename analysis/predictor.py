"""
Predictor — Modelo de ML para predição de partidas CS2.

Usa XGBoost (ou fallback LogisticRegression) para classificação binária.
Output: probabilidade de team1 vencer.
"""

import logging
from pathlib import Path

import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class Predictor:
    """Modelo preditivo de partidas CS2."""

    def __init__(self, config: dict):
        model_cfg = config.get("model", {})
        self.model_path = model_cfg.get("path", "data/model.joblib")
        self.min_confidence = model_cfg.get("min_confidence", 55.0)
        self.pipeline = None
        self._feature_names: list[str] = []
        self._is_trained = False

        # Tenta carregar modelo salvo
        self._load()

    def train(self, features_list: list[dict], labels: list[int]) -> dict:
        """
        Treina o modelo com dados históricos.

        Args:
            features_list: lista de dicts de features
            labels: lista de 0/1 (team2/team1 venceu)

        Returns:
            dict com métricas de treinamento
        """
        if len(features_list) < 50:
            logger.warning(f"[MODEL] Poucas amostras ({len(features_list)}), mínimo recomendado: 50")
            if len(features_list) < 20:
                return {"error": "Dados insuficientes para treinar"}

        # Converte features pra array
        self._feature_names = sorted(features_list[0].keys())
        X = np.array([[f.get(k, 0) for k in self._feature_names] for f in features_list])
        y = np.array(labels)

        # Substitui NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        logger.info(f"[MODEL] Treinando com {len(X)} amostras, {len(self._feature_names)} features")

        # Tenta XGBoost, fallback pra LogisticRegression
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
            )
            model_name = "XGBoost"
        except ImportError:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
            )
            model_name = "LogisticRegression"
            logger.info("[MODEL] XGBoost não disponível, usando LogisticRegression")

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

        # Cross-validation temporal (não embaralha — respeita cronologia)
        n_splits = min(5, len(X) // 20)
        if n_splits >= 2:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            cv_scores = cross_val_score(self.pipeline, X, y, cv=tscv, scoring="accuracy")
            cv_accuracy = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_accuracy = 0.0
            cv_std = 0.0

        # Treina no dataset completo
        self.pipeline.fit(X, y)
        self._is_trained = True

        # Métricas no training set (pra referência)
        y_pred = self.pipeline.predict(X)
        y_proba = self.pipeline.predict_proba(X)[:, 1]
        train_acc = accuracy_score(y, y_pred)
        train_logloss = log_loss(y, y_proba)
        train_brier = brier_score_loss(y, y_proba)

        # Feature importance
        importances = self._get_feature_importance()

        # Salva modelo
        self._save()

        metrics = {
            "model": model_name,
            "samples": len(X),
            "features": len(self._feature_names),
            "cv_accuracy": round(cv_accuracy * 100, 1),
            "cv_std": round(cv_std * 100, 1),
            "train_accuracy": round(train_acc * 100, 1),
            "train_logloss": round(train_logloss, 4),
            "train_brier": round(train_brier, 4),
            "top_features": importances[:10],
        }

        logger.info(
            f"[MODEL] {model_name} treinado: "
            f"CV accuracy={metrics['cv_accuracy']:.1f}% (±{metrics['cv_std']:.1f}%), "
            f"train accuracy={metrics['train_accuracy']:.1f}%"
        )

        return metrics

    def predict(self, features: dict) -> dict | None:
        """
        Prevê resultado de uma partida.

        Args:
            features: dict de features da partida

        Returns:
            dict com probabilidades ou None se modelo não treinado
        """
        if not self._is_trained or self.pipeline is None:
            logger.warning("[MODEL] Modelo não treinado")
            return None

        X = np.array([[features.get(k, 0) for k in self._feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        proba = self.pipeline.predict_proba(X)[0]
        team1_prob = float(proba[1])  # prob de label=1 (team1 vence)
        team2_prob = float(proba[0])  # prob de label=0 (team2 vence)

        confidence = max(team1_prob, team2_prob) * 100

        return {
            "team1_win_prob": round(team1_prob * 100, 1),
            "team2_win_prob": round(team2_prob * 100, 1),
            "confidence": round(confidence, 1),
            "predicted_winner": 1 if team1_prob > team2_prob else 2,
        }

    def _get_feature_importance(self) -> list[tuple[str, float]]:
        """Retorna features mais importantes."""
        if not self.pipeline:
            return []

        model = self.pipeline.named_steps["model"]

        try:
            # XGBoost
            importances = model.feature_importances_
        except AttributeError:
            try:
                # LogisticRegression
                importances = np.abs(model.coef_[0])
            except AttributeError:
                return []

        pairs = list(zip(self._feature_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [(name, round(float(imp), 4)) for name, imp in pairs]

    def _save(self):
        """Salva modelo e metadata."""
        if not self.pipeline:
            return
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"pipeline": self.pipeline, "feature_names": self._feature_names},
            self.model_path,
        )
        logger.info(f"[MODEL] Salvo em {self.model_path}")

    def _load(self):
        """Carrega modelo salvo."""
        path = Path(self.model_path)
        if path.exists():
            try:
                data = joblib.load(self.model_path)
                self.pipeline = data["pipeline"]
                self._feature_names = data["feature_names"]
                self._is_trained = True
                logger.info(f"[MODEL] Carregado de {self.model_path}")
            except Exception as e:
                logger.warning(f"[MODEL] Erro ao carregar: {e}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained
