"""
Predictor - Modelo de ML para predicao de partidas CS2.

Usa XGBoost (ou fallback LogisticRegression) para classificacao binaria.
Output: probabilidade de team1 vencer.
"""

import logging
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class Predictor:
    """Modelo preditivo de partidas CS2."""

    def __init__(self, config: dict):
        model_cfg = config.get("model", {})
        self.model_path = model_cfg.get("path", "data/model.joblib")
        self.min_confidence = model_cfg.get("min_confidence", 55.0)

        # Robustez/calibracao
        self.enable_calibration = bool(model_cfg.get("enable_calibration", True))
        self.calibration_method = str(model_cfg.get("calibration_method", "sigmoid")).strip() or "sigmoid"
        self.calibration_min_samples = max(40, int(model_cfg.get("calibration_min_samples", 80)))
        self.calibration_cv_splits = max(2, int(model_cfg.get("calibration_cv_splits", 3)))

        # Separacao das probabilidades em inferencia
        self.prediction_temperature = _clamp_float(model_cfg.get("prediction_temperature", 0.9), 0.5, 1.5)
        self.low_data_rank_blend = _clamp_float(model_cfg.get("low_data_rank_blend", 0.35), 0.0, 0.9)
        self.rank_prior_scale = max(1.0, float(model_cfg.get("rank_prior_scale", 12.0)))

        self.pipeline = None
        self.proba_model = None
        self._feature_names: list[str] = []
        self._is_trained = False
        self._is_calibrated = False

        # Tenta carregar modelo salvo
        self._load()

    def train(self, features_list: list[dict], labels: list[int]) -> dict:
        """
        Treina o modelo com dados historicos.

        Args:
            features_list: lista de dicts de features
            labels: lista de 0/1 (team2/team1 venceu)

        Returns:
            dict com metricas de treinamento
        """
        if len(features_list) < 50:
            logger.warning(f"[MODEL] Poucas amostras ({len(features_list)}), minimo recomendado: 50")
            if len(features_list) < 20:
                return {"error": "Dados insuficientes para treinar"}

        self._feature_names = sorted({k for row in features_list for k in row.keys()})
        if not self._feature_names:
            return {"error": "Sem features para treinar"}

        X = np.array([[f.get(k, 0) for k in self._feature_names] for f in features_list])
        y = np.array(labels)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0).astype(int)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return {"error": "Labels com apenas uma classe; treino cancelado"}

        logger.info(f"[MODEL] Treinando com {len(X)} amostras, {len(self._feature_names)} features")

        # Tenta XGBoost, fallback pra LogisticRegression
        model, model_name = self._build_model(y)

        self.pipeline = self._build_pipeline(model)
        self.proba_model = self.pipeline
        self._is_calibrated = False

        # Cross-validation temporal (nao embaralha - respeita cronologia)
        n_splits = min(5, len(X) // 20)
        if n_splits >= 2:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            try:
                cv_scores = cross_val_score(self.pipeline, X, y, cv=tscv, scoring="accuracy")
                cv_accuracy = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as exc:
                logger.warning("[MODEL] CV temporal falhou: %s", exc)
                cv_accuracy = 0.0
                cv_std = 0.0
        else:
            cv_accuracy = 0.0
            cv_std = 0.0

        # Treina no dataset completo
        self.pipeline.fit(X, y)
        self._fit_calibrator(X, y, n_splits)
        self._is_trained = True

        # Metricas no training set (referencia)
        y_proba = self._predict_proba_vector(X)
        y_pred = (y_proba >= 0.5).astype(int)
        train_acc = accuracy_score(y, y_pred)
        train_logloss = log_loss(y, np.clip(y_proba, 1e-6, 1 - 1e-6))
        train_brier = brier_score_loss(y, y_proba)

        importances = self._get_feature_importance()

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
            "calibrated": self._is_calibrated,
            "top_features": importances[:10],
        }

        logger.info(
            f"[MODEL] {model_name} treinado: "
            f"CV accuracy={metrics['cv_accuracy']:.1f}% (+/-{metrics['cv_std']:.1f}%), "
            f"train accuracy={metrics['train_accuracy']:.1f}%"
        )

        return metrics

    def predict(self, features: dict) -> dict | None:
        """
        Preve resultado de uma partida.

        Args:
            features: dict de features da partida

        Returns:
            dict com probabilidades ou None se modelo nao treinado
        """
        if not self._is_trained or self.pipeline is None:
            logger.warning("[MODEL] Modelo nao treinado")
            return None

        X = np.array([[features.get(k, 0) for k in self._feature_names]])
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        raw_team1_prob = self._predict_proba_vector(X)[0]
        team1_prob = self._apply_low_data_rank_prior(raw_team1_prob, features)
        team1_prob = _apply_temperature(team1_prob, self.prediction_temperature)
        team1_prob = _clamp_float(team1_prob, 0.01, 0.99)
        team2_prob = 1.0 - team1_prob

        confidence = max(team1_prob, team2_prob) * 100

        return {
            "team1_win_prob": round(team1_prob * 100, 1),
            "team2_win_prob": round(team2_prob * 100, 1),
            "confidence": round(confidence, 1),
            "predicted_winner": 1 if team1_prob > team2_prob else 2,
        }

    def _build_model(self, y: np.ndarray):
        try:
            from xgboost import XGBClassifier

            pos = max(1, int(np.sum(y == 1)))
            neg = max(1, int(np.sum(y == 0)))
            model = XGBClassifier(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_weight=2,
                reg_lambda=1.0,
                reg_alpha=0.05,
                eval_metric="logloss",
                random_state=42,
                scale_pos_weight=(neg / pos),
                n_jobs=1,
                verbosity=0,
            )
            return model, "XGBoost"
        except ImportError:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                max_iter=1200,
                random_state=42,
                C=1.0,
                class_weight="balanced",
            )
            logger.info("[MODEL] XGBoost nao disponivel, usando LogisticRegression")
            return model, "LogisticRegression"

    def _build_pipeline(self, model):
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", model),
        ])

    def _fit_calibrator(self, X: np.ndarray, y: np.ndarray, n_splits: int):
        if not self.enable_calibration:
            return
        if len(X) < self.calibration_min_samples:
            return
        if n_splits < 2:
            return

        cal_splits = min(self.calibration_cv_splits, n_splits)
        if cal_splits < 2:
            return

        try:
            from sklearn.base import clone

            base_estimator = clone(self.pipeline)
            calibrator = CalibratedClassifierCV(
                estimator=base_estimator,
                method=self.calibration_method,
                cv=TimeSeriesSplit(n_splits=cal_splits),
            )
            calibrator.fit(X, y)
            self.proba_model = calibrator
            self._is_calibrated = True
            logger.info("[MODEL] Calibracao de probabilidade ativada (%s)", self.calibration_method)
        except Exception as exc:
            logger.warning("[MODEL] Calibracao indisponivel, mantendo probabilidade base: %s", exc)

    def _predict_proba_vector(self, X: np.ndarray) -> np.ndarray:
        model = self.proba_model or self.pipeline
        if model is None:
            return np.zeros(len(X))

        proba = model.predict_proba(X)
        if proba.ndim != 2 or proba.shape[1] < 2:
            return np.zeros(len(X))
        return np.clip(proba[:, 1].astype(float), 1e-6, 1 - 1e-6)

    def _apply_low_data_rank_prior(self, team1_prob: float, features: dict) -> float:
        if self.low_data_rank_blend <= 0:
            return team1_prob

        t1_games = max(0.0, float(features.get("team1_matches_played", 0)))
        t2_games = max(0.0, float(features.get("team2_matches_played", 0)))
        h2h_games = max(0.0, float(features.get("h2h_matches", 0)))

        # Menos jogos historicos => maior peso para prior de ranking.
        coverage = min(1.0, min(t1_games, t2_games) / 8.0)
        h2h_coverage = min(1.0, h2h_games / 5.0)
        info_score = max(coverage, h2h_coverage)
        blend = self.low_data_rank_blend * (1.0 - info_score)
        if blend <= 0:
            return team1_prob

        rank_diff = _clamp_float(features.get("ranking_diff", 0.0), -200.0, 200.0)
        rank_prior = 1.0 / (1.0 + math.exp(-(rank_diff / self.rank_prior_scale)))
        return ((1.0 - blend) * team1_prob) + (blend * rank_prior)

    def _get_feature_importance(self) -> list[tuple[str, float]]:
        """Retorna features mais importantes."""
        if not self.pipeline:
            return []

        model = self.pipeline.named_steps["model"]

        try:
            importances = model.feature_importances_
        except AttributeError:
            try:
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
            {
                "pipeline": self.pipeline,
                "proba_model": self.proba_model,
                "feature_names": self._feature_names,
                "is_calibrated": self._is_calibrated,
            },
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
                self.proba_model = data.get("proba_model", self.pipeline)
                self._feature_names = data["feature_names"]
                self._is_calibrated = bool(data.get("is_calibrated", False))
                self._is_trained = True
                logger.info(f"[MODEL] Carregado de {self.model_path}")
            except Exception as e:
                logger.warning(f"[MODEL] Erro ao carregar: {e}")

    @property
    def is_trained(self) -> bool:
        return self._is_trained


def _clamp_float(value, low: float, high: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        val = low
    return max(low, min(high, val))


def _apply_temperature(prob: float, temperature: float) -> float:
    p = _clamp_float(prob, 1e-6, 1.0 - 1e-6)
    t = _clamp_float(temperature, 0.5, 2.0)
    if abs(t - 1.0) < 1e-9:
        return p

    logit = math.log(p / (1.0 - p))
    adjusted = 1.0 / (1.0 + math.exp(-(logit / t)))
    return _clamp_float(adjusted, 1e-6, 1.0 - 1e-6)
