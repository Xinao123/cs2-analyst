"""
Predictor - Modelo de ML para predicao de partidas CS2.

Usa XGBoost (ou fallback LogisticRegression) para classificacao binaria.
Output: probabilidade de team1 vencer.
"""

import logging
import math
from datetime import datetime, timedelta
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
        self.holdout_days = max(3, int(model_cfg.get("holdout_days", 14)))
        self.recency_half_life_days = max(7.0, float(model_cfg.get("recency_half_life_days", 120)))
        self.min_train_samples = max(20, int(model_cfg.get("min_train_samples", 120)))
        self.min_class_samples = max(5, int(model_cfg.get("min_class_samples", 30)))
        self.confidence_auto_tune = bool(model_cfg.get("confidence_auto_tune", True))
        self.confidence_grid = _parse_confidence_grid(model_cfg.get("confidence_grid", [60, 65, 70, 75, 80]))
        self.coverage_min_holdout = _clamp_float(model_cfg.get("coverage_min_holdout", 0.15), 0.01, 1.0)

        self.pipeline = None
        self.proba_model = None
        self._feature_names: list[str] = []
        self._is_trained = False
        self._is_calibrated = False

        # Tenta carregar modelo salvo
        self._load()

    def train(
        self,
        features_list: list[dict],
        labels: list[int],
        match_dates: list[str] | None = None,
        sample_weights: list[float] | np.ndarray | None = None,
    ) -> dict:
        """
        Treina o modelo com dados historicos.

        Args:
            features_list: lista de dicts de features
            labels: lista de 0/1 (team2/team1 venceu)

        Returns:
            dict com metricas de treinamento
        """
        if len(features_list) < self.min_train_samples:
            logger.warning(
                "[MODEL] Poucas amostras (%s), minimo recomendado: %s",
                len(features_list),
                self.min_train_samples,
            )
            if len(features_list) < max(20, self.min_train_samples // 2):
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
        class_1 = int(np.sum(y == 1))
        class_0 = int(np.sum(y == 0))
        if class_1 < self.min_class_samples or class_0 < self.min_class_samples:
            return {
                "error": (
                    "Classes insuficientes para treino robusto "
                    f"(team1={class_1}, team2={class_0}, min={self.min_class_samples})"
                )
            }

        logger.info(f"[MODEL] Treinando com {len(X)} amostras, {len(self._feature_names)} features")

        # Tenta XGBoost, fallback pra LogisticRegression
        model, model_name = self._build_model(y)

        self.pipeline = self._build_pipeline(model)
        self.proba_model = self.pipeline
        self._is_calibrated = False

        recency_weights = _build_recency_weights(
            match_dates=match_dates,
            total_samples=len(X),
            half_life_days=self.recency_half_life_days,
        )
        quality_weights = _coerce_sample_weights(sample_weights, len(X))
        sample_weight = np.clip(recency_weights * quality_weights, 0.01, 10.0)
        weight_mean = float(np.mean(sample_weight)) if len(sample_weight) > 0 else 1.0
        if weight_mean > 0:
            sample_weight = sample_weight / weight_mean

        train_idx, holdout_idx = _temporal_split_indices(
            match_dates=match_dates,
            total_samples=len(X),
            holdout_days=self.holdout_days,
        )
        if len(train_idx) < max(20, self.min_train_samples // 2):
            return {"error": "Conjunto de treino insuficiente apos holdout temporal"}

        # Cross-validation temporal (nao embaralha - respeita cronologia)
        X_train = X[train_idx]
        y_train = y[train_idx]
        w_train = sample_weight[train_idx]
        n_splits = min(5, len(X_train) // 20)
        if n_splits >= 2:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            try:
                cv_scores = cross_val_score(self.pipeline, X_train, y_train, cv=tscv, scoring="accuracy")
                cv_accuracy = cv_scores.mean()
                cv_std = cv_scores.std()
            except Exception as exc:
                logger.warning("[MODEL] CV temporal falhou: %s", exc)
                cv_accuracy = 0.0
                cv_std = 0.0
        else:
            cv_accuracy = 0.0
            cv_std = 0.0

        # Treina no conjunto train (expanding + holdout temporal no fim)
        self._fit_pipeline(X_train, y_train, w_train)
        self._fit_calibrator(X_train, y_train, n_splits)
        self._is_trained = True

        # Metricas no training set (referencia)
        y_train_proba = self._predict_proba_vector(X_train)
        y_train_pred = (y_train_proba >= 0.5).astype(int)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_logloss = _safe_logloss(y_train, y_train_proba)
        train_brier = brier_score_loss(y_train, y_train_proba)

        holdout_acc = 0.0
        holdout_logloss = 0.0
        holdout_brier = 0.0
        holdout_size = 0
        tuned_confidence = float(self.min_confidence)
        tuned_precision = 0.0
        tuned_coverage = 0.0
        if len(holdout_idx) > 0:
            X_holdout = X[holdout_idx]
            y_holdout = y[holdout_idx]
            holdout_proba = self._predict_proba_vector(X_holdout)
            holdout_pred = (holdout_proba >= 0.5).astype(int)
            holdout_acc = accuracy_score(y_holdout, holdout_pred)
            holdout_logloss = _safe_logloss(y_holdout, holdout_proba)
            holdout_brier = brier_score_loss(y_holdout, holdout_proba)
            holdout_size = len(y_holdout)
            if self.confidence_auto_tune:
                tuned_confidence, tuned_precision, tuned_coverage = self._tune_confidence_threshold(
                    y_true=y_holdout,
                    y_proba=holdout_proba,
                )
                self.min_confidence = tuned_confidence

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
            "holdout_size": int(holdout_size),
            "holdout_accuracy": round(holdout_acc * 100, 1),
            "holdout_logloss": round(holdout_logloss, 4),
            "holdout_brier": round(holdout_brier, 4),
            "recommended_min_confidence": round(float(self.min_confidence), 1),
            "threshold_precision": round(float(tuned_precision) * 100, 1),
            "threshold_coverage": round(float(tuned_coverage) * 100, 1),
            "calibrated": self._is_calibrated,
            "sample_weight_mean": round(float(np.mean(sample_weight)), 3) if len(sample_weight) else 0.0,
            "sample_weight_min": round(float(np.min(sample_weight)), 3) if len(sample_weight) else 0.0,
            "sample_weight_max": round(float(np.max(sample_weight)), 3) if len(sample_weight) else 0.0,
            "top_features": importances[:10],
        }

        logger.info(
            f"[MODEL] {model_name} treinado: "
            f"CV accuracy={metrics['cv_accuracy']:.1f}% (+/-{metrics['cv_std']:.1f}%), "
            f"train accuracy={metrics['train_accuracy']:.1f}%"
        )
        if holdout_size > 0:
            logger.info(
                "[MODEL] Holdout temporal: acc=%s%% logloss=%s brier=%s (n=%s)",
                metrics["holdout_accuracy"],
                metrics["holdout_logloss"],
                metrics["holdout_brier"],
                holdout_size,
            )
        if self.confidence_auto_tune:
            logger.info(
                "[MODEL] Threshold recomendado de confianca: %.1f%% (precision=%s%% coverage=%s%%)",
                float(self.min_confidence),
                metrics["threshold_precision"],
                metrics["threshold_coverage"],
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

    def _fit_pipeline(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        if sample_weight is None:
            self.pipeline.fit(X, y)
            return
        try:
            self.pipeline.fit(X, y, model__sample_weight=sample_weight)
        except TypeError:
            self.pipeline.fit(X, y)

    def _tune_confidence_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float, float]:
        if len(y_true) == 0:
            return float(self.min_confidence), 0.0, 0.0

        best_thr = float(self.min_confidence)
        best_precision = -1.0
        best_coverage = -1.0

        y_pred = (y_proba >= 0.5).astype(int)
        confidence = np.maximum(y_proba, 1.0 - y_proba) * 100.0

        for thr in self.confidence_grid:
            mask = confidence >= float(thr)
            coverage = float(np.mean(mask))
            if coverage < self.coverage_min_holdout:
                continue

            selected = int(np.sum(mask))
            if selected == 0:
                continue
            precision = float(np.mean(y_pred[mask] == y_true[mask]))

            if precision > best_precision or (
                abs(precision - best_precision) < 1e-9 and coverage > best_coverage
            ):
                best_thr = float(thr)
                best_precision = precision
                best_coverage = coverage

        if best_precision < 0:
            return float(self.min_confidence), 0.0, 0.0
        return best_thr, best_precision, best_coverage

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
                "recommended_min_confidence": float(self.min_confidence),
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
                self.min_confidence = float(data.get("recommended_min_confidence", self.min_confidence))
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


def _safe_logloss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    try:
        return float(log_loss(y_true, np.clip(y_proba, 1e-6, 1 - 1e-6)))
    except ValueError:
        return 0.0


def _parse_confidence_grid(values) -> list[float]:
    if not isinstance(values, list):
        values = [60, 65, 70, 75, 80]
    out = []
    for value in values:
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    out = [max(50.0, min(99.0, v)) for v in out]
    out = sorted(set(out))
    return out or [60.0, 65.0, 70.0, 75.0, 80.0]


def _build_recency_weights(
    match_dates: list[str] | None,
    total_samples: int,
    half_life_days: float,
) -> np.ndarray:
    if not match_dates or len(match_dates) != total_samples:
        return np.ones(total_samples, dtype=float)

    parsed = [_parse_match_dt(d) for d in match_dates]
    if any(dt is None for dt in parsed):
        return np.ones(total_samples, dtype=float)

    max_dt = max(parsed)
    if max_dt is None:
        return np.ones(total_samples, dtype=float)

    hl = max(1.0, float(half_life_days))
    weights = []
    for dt in parsed:
        delta_days = max(0.0, (max_dt - dt).total_seconds() / 86400.0)
        weight = math.pow(0.5, delta_days / hl)
        weights.append(max(0.1, min(1.0, weight)))
    return np.array(weights, dtype=float)


def _coerce_sample_weights(
    sample_weights: list[float] | np.ndarray | None,
    total_samples: int,
) -> np.ndarray:
    if sample_weights is None:
        return np.ones(total_samples, dtype=float)

    arr = np.asarray(sample_weights, dtype=float).reshape(-1)
    if arr.size != total_samples:
        return np.ones(total_samples, dtype=float)
    arr = np.nan_to_num(arr, nan=1.0, posinf=1.0, neginf=1.0)
    arr = np.clip(arr, 0.05, 10.0)
    return arr


def _temporal_split_indices(
    match_dates: list[str] | None,
    total_samples: int,
    holdout_days: int,
) -> tuple[np.ndarray, np.ndarray]:
    if total_samples <= 1:
        idx = np.arange(total_samples)
        return idx, np.array([], dtype=int)

    if match_dates and len(match_dates) == total_samples:
        parsed = [_parse_match_dt(d) for d in match_dates]
        if all(dt is not None for dt in parsed):
            max_dt = max(parsed)
            cutoff = max_dt - timedelta(days=max(1, int(holdout_days)))
            holdout_idx = np.array([i for i, dt in enumerate(parsed) if dt >= cutoff], dtype=int)
            train_idx = np.array([i for i, dt in enumerate(parsed) if dt < cutoff], dtype=int)
            if len(holdout_idx) >= 10 and len(train_idx) >= 20:
                return train_idx, holdout_idx

    # Fallback temporal: ultimos ~15% como holdout
    holdout_size = max(10, int(total_samples * 0.15))
    holdout_size = min(holdout_size, max(1, total_samples // 3))
    cut = total_samples - holdout_size
    train_idx = np.arange(max(0, cut))
    holdout_idx = np.arange(max(0, cut), total_samples)
    return train_idx, holdout_idx


def _parse_match_dt(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d-%m-%Y %H:%M:%S", "%d-%m-%Y"):
                try:
                    dt = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
            else:
                return None
    if dt.tzinfo:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt
