"""
Predictor - Modelo de ML para predicao de partidas CS2.

Usa XGBoost (ou fallback LogisticRegression) para classificacao binaria.
Output: probabilidade de team1 vencer.
"""

from __future__ import annotations

import itertools
import logging
import math
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
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

        # Robustez temporal
        self.holdout_days = max(3, int(model_cfg.get("holdout_days", 14)))
        self.recency_half_life_days = max(7.0, float(model_cfg.get("recency_half_life_days", 120)))
        self.min_train_samples = max(20, int(model_cfg.get("min_train_samples", 120)))
        self.min_class_samples = max(5, int(model_cfg.get("min_class_samples", 30)))
        self.confidence_auto_tune = bool(model_cfg.get("confidence_auto_tune", True))
        self.confidence_grid = _parse_confidence_grid(model_cfg.get("confidence_grid", [60, 65, 70, 75, 80]))
        self.coverage_min_holdout = _clamp_float(model_cfg.get("coverage_min_holdout", 0.15), 0.01, 1.0)

        # ML improvements v2
        self.enable_hyperparam_tuning = bool(model_cfg.get("enable_hyperparam_tuning", False))
        self.tuning_max_combinations = max(1, int(model_cfg.get("tuning_max_combinations", 50)))
        self.enable_ensemble = bool(model_cfg.get("enable_ensemble", False))

        self.pipeline = None
        self.proba_model = None
        self.ensemble_models: list[tuple[str, object]] = []
        self.primary_model_name = ""
        self.tuning_summary: dict = {}
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

        logger.info("[MODEL] Treinando com %s amostras, %s features", len(X), len(self._feature_names))

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

        X_train = X[train_idx]
        y_train = y[train_idx]
        w_train = sample_weight[train_idx]

        model, model_name, tuning_meta = self._select_primary_model(X_train, y_train, y)
        self.primary_model_name = model_name
        self.tuning_summary = tuning_meta

        self.pipeline = self._build_pipeline(model)
        self.proba_model = self.pipeline
        self._is_calibrated = False
        self.ensemble_models = []

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

        self._fit_pipeline(self.pipeline, X_train, y_train, w_train)
        self._fit_calibrator(X_train, y_train, n_splits)

        if self.enable_ensemble:
            self._fit_ensemble_members(X_train, y_train, w_train, y)

        self._is_trained = True

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
            "hyperparam_tuning": bool(self.enable_hyperparam_tuning),
            "tuning_evaluated": int(self.tuning_summary.get("evaluated", 0)),
            "tuning_best_brier": round(float(self.tuning_summary.get("best_brier", 0.0)), 4),
            "ensemble_enabled": bool(self.enable_ensemble),
            "ensemble_members": len(self.ensemble_models),
        }

        logger.info(
            "[MODEL] %s treinado: CV accuracy=%s%% (+/-%s%%), train accuracy=%s%%",
            model_name,
            metrics["cv_accuracy"],
            metrics["cv_std"],
            metrics["train_accuracy"],
        )
        if holdout_size > 0:
            logger.info(
                "[MODEL] Holdout temporal: acc=%s%% logloss=%s brier=%s (n=%s)",
                metrics["holdout_accuracy"],
                metrics["holdout_logloss"],
                metrics["holdout_brier"],
                holdout_size,
            )
        if self.enable_hyperparam_tuning:
            logger.info(
                "[MODEL] Tuning temporal (brier): avaliadas=%s melhor=%s",
                metrics["tuning_evaluated"],
                metrics["tuning_best_brier"],
            )
        if self.enable_ensemble:
            logger.info("[MODEL] Ensemble ativo com %s membro(s)", len(self.ensemble_models))
        if self.confidence_auto_tune:
            logger.info(
                "[MODEL] Threshold recomendado de confianca: %.1f%% (precision=%s%% coverage=%s%%)",
                float(self.min_confidence),
                metrics["threshold_precision"],
                metrics["threshold_coverage"],
            )

        return metrics

    def predict(self, features: dict) -> dict | None:
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

    def _select_primary_model(self, X_train: np.ndarray, y_train: np.ndarray, y_full: np.ndarray):
        if not self.enable_hyperparam_tuning:
            model, name = self._build_default_model(y_full)
            return model, name, {"evaluated": 0, "best_brier": 0.0}

        tune_X = X_train[-6000:] if len(X_train) > 6000 else X_train
        tune_y = y_train[-6000:] if len(y_train) > 6000 else y_train
        n_splits = min(4, len(tune_X) // 50)
        if n_splits < 2:
            model, name = self._build_default_model(y_full)
            return model, name, {"evaluated": 0, "best_brier": 0.0}

        tscv = TimeSeriesSplit(n_splits=n_splits)
        best_model = None
        best_name = ""
        best_brier = float("inf")
        evaluated = 0

        candidates = self._build_tuning_candidates(y_full)
        for cand in candidates:
            if evaluated >= self.tuning_max_combinations:
                break
            model = cand["model"]
            name = cand["name"]
            pipeline = self._build_pipeline(model)
            try:
                scores = cross_val_score(
                    pipeline,
                    tune_X,
                    tune_y,
                    cv=tscv,
                    scoring="neg_brier_score",
                )
                brier = float(-np.mean(scores))
            except Exception as exc:
                logger.debug("[MODEL] Tuning skip %s (%s)", name, exc)
                continue

            evaluated += 1
            if brier < best_brier:
                best_brier = brier
                best_model = model
                best_name = name

        if best_model is None:
            model, name = self._build_default_model(y_full)
            return model, name, {"evaluated": evaluated, "best_brier": 0.0}

        return best_model, best_name, {"evaluated": evaluated, "best_brier": best_brier}
    def _build_tuning_candidates(self, y: np.ndarray) -> list[dict]:
        candidates: list[dict] = []

        # XGBoost templates
        try:
            from xgboost import XGBClassifier

            pos = max(1, int(np.sum(y == 1)))
            neg = max(1, int(np.sum(y == 0)))
            scale_pos = float(neg / pos)
            xgb_grid = {
                "n_estimators": [160, 220],
                "max_depth": [4, 6],
                "learning_rate": [0.03, 0.05],
                "subsample": [0.85, 0.95],
                "colsample_bytree": [0.85, 0.95],
            }
            for params in _expand_grid(xgb_grid):
                candidates.append(
                    {
                        "name": "XGBoost",
                        "model": XGBClassifier(
                            eval_metric="logloss",
                            random_state=42,
                            scale_pos_weight=scale_pos,
                            min_child_weight=2,
                            reg_lambda=1.0,
                            reg_alpha=0.05,
                            n_jobs=1,
                            verbosity=0,
                            **params,
                        ),
                    }
                )
        except Exception:
            pass

        # LightGBM templates (opcional)
        try:
            from lightgbm import LGBMClassifier

            lgbm_grid = {
                "n_estimators": [200, 280],
                "num_leaves": [31, 63],
                "learning_rate": [0.03, 0.05],
                "subsample": [0.85],
                "colsample_bytree": [0.85, 0.95],
            }
            for params in _expand_grid(lgbm_grid):
                candidates.append(
                    {
                        "name": "LightGBM",
                        "model": LGBMClassifier(
                            objective="binary",
                            random_state=42,
                            class_weight="balanced",
                            **params,
                        ),
                    }
                )
        except Exception:
            pass

        # Logistic templates
        from sklearn.linear_model import LogisticRegression

        for c in [0.6, 1.0, 1.6]:
            candidates.append(
                {
                    "name": "LogisticRegression",
                    "model": LogisticRegression(
                        max_iter=1200,
                        random_state=42,
                        C=float(c),
                        class_weight="balanced",
                    ),
                }
            )

        return candidates

    def _build_default_model(self, y: np.ndarray):
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

    def _fit_ensemble_members(self, X_train: np.ndarray, y_train: np.ndarray, w_train: np.ndarray, y_full: np.ndarray):
        members: list[tuple[str, object]] = []
        primary = self.proba_model if self.proba_model is not None else self.pipeline
        if primary is not None:
            members.append(("primary", primary))

        # Logistic member
        from sklearn.linear_model import LogisticRegression

        lr_member = self._build_pipeline(
            LogisticRegression(
                max_iter=1200,
                random_state=42,
                C=1.0,
                class_weight="balanced",
            )
        )
        self._fit_pipeline(lr_member, X_train, y_train, w_train)
        members.append(("logreg", lr_member))

        # LightGBM member (se disponivel)
        try:
            from lightgbm import LGBMClassifier

            lgbm_member = self._build_pipeline(
                LGBMClassifier(
                    objective="binary",
                    random_state=42,
                    n_estimators=240,
                    num_leaves=31,
                    learning_rate=0.05,
                    class_weight="balanced",
                    subsample=0.9,
                    colsample_bytree=0.9,
                )
            )
            self._fit_pipeline(lgbm_member, X_train, y_train, w_train)
            members.append(("lgbm", lgbm_member))
        except Exception:
            pass

        # XGBoost member adicional (se primario nao for XGB)
        if self.primary_model_name != "XGBoost":
            try:
                xgb_member, _ = self._build_default_model(y_full)
                xgb_pipeline = self._build_pipeline(xgb_member)
                self._fit_pipeline(xgb_pipeline, X_train, y_train, w_train)
                members.append(("xgb", xgb_pipeline))
            except Exception:
                pass

        self.ensemble_models = members if len(members) > 1 else []

    def _build_pipeline(self, model):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

    def _fit_pipeline(self, pipeline_obj, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        if sample_weight is None:
            pipeline_obj.fit(X, y)
            return
        try:
            pipeline_obj.fit(X, y, model__sample_weight=sample_weight)
        except TypeError:
            pipeline_obj.fit(X, y)
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

            if precision > best_precision or (abs(precision - best_precision) < 1e-9 and coverage > best_coverage):
                best_thr = float(thr)
                best_precision = precision
                best_coverage = coverage

        if best_precision < 0:
            return float(self.min_confidence), 0.0, 0.0
        return best_thr, best_precision, best_coverage

    def _fit_calibrator(self, X: np.ndarray, y: np.ndarray, n_splits: int):
        if not self.enable_calibration:
            return
        if self.pipeline is None:
            return
        if len(X) < self.calibration_min_samples:
            return
        if n_splits < 2:
            return

        cal_splits = min(self.calibration_cv_splits, n_splits)
        if cal_splits < 2:
            return

        try:
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
        if self.enable_ensemble and self.ensemble_models:
            probs = []
            for _, model in self.ensemble_models:
                p = _predict_proba_any(model, X)
                if p is not None and len(p) == len(X):
                    probs.append(np.clip(p, 1e-6, 1 - 1e-6))
            if probs:
                return np.clip(np.mean(np.vstack(probs), axis=0), 1e-6, 1 - 1e-6)

        model = self.proba_model or self.pipeline
        if model is None:
            return np.zeros(len(X))

        p = _predict_proba_any(model, X)
        if p is None:
            return np.zeros(len(X))
        return np.clip(p.astype(float), 1e-6, 1 - 1e-6)

    def _apply_low_data_rank_prior(self, team1_prob: float, features: dict) -> float:
        if self.low_data_rank_blend <= 0:
            return team1_prob

        t1_games = max(0.0, float(features.get("team1_matches_played", 0)))
        t2_games = max(0.0, float(features.get("team2_matches_played", 0)))
        h2h_games = max(0.0, float(features.get("h2h_matches", 0)))

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
        if not self.pipeline:
            return []

        model = self.pipeline.named_steps.get("model")
        if model is None:
            return []

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
        if not self.pipeline:
            return

        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "proba_model": self.proba_model,
                "ensemble_models": self.ensemble_models,
                "enable_ensemble": self.enable_ensemble,
                "primary_model_name": self.primary_model_name,
                "tuning_summary": self.tuning_summary,
                "feature_names": self._feature_names,
                "is_calibrated": self._is_calibrated,
                "recommended_min_confidence": float(self.min_confidence),
            },
            self.model_path,
        )
        logger.info("[MODEL] Salvo em %s", self.model_path)

    def _load(self):
        path = Path(self.model_path)
        if not path.exists():
            return

        try:
            data = joblib.load(self.model_path)
            self.pipeline = data["pipeline"]
            self.proba_model = data.get("proba_model", self.pipeline)
            self.ensemble_models = data.get("ensemble_models", []) or []
            self.enable_ensemble = bool(data.get("enable_ensemble", self.enable_ensemble))
            self.primary_model_name = str(data.get("primary_model_name", ""))
            self.tuning_summary = data.get("tuning_summary", {}) or {}
            self._feature_names = data["feature_names"]
            self._is_calibrated = bool(data.get("is_calibrated", False))
            self.min_confidence = float(data.get("recommended_min_confidence", self.min_confidence))
            self._is_trained = True
            logger.info("[MODEL] Carregado de %s", self.model_path)
        except Exception as e:
            logger.warning("[MODEL] Erro ao carregar: %s", e)

    @property
    def is_trained(self) -> bool:
        return self._is_trained

def _predict_proba_any(model, X: np.ndarray) -> np.ndarray | None:
    try:
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return np.asarray(proba[:, 1], dtype=float)
    except Exception:
        pass

    try:
        decision = np.asarray(model.decision_function(X), dtype=float)
        return 1.0 / (1.0 + np.exp(-decision))
    except Exception:
        return None


def _expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    out = []
    for combo in itertools.product(*values):
        out.append({k: v for k, v in zip(keys, combo)})
    return out


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
