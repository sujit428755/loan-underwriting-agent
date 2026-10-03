"""Author: Sujit (github.com/sujit428755)
Probability-of-default model (gradient boosting) + SHAP explanations.
Trained on synthetic data so the repo is self-contained. Swap in your real
credit-risk model from the fairness/MLOps project by replacing train()/load()."""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from .config import MODEL_FEATURES, MODEL_PATH

_cache = {}


def _synthetic_training_data(n=8000, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "credit_score": np.clip(rng.normal(690, 85, n), 450, 900),
        "dti_post": rng.uniform(0.05, 0.85, n),
        "loan_to_monthly_income": rng.uniform(1, 20, n),
        "years_employed": rng.uniform(0, 15, n),
        "age": rng.integers(21, 61, n),
        "bounce_count": np.clip(rng.poisson(0.6, n), 0, 6),
        "tenure_months": rng.choice([12, 24, 36, 48, 60], n),
    })
    z = (-2.6 + (700 - df.credit_score) / 55 + 4.0 * (df.dti_post - 0.3)
         + 0.12 * (df.loan_to_monthly_income - 6) - 0.1 * np.minimum(df.years_employed, 8)
         + 0.5 * df.bounce_count)
    df["default"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    return df


def train(save=True, log_mlflow=False) -> dict:
    df = _synthetic_training_data()
    X, y = df[MODEL_FEATURES], df["default"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    model = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=1)
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
    if log_mlflow:
        try:
            import mlflow
            with mlflow.start_run(run_name="credit_model"):
                mlflow.log_params({"model": "GradientBoosting", "features": ",".join(MODEL_FEATURES)})
                mlflow.log_metric("auc", auc)
        except Exception as e:  # tracking is optional
            print("mlflow skipped:", e)
    _cache["model"] = model
    return {"auc": round(auc, 4), "default_rate": round(float(y.mean()), 4)}


def _model():
    if "model" not in _cache:
        if not MODEL_PATH.exists():
            train()
        _cache["model"] = joblib.load(MODEL_PATH)
    return _cache["model"]


def predict(features: dict, explain=True) -> dict:
    row = pd.DataFrame([{k: features[k] for k in MODEL_FEATURES}])
    model = _model()
    pd_ = float(model.predict_proba(row)[:, 1][0])
    out = {"pd": round(pd_, 4), "top_factors": []}
    if explain:
        try:
            import shap
            if "explainer" not in _cache:
                _cache["explainer"] = shap.TreeExplainer(model)
            sv = _cache["explainer"].shap_values(row)
            sv = np.asarray(sv[1] if isinstance(sv, list) else sv)[0]
            ranked = sorted(zip(MODEL_FEATURES, row.iloc[0].tolist(), sv), key=lambda t: -abs(t[2]))
            out["top_factors"] = [{"feature": f, "value": round(float(v), 3), "shap": round(float(s), 3),
                                   "effect": "raises risk" if s > 0 else "lowers risk"} for f, v, s in ranked[:4]]
        except Exception as e:
            out["explain_error"] = str(e)
    return out
