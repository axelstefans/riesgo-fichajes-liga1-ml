"""
tests/test_risk.py
Unit test suite validating risk scoring, threshold boundaries, and inference safety contracts.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

# Guarantee project root and web_streamlit are in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
WEB_STREAMLIT_DIR = ROOT_DIR / "web_streamlit"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(WEB_STREAMLIT_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_STREAMLIT_DIR))

from utils.featurize import FEATURES_FINALES_31
from utils.model_io import (
    clasificar_riesgo,
    load_prediction_assets,
    predict_proba_safe,
    validar_consistencia_features,
)


@pytest.mark.parametrize(
    "proba, threshold, expected_label, expected_color",
    [
        # --- Default threshold = 0.50 ---
        (0.50, 0.50, "ALTO RIESGO", "#ef4444"),  # Exact cut-off boundary
        (0.5000001, 0.50, "ALTO RIESGO", "#ef4444"),  # Epsilon above
        (0.4999999, 0.50, "BAJO RIESGO", "#22c55e"),  # Epsilon below
        (1.0, 0.50, "ALTO RIESGO", "#ef4444"),  # Maximum boundary
        (0.0, 0.50, "BAJO RIESGO", "#22c55e"),  # Minimum boundary
        (0.75, 0.50, "ALTO RIESGO", "#ef4444"),  # Clear high risk
        (0.25, 0.50, "BAJO RIESGO", "#22c55e"),  # Clear low risk
        # --- Custom threshold = 0.40 (training quantile cut-off) ---
        (0.40, 0.40, "ALTO RIESGO", "#ef4444"),  # Exact boundary
        (0.3999, 0.40, "BAJO RIESGO", "#22c55e"),  # Epsilon below
        (0.4001, 0.40, "ALTO RIESGO", "#ef4444"),  # Epsilon above
        # --- Conservative threshold = 0.70 ---
        (0.70, 0.70, "ALTO RIESGO", "#ef4444"),  # Exact boundary
        (0.6999, 0.70, "BAJO RIESGO", "#22c55e"),  # Just under
    ],
)
def test_clasificar_riesgo_boundary_precision(proba, threshold, expected_label, expected_color):
    """
    Validates that clasificar_riesgo behaves deterministically at exact threshold boundaries,
    just above, and just below across varying thresholds.
    """
    label, color = clasificar_riesgo(probabilidad=proba, threshold=threshold)
    assert label == expected_label
    assert color == expected_color


def test_predict_proba_safe_contract_validations():
    """
    Validates input contract enforcement in predict_proba_safe:
    1. Rejects non-DataFrames (dict, list, numpy array).
    2. Rejects batch inputs with length != 1.
    3. Rejects DataFrames with feature count != 31.
    """
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    # 1. Non-DataFrame input
    with pytest.raises(TypeError, match="La entrada X debe ser un DataFrame"):
        predict_proba_safe(mock_model, {"minutesPlayed": 900})

    # 2. Multi-row input (batch size > 1)
    df_multi = pd.DataFrame(np.zeros((2, 31)), columns=FEATURES_FINALES_31)
    with pytest.raises(ValueError, match="Se esperaba 1 fila para la predicción"):
        predict_proba_safe(mock_model, df_multi)

    # 3. Wrong column count (e.g. 30 features instead of 31)
    df_wrong_cols = pd.DataFrame(np.zeros((1, 30)), columns=FEATURES_FINALES_31[:-1])
    with pytest.raises(ValueError, match="Se esperaban 31 features"):
        predict_proba_safe(mock_model, df_wrong_cols)


def test_predict_proba_safe_model_fallbacks():
    """
    Validates inference fallback paths:
    1. Primary: predict_proba.
    2. Fallback: decision_function mapped through sigmoid 1 / (1 + exp(-d)).
    3. Fallback: predict().
    """
    df_valid = pd.DataFrame(np.zeros((1, 31)), columns=FEATURES_FINALES_31)

    # Case 1: Standard predict_proba model
    model_proba = MagicMock(spec=["predict_proba"])
    model_proba.predict_proba.return_value = np.array([[0.35, 0.65]])
    res_proba = predict_proba_safe(model_proba, df_valid)
    assert pytest.approx(res_proba, rel=1e-5) == 0.65

    # Case 2: decision_function model (e.g. SVM / Logistic linear decision)
    # decision = 0.0 -> sigmoid = 0.50
    model_decision = MagicMock(spec=["decision_function"])
    model_decision.decision_function.return_value = np.array([0.0])
    res_decision = predict_proba_safe(model_decision, df_valid)
    assert pytest.approx(res_decision, rel=1e-5) == 0.50

    # Case 3: Only predict() available
    model_predict = MagicMock(spec=["predict"])
    model_predict.predict.return_value = np.array([1.0])
    res_predict = predict_proba_safe(model_predict, df_valid)
    assert res_predict == 1.0


def test_validar_consistencia_features():
    """
    Validates feature schema verification against metadata:
    1. Passes cleanly when features match exactly.
    2. Raises ValueError if columns are missing or extraneous columns exist.
    """
    metadata = {"features_list": FEATURES_FINALES_31}

    # Clean match
    df_valid = pd.DataFrame(np.zeros((1, 31)), columns=FEATURES_FINALES_31)
    validar_consistencia_features(df_valid, metadata)  # Should not raise

    # Missing column
    df_missing = df_valid.drop(columns=["accurateCrosses_p90"])
    with pytest.raises(ValueError, match="Las features no coinciden con las esperadas"):
        validar_consistencia_features(df_missing, metadata)

    # Extra unexpected column
    df_extra = df_valid.copy()
    df_extra["unwanted_metric"] = 1.0
    with pytest.raises(ValueError, match="Las features no coinciden con las esperadas"):
        validar_consistencia_features(df_extra, metadata)


def test_end_to_end_risk_inference_with_production_assets():
    """
    Integration test:
    Loads real production artifacts from assets/ and executes an end-to-end
    inference and risk classification flow on a synthetic player vector.
    """
    assets_dir = WEB_STREAMLIT_DIR / "assets"
    if not (assets_dir / "randomforest_model.joblib").exists():
        pytest.skip("Production artifacts not found in assets/ directory")

    model, metadata, _ = load_prediction_assets(assets_dir)
    assert metadata["model_name"] == "RandomForest"
    assert metadata["decision_threshold"] == 0.5

    # Synthetic player row with all 31 expected features
    df_synthetic = pd.DataFrame(np.zeros((1, 31)), columns=metadata["features_list"])

    # Validate schema
    validar_consistencia_features(df_synthetic, metadata)

    # Run inference
    proba = predict_proba_safe(model, df_synthetic)
    assert isinstance(proba, float)
    assert 0.0 <= proba <= 1.0

    # Run threshold classification
    label, color = clasificar_riesgo(proba, threshold=metadata["decision_threshold"])
    assert label in ["ALTO RIESGO", "BAJO RIESGO"]
    assert color in ["#ef4444", "#22c55e"]
