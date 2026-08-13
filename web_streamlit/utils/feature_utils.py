# web_streamlit/utils/feature_utils.py

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)
def validate_features_schema(
    X: pd.DataFrame,
    expected_features: list[str],
    expected_dtypes: dict[str, str] | None = None
) -> tuple[bool, list[str]]:

    errores = []
    
    # Validar número de features
    if X.shape[1] != len(expected_features):
        errores.append(
            f"Número incorrecto de features: "
            f"esperadas {len(expected_features)}, encontradas {X.shape[1]}"
        )
    
    # Validar nombres de features
    missing = set(expected_features) - set(X.columns)
    if missing:
        errores.append(f"Features faltantes: {missing}")
    
    extra = set(X.columns) - set(expected_features)
    if extra:
        errores.append(f"Features extras no esperadas: {extra}")
    
    # Validar tipos de datos (opcional)
    if expected_dtypes:
        for feat, expected_dtype in expected_dtypes.items():
            if feat in X.columns:
                actual_dtype = str(X[feat].dtype)
                if actual_dtype != expected_dtype:
                    errores.append(
                        f"Tipo incorrecto para '{feat}': "
                        f"esperado {expected_dtype}, encontrado {actual_dtype}"
                    )
    
    es_valido = len(errores) == 0
    return es_valido, errores