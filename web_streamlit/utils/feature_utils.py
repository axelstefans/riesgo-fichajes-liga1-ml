# web_streamlit/utils/feature_utils.py

from __future__ import annotations
from typing import Sequence
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def align_and_cast(
    X: pd.DataFrame,
    feature_order: Sequence[str] | None = None,
    *,
    strict: bool = True,
    allow_extra: bool = False,
    fill_value: float = 0.0,
    dtype: str = "float32",
) -> pd.DataFrame:
    
    Xc = X.copy()

    if feature_order:
        expected = list(feature_order)
        current = set(Xc.columns)

        # Verificar columnas faltantes
        missing = [c for c in expected if c not in current]
        if missing:
            if strict:
                raise ValueError(
                    f"Faltan columnas requeridas: {missing}\n"
                    f"Columnas disponibles: {list(Xc.columns)}"
                )
            else:
                logger.warning(f"Creando columnas faltantes con valor {fill_value}: {missing}")
                for c in missing:
                    Xc[c] = fill_value

        # Verificar columnas extras
        if not allow_extra:
            extras = [c for c in Xc.columns if c not in expected]
            if extras:
                raise ValueError(
                    f"Columnas no esperadas encontradas: {extras}\n"
                    f"Solo se esperaban: {expected}"
                )

        # Reordenar según feature_order
        Xc = Xc[expected]

    # Casting a tipo numérico
    for c in Xc.columns:
        if Xc[c].dtype.kind in ("i", "u", "f", "b"):  # int/uint/float/bool
            Xc[c] = Xc[c].astype(dtype)
        else:
            # Si llega alguna columna no numérica, coerce a numérico
            logger.warning(f"Columna '{c}' no es numérica, aplicando coerción")
            Xc[c] = pd.to_numeric(Xc[c], errors="coerce").fillna(fill_value).astype(dtype)

    logger.debug(f"DataFrame alineado: {Xc.shape[0]} filas x {Xc.shape[1]} columnas")
    
    return Xc


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