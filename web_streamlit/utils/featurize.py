# web_streamlit/utils/featurize.py

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import sys
import sys
from functools import lru_cache

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.constants import POS_MAP
from core.features import crear_features_numericas, crear_features_contextuales

logger = logging.getLogger(__name__)
ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

FEATURES_FINALES_31 = [
    'accurateCrosses_p90',
    'accurateFinalThirdPasses_p90',
    'accurateLongBalls_p90',
    'accuratePassesPercentage',
    'aerialDuelsWon_p90',
    'assists_p90',
    'blockedShots_p90',
    'clearances_p90',
    'contexto_equipo_top',
    'dribbledPast_p90',
    'edad',
    'fouls_p90',
    'goalConversionPercentage',
    'goals_p90',
    'keyPasses_p90',
    'minutesPlayed',
    'nac_Argentina',
    'nac_Colombia',
    'nac_Perú',
    'nac_Uruguay',
    'offsides_p90',
    'penaltyConversionPercentage',
    'pos_Delantero',
    'pos_Mediocampista',
    'proviene_club_grande',
    'proviene_liga_extranjera',
    'shotsOffTarget_p90',
    'shotsOnTarget_p90',
    'successfulDribbles_p90',
    'wasFouled_p90',
    'yellowCards_p90'
]

# ✅ NUEVO: Métricas que SÍ necesitan normalización _p90
METRICAS_P90_USADAS = [
    'goals',
    'assists',
    'shotsOnTarget',
    'shotsOffTarget',
    'keyPasses',
    'successfulDribbles',
    'offsides',
    'wasFouled',
    'clearances',
    'dribbledPast',
    'fouls',
    'aerialDuelsWon',
    'accurateFinalThirdPasses',
    'accurateLongBalls',
    'accurateCrosses',
    'blockedShots',
    'yellowCards'
]



def seleccionar_features_finales(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Seleccionando y ordenando las 31 características finales...")
    
    # Verificar que todas las features esperadas existen
    missing = [col for col in FEATURES_FINALES_31 if col not in df.columns]
    if missing:
        raise ValueError(
            f"❌ Faltan features requeridas para el modelo:\n"
            f"   {missing}\n"
            f"   Columnas disponibles: {list(df.columns)}"
        )
    
    # Seleccionar SOLO las 31 features en el orden correcto
    df_final = df[FEATURES_FINALES_31].copy()
    
    # Validación final
    if len(df_final.columns) != 31:
        raise ValueError(
            f"❌ ERROR CRÍTICO: Se esperaban 31 features, "
            f"pero se obtuvieron {len(df_final.columns)}"
        )
    
    logger.info(f"   ✅ Seleccionadas {len(df_final.columns)} features en orden correcto")
    
    return df_final


# =============================================================================
# FUNCIÓN ORQUESTADORA PRINCIPAL
# =============================================================================

def featurize_single_player(raw_data: dict) -> pd.DataFrame:

    logger.info("🚀 Iniciando featurización para un jugador...")
    
    # Validar campos críticos en raw_data
    required_fields = ['posicion', 'nacionalidad_str', 'club_origen', 'club_destino', 'edad']
    missing_fields = [field for field in required_fields if field not in raw_data]
    if missing_fields:
        raise ValueError(
            f"❌ Faltan campos requeridos en raw_data: {missing_fields}"
        )
    
    try:
        df = pd.DataFrame([raw_data])
    except Exception as e:
        logger.error(f"❌ No se pudo convertir raw_data a DataFrame: {e}")
        raise
    
    # Pipeline de transformación
    df = crear_features_numericas(df)
    df = crear_features_contextuales(df)
    df = seleccionar_features_finales(df)
    
    logger.info("✅ Featurización completada exitosamente")
    logger.info(f"   Shape final: {df.shape} (debe ser (1, 31))")
    
    return df