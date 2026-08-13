"""
core/features.py
Centralized feature engineering logic to prevent Training-Serving Skew.
"""
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import Config
from core.constants import POS_MAP

logger = logging.getLogger(__name__)

# Only calculate the 17 metrics actually required by the final model
METRICAS_P90_USADAS = [
    'goals', 'assists', 'shotsOnTarget', 'shotsOffTarget', 'keyPasses',
    'successfulDribbles', 'offsides', 'wasFouled', 'clearances', 'dribbledPast',
    'fouls', 'aerialDuelsWon', 'accurateFinalThirdPasses', 'accurateLongBalls',
    'accurateCrosses', 'blockedShots', 'yellowCards'
]

def crear_features_numericas(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculando características numéricas (Unificado)...")
    df_featured = df.copy()
    
    numeric_inputs = [
        'minutesPlayed', 'totalPasses', 'accuratePasses', 'totalShots', 'goals', 
        'penaltiesTaken', 'penaltyGoals', 'assists', 'shotsOnTarget', 'shotsOffTarget', 
        'blockedShots', 'keyPasses', 'successfulDribbles', 'offsides', 'clearances', 
        'dribbledPast', 'fouls', 'wasFouled', 'aerialDuelsWon', 'accurateFinalThirdPasses', 
        'accurateLongBalls', 'accurateCrosses', 'yellowCards'
    ]
    
    # Safe coercion for live data
    for col in numeric_inputs:
        if col in df_featured.columns:
            df_featured[col] = pd.to_numeric(df_featured[col], errors='coerce').fillna(0)
        else:
            df_featured[col] = 0
            
    noventa = (df_featured['minutesPlayed'] / 90).replace(0, np.nan)
    
    df_featured['accuratePassesPercentage'] = np.where(df_featured['totalPasses'] > 0, (df_featured['accuratePasses'] / df_featured['totalPasses']) * 100, 0)
    df_featured['goalConversionPercentage'] = np.where(df_featured['totalShots'] > 0, (df_featured['goals'] / df_featured['totalShots']) * 100, 0)
    df_featured['penaltyConversionPercentage'] = np.where(df_featured['penaltiesTaken'] > 0, (df_featured['penaltyGoals'] / df_featured['penaltiesTaken']) * 100, 0)
    
    for metrica in METRICAS_P90_USADAS:
        df_featured[f'{metrica}_p90'] = df_featured[metrica] / noventa
        
    df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
    columnas_numericas = df_featured.select_dtypes(include=np.number).columns
    df_featured[columnas_numericas] = df_featured[columnas_numericas].fillna(0)
    
    return df_featured

def crear_features_contextuales(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Calculando características contextuales (Unificado)...")
    df_context = df.copy()

    # Dummies Posición
    df_context['posicion_agrupada'] = df_context['posicion'].map(POS_MAP).fillna('Defensa')
    df_context = pd.get_dummies(df_context, columns=['posicion_agrupada'], prefix='pos', dtype=int)
    for pos in ['pos_Delantero', 'pos_Mediocampista', 'pos_Defensa']:
        if pos not in df_context.columns:
            df_context[pos] = 0
    if 'pos_Defensa' in df_context.columns:
        df_context = df_context.drop(columns=['pos_Defensa'])

    # Dummies Nacionalidad
    nacionalidades_validas = ['Perú', 'Argentina', 'Colombia', 'Uruguay']
    df_context['nacionalidad_agrupada'] = df_context['nacionalidad_str'].apply(lambda x: x if x in nacionalidades_validas else 'Otras')
    df_context = pd.get_dummies(df_context, columns=['nacionalidad_agrupada'], prefix='nac', dtype=int)
    for nac in ['nac_Perú', 'nac_Argentina', 'nac_Colombia', 'nac_Uruguay', 'nac_Otras']:
        if nac not in df_context.columns:
            df_context[nac] = 0
    if 'nac_Otras' in df_context.columns:
        df_context = df_context.drop(columns=['nac_Otras'])

    # Contexto de Clubes (directo desde Config)
    df_context['contexto_equipo_top'] = df_context['club_destino'].isin(Config.EQUIPOS_TOP4).astype(int)
    df_context['proviene_liga_extranjera'] = (~df_context['club_origen'].isin(Config.CLUBES_LIGA1)).astype(int)
    
    def es_club_grande(nombre_club):
        if pd.isna(nombre_club): return 0
        nombre = str(nombre_club).lower()
        for keywords in Config.CLUBES_GRANDES_KEYWORDS.values():
            if any(kw in nombre for kw in keywords): return 1
        return 0
        
    df_context['proviene_club_grande'] = df_context['club_origen'].apply(es_club_grande)
    return df_context
