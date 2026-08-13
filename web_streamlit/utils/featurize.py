# web_streamlit/utils/featurize.py

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
import sys
from functools import lru_cache

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.constants import POS_MAP

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

# =============================================================================
# FUNCIONES DE CARGA DE CONFIGURACIÓN DESDE JSON (con caché)
# =============================================================================

@lru_cache(maxsize=None)
def _load_json_asset(filename: str):
    path = ASSETS_DIR / filename
    if not path.exists():
        logger.error(f"Archivo de asset no encontrado: {path}")
        raise FileNotFoundError(f"Archivo esencial no encontrado: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar JSON '{filename}': {e}")
        raise

def get_equipos_top4() -> list:
    """Retorna lista de equipos top 4 de la Liga 1."""
    return _load_json_asset("equipos_top4.json") or []

def get_clubes_liga1() -> list:
    """Retorna lista de todos los clubes de la Liga 1."""
    return _load_json_asset("clubes_liga1.json") or []

def get_clubes_grandes_keywords() -> dict:
    """Retorna diccionario de keywords para identificar clubes grandes."""
    return _load_json_asset("clubes_grandes.json") or {}

# =============================================================================
# FUNCIONES DE TRANSFORMACIÓN DE FEATURES
# =============================================================================

def crear_features_numericas(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Calculando características numéricas...")
    df_featured = df.copy()
    
    # ✅ Todas las métricas brutas necesarias para cálculos
    numeric_inputs = [
        'minutesPlayed', 'totalPasses', 'accuratePasses', 'totalShots', 'goals', 
        'penaltiesTaken', 'penaltyGoals', 'assists', 'shotsOnTarget', 'shotsOffTarget', 
        'blockedShots', 'keyPasses', 'successfulDribbles', 'offsides', 'clearances', 
        'dribbledPast', 'fouls', 'wasFouled', 'aerialDuelsWon', 'accurateFinalThirdPasses', 
        'accurateLongBalls', 'accurateCrosses', 'yellowCards'
    ]
    
    # Asegurar que todas las columnas necesarias existen y son numéricas
    for col in numeric_inputs:
        if col in df_featured:
            df_featured[col] = pd.to_numeric(df_featured[col], errors='coerce').fillna(0)
        else:
            df_featured[col] = 0
    
    # Calcular minutesPlayed/90 para normalización
    noventa = (df_featured['minutesPlayed'] / 90).replace(0, np.nan)
    
    # ========== PORCENTAJES DE EFICIENCIA ==========
    df_featured['accuratePassesPercentage'] = np.where(
        df_featured['totalPasses'] > 0, 
        (df_featured['accuratePasses'] / df_featured['totalPasses']) * 100, 
        0
    )
    
    df_featured['goalConversionPercentage'] = np.where(
        df_featured['totalShots'] > 0, 
        (df_featured['goals'] / df_featured['totalShots']) * 100, 
        0
    )
    
    df_featured['penaltyConversionPercentage'] = np.where(
        df_featured['penaltiesTaken'] > 0, 
        (df_featured['penaltyGoals'] / df_featured['penaltiesTaken']) * 100, 
        0
    )
    
    # ========== MÉTRICAS NORMALIZADAS POR 90 MINUTOS ==========
    # ✅ CORREGIDO: Solo calcular las 17 métricas _p90 que se usan
    for metrica in METRICAS_P90_USADAS:
        df_featured[f'{metrica}_p90'] = df_featured[metrica] / noventa
    
    # Limpiar infinitos y NaNs
    df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
    columnas_numericas = df_featured.select_dtypes(include=np.number).columns
    df_featured[columnas_numericas] = df_featured[columnas_numericas].fillna(0)
    
    logger.info(f"   ✅ Calculadas {len(METRICAS_P90_USADAS)} métricas _p90")
    logger.info("   ✅ Calculados 3 porcentajes de eficiencia")
    
    return df_featured


def crear_features_contextuales(df: pd.DataFrame) -> pd.DataFrame:

    logger.info("Calculando características contextuales...")
    df_context = df.copy()

    # Cargar configuraciones
    equipos_top4 = get_equipos_top4()
    clubes_liga1 = get_clubes_liga1()
    clubes_grandes_keywords = get_clubes_grandes_keywords()

    # ========== DUMMIES DE POSICIÓN ==========
    df_context['posicion_agrupada'] = df_context['posicion'].map(POS_MAP).fillna('Defensa')
    
    df_context = pd.get_dummies(
        df_context, 
        columns=['posicion_agrupada'], 
        prefix='pos', 
        dtype=int,
        drop_first=False
    )
    
    # Asegurar que TODAS las columnas existan
    for pos in ['pos_Delantero', 'pos_Mediocampista', 'pos_Defensa']:
        if pos not in df_context.columns:
            df_context[pos] = 0
    
    # Eliminar categoría de referencia
    if 'pos_Defensa' in df_context.columns:
        df_context = df_context.drop(columns=['pos_Defensa'])
    
    logger.info("   ✅ Creadas dummies de posición")

    # ========== DUMMIES DE NACIONALIDAD ==========
    nacionalidades_validas = ['Perú', 'Argentina', 'Colombia', 'Uruguay']
    
    df_context['nacionalidad_agrupada'] = df_context['nacionalidad_str'].apply(
        lambda x: x if x in nacionalidades_validas else 'Otras'
    )
    
    df_context = pd.get_dummies(
        df_context, 
        columns=['nacionalidad_agrupada'], 
        prefix='nac', 
        dtype=int,
        drop_first=False
    )
    
    # Asegurar que TODAS las columnas existan
    for nac in ['nac_Perú', 'nac_Argentina', 'nac_Colombia', 'nac_Uruguay', 'nac_Otras']:
        if nac not in df_context.columns:
            df_context[nac] = 0
    
    # Eliminar categoría de referencia
    if 'nac_Otras' in df_context.columns:
        df_context = df_context.drop(columns=['nac_Otras'])
    
    logger.info("   ✅ Creadas dummies de nacionalidad")

    # ========== FEATURES DE CONTEXTO DE CLUB (CON FUZZY MATCHING) ==========
    
    def normalizar_nombre_club(nombre):
        """Normaliza nombre de club para comparación flexible."""
        if pd.isna(nombre):
            return ""
        # Convertir a minúsculas, eliminar espacios extra, quitar acentos básicos
        nombre_norm = str(nombre).lower().strip()
        # Eliminar prefijos comunes
        prefijos = ['club ', 'c.a. ', 'c.d. ', 'fc ', 'fbc ', 'cd ']
        for prefijo in prefijos:
            if nombre_norm.startswith(prefijo):
                nombre_norm = nombre_norm[len(prefijo):]
        return nombre_norm
    
    def es_equipo_top(club_destino_str):
        """Verifica si el club está en la lista top 4 con fuzzy matching."""
        club_norm = normalizar_nombre_club(club_destino_str)
        
        # Normalizar lista de equipos top
        equipos_top4_norm = [normalizar_nombre_club(eq) for eq in equipos_top4]
        
        # 1. Comparación exacta normalizada
        if club_norm in equipos_top4_norm:
            return 1
        
        # 2. Comparación por substring (fuzzy matching)
        keywords_top = ['alianza', 'universitario', 'sporting', 'cristal', 'melgar']
        for keyword in keywords_top:
            if keyword in club_norm:
                return 1
        
        return 0
    
    def es_liga_extranjera(club_origen_str):
        """Verifica si el club NO es de Liga 1 Perú."""
        club_norm = normalizar_nombre_club(club_origen_str)
        
        # Normalizar lista de clubes Liga 1
        clubes_liga1_norm = [normalizar_nombre_club(cl) for cl in clubes_liga1]
        
        # 1. Comparación exacta normalizada
        if club_norm in clubes_liga1_norm:
            return 0
        
        # 2. Keywords de Liga 1 (para fuzzy matching)
        keywords_liga1 = [
            'alianza', 'universitario', 'cristal', 'melgar', 'boys', 
            'municipal', 'cienciano', 'cusco', 'mannucci', 'vallejo',
            'garcilaso', 'huancayo', 'grau', 'cantolao', 'binacional',
            'tarma', 'comercio', 'chankas', 'stein', 'ayacucho'
        ]
        
        for keyword in keywords_liga1:
            if keyword in club_norm:
                return 0
        
        # Si no coincide con nada, es extranjero
        return 1
    
    def es_club_grande(nombre_club):
        """Verifica si el club es considerado 'grande' usando keywords."""
        if pd.isna(nombre_club):
            return 0
        nombre_norm = normalizar_nombre_club(nombre_club)
        
        # Buscar en keywords de clubes grandes
        for club_oficial, keywords in clubes_grandes_keywords.items():
            keywords_norm = [kw.lower().strip() for kw in keywords]
            # Si algún keyword coincide con el nombre normalizado
            if any(kw in nombre_norm for kw in keywords_norm):
                return 1
        
        return 0
    
    # Aplicar las funciones
    df_context['contexto_equipo_top'] = df_context['club_destino'].apply(es_equipo_top)
    df_context['proviene_liga_extranjera'] = df_context['club_origen'].apply(es_liga_extranjera)
    df_context['proviene_club_grande'] = df_context['club_origen'].apply(es_club_grande)
    
    # Debug logging
    logger.info(f"   🔍 Club destino: '{df_context['club_destino'].values[0]}' → contexto_equipo_top: {df_context['contexto_equipo_top'].values[0]}")
    logger.info(f"   🔍 Club origen: '{df_context['club_origen'].values[0]}' → proviene_liga_extranjera: {df_context['proviene_liga_extranjera'].values[0]}, proviene_club_grande: {df_context['proviene_club_grande'].values[0]}")
    logger.info("   ✅ Creadas 3 features de contexto de club")
    
    return df_context

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