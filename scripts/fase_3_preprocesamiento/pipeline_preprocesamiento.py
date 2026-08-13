import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from config import Config
from core.constants import POS_MAP, COLUMNAS_A_ELIMINAR, FEATURES_BAJA_RELEVANCIA
from core.features import crear_features_numericas, crear_features_contextuales

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_y_unificar_datos_brutos(directorio_brutos: Path) -> pd.DataFrame:
    logger.info(f"Iniciando carga y unificación desde: '{directorio_brutos}'")
    if not directorio_brutos.is_dir():
        raise FileNotFoundError(f"El directorio de datos brutos no existe: {directorio_brutos}")

    archivos_csv = sorted(directorio_brutos.glob("*.csv"))
    if not archivos_csv:
        raise FileNotFoundError(f"No se encontraron archivos CSV en {directorio_brutos}")

    lista_df = []
    for archivo in archivos_csv:
        df_temp = pd.read_csv(archivo, sep=Config.CSV_SEPARATOR)
        temporada = archivo.stem.replace('fichajes_brutos_', '')
        df_temp['season'] = temporada
        lista_df.append(df_temp)
    
    df_unificado = pd.concat(lista_df, ignore_index=True)
    logger.info(f"Datos unificados. Total de registros brutos: {len(df_unificado)}")
    return df_unificado

def corregir_anomalia_started_appearances(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando corrección de anomalías en 'started' y 'appearances'...")
    
    filtro_anomalia = df['started'] > df['appearances']
    num_anomalias = filtro_anomalia.sum()
    
    if num_anomalias > 0:
        logger.warning(f"  - Se detectaron {num_anomalias} registros con 'started' > 'appearances'. Procediendo a corregir.")
        
        cols = ['started', 'appearances']
        
        started_original = df.loc[filtro_anomalia, 'started']
        appearances_original = df.loc[filtro_anomalia, 'appearances']
        
        df.loc[filtro_anomalia, 'appearances'] = started_original.combine(appearances_original, max)
        df.loc[filtro_anomalia, 'started'] = started_original.combine(appearances_original, min)
        
        logger.info(f"  - {num_anomalias} registros han sido corregidos lógicamente.")
    else:
        logger.info("  - No se encontraron anomalías en 'started' vs 'appearances'.")
        
    return df

def agregar_stats_por_temporada(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando agregación de estadísticas por jugador y temporada...")
    columnas_a_sumar = ['minutesPlayed', 'appearances', 'started', 'goals', 'totalShots', 'shotsOnTarget', 'shotsOffTarget', 'blockedShots', 'assists', 'keyPasses', 'bigChancesCreated', 'bigChancesMissed', 'successfulDribbles', 'penaltiesWon', 'penaltiesTaken', 'penaltyGoals', 'offsides', 'tackles', 'interceptions', 'clearances', 'dribbledPast', 'penaltiesCommitted', 'fouls', 'wasFouled', 'aerialDuelsWon', 'groundDuelsWon', 'totalPasses', 'accuratePasses', 'accurateFinalThirdPasses', 'accurateLongBalls', 'accurateCrosses', 'possessionLost', 'dispossessed', 'yellowCards', 'redCards']
    columnas_metadata = ['tm_id', 'ss_id', 'nombre_jugador', 'edad', 'nacionalidad_str', 'posicion', 'club_origen', 'club_destino', 'season']
    columnas_sumar_existentes = [col for col in columnas_a_sumar if col in df.columns]
    columnas_meta_existentes = [col for col in columnas_metadata if col in df.columns]

    df_agregado = df.groupby(['tm_id', 'season']).agg({**{col: 'sum' for col in columnas_sumar_existentes}, **{col: 'first' for col in columnas_meta_existentes if col not in ['tm_id', 'season']}}).reset_index()
    logger.info(f"Agregación completada. Registros únicos por jugador-temporada: {len(df_agregado)}")
    return df_agregado

def limpiar_filas(df: pd.DataFrame, minutos_threshold: int = 450) -> pd.DataFrame:
    registros_iniciales = len(df)
    logger.info(f"Iniciando limpieza de filas. Registros iniciales: {registros_iniciales}")
    
    df_limpio = df.dropna(subset=['minutesPlayed']).copy()
    eliminados_nulos = registros_iniciales - len(df_limpio)
    logger.info(f"  - Se eliminaron {eliminados_nulos} filas con 'minutesPlayed' nulo.")
    
    registros_antes = len(df_limpio)
    filtro_problematicos = (
        (df_limpio['minutesPlayed'] > 90) & 
        (df_limpio['totalShots'] == 0) & 
        (df_limpio['totalPasses'] == 0) & 
        (df_limpio['tackles'] == 0)
    )
    df_limpio = df_limpio[~filtro_problematicos]
    eliminados_problematicos = registros_antes - len(df_limpio)
    logger.info(f"  - Se eliminaron {eliminados_problematicos} filas 'problemáticas'.")
    
    registros_antes = len(df_limpio)
    filtro_conversion_absurda = (
    (df_limpio['goals'] > df_limpio['totalShots']) |
    ((df_limpio['goals'] > 0) & (df_limpio['totalShots'].fillna(0) == 0)))
    df_limpio = df_limpio[~filtro_conversion_absurda]
    eliminados_conversion = registros_antes - len(df_limpio)
    logger.info(f"  - Se eliminaron {eliminados_conversion} filas con 'goals > totalShots' (error de scraping).")
    
    registros_antes = len(df_limpio)
    df_limpio = df_limpio[df_limpio['minutesPlayed'] >= minutos_threshold].copy()
    eliminados_umbral = registros_antes - len(df_limpio)
    logger.info(f"  - Se eliminaron {eliminados_umbral} filas con menos de {minutos_threshold} minutos.")
    
    registros_finales = len(df_limpio)
    logger.info(f"Limpieza de filas completada. Registros restantes: {registros_finales}")
    return df_limpio



def seleccionar_features_finales(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando selección final de características...")
    
    # Variables are imported from core.constants

    columnas_existentes_a_eliminar = [col for col in COLUMNAS_A_ELIMINAR if col in df.columns]
    
    df_final = df.drop(columns=columnas_existentes_a_eliminar)
    
    features_baja_relevancia_eliminadas = [col for col in FEATURES_BAJA_RELEVANCIA if col in df.columns]
    
    logger.info(f"  - Columnas originales eliminadas: {len(COLUMNAS_NUMERICAS_ORIGINALES + COLUMNAS_CONTEXTUALES_ORIGINALES)}")
    logger.info(f"  - Features redundantes eliminadas: {len(COLUMNAS_ENGINEERED_REDUNDANTES)}")
    logger.info(f"  - Features de baja relevancia eliminadas: {len(features_baja_relevancia_eliminadas)}")
    logger.info(f"  - Features multicolineales eliminadas: {len(FEATURES_MULTICOLINEALES)}")
    logger.info(f"  - Features incompatibles (Cobertura 2da División) eliminadas: {len(COLUMNAS_COMPLEJAS_INCOMPATIBLES)}")
    
    if 'pos_Delantero' in df_final.columns and 'pos_Mediocampista' in df_final.columns:
        logger.info(f"  ✅ Dummies de posición (pos_Delantero, pos_Mediocampista) mantenidas para modelo.")
    else:
        logger.warning(f"  ⚠️ Dummies de posición NO encontradas.")
    
    auxiliares = ['tm_id', 'ss_id', 'nombre_jugador', 'season']
    features_finales = [c for c in df_final.columns if c not in auxiliares]
    
    logger.info(f"  - Total de features finales: {len(features_finales)}")
    logger.info("Selección de características completada.")
    
    return df_final