import pandas as pd
import numpy as np
import logging
from pathlib import Path
from config import Config

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

def crear_features_numericas(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando ingeniería de características NUMÉRICAS...")
    df_featured = df.copy()
    noventa = (df_featured['minutesPlayed'] / 90).replace(0, np.nan)
    df_featured['accuratePassesPercentage'] = np.where(df_featured['totalPasses'] > 0, (df_featured['accuratePasses'] / df_featured['totalPasses']) * 100, 0)
    df_featured['goalConversionPercentage'] = np.where(df_featured['totalShots'] > 0, (df_featured['goals'] / df_featured['totalShots']) * 100, 0)
    df_featured['penaltyConversionPercentage'] = np.where(df_featured['penaltiesTaken'] > 0, (df_featured['penaltyGoals'] / df_featured['penaltiesTaken']) * 100, 0)
    
    metricas_de_conteo = ['goals', 'assists', 'totalShots', 'shotsOnTarget', 'shotsOffTarget', 'blockedShots', 'keyPasses', 'bigChancesCreated', 'bigChancesMissed', 'successfulDribbles', 'penaltiesWon', 'offsides', 'tackles', 'interceptions', 'clearances', 'dribbledPast', 'penaltiesCommitted', 'fouls', 'wasFouled', 'aerialDuelsWon', 'groundDuelsWon', 'accurateFinalThirdPasses', 'accurateLongBalls', 'accurateCrosses', 'possessionLost', 'dispossessed', 'yellowCards', 'redCards', 'appearances']
    for metrica in metricas_de_conteo:
        if metrica in df_featured.columns:
            df_featured[f'{metrica}_p90'] = df_featured[metrica] / noventa
    
    df_featured.replace([np.inf, -np.inf], np.nan, inplace=True)
    columnas_a_rellenar = df_featured.select_dtypes(include=np.number).columns
    df_featured[columnas_a_rellenar] = df_featured[columnas_a_rellenar].fillna(0)
    logger.info("Ingeniería de características NUMÉRICAS completada.")
    return df_featured

def crear_features_contextuales(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando ingeniería de características CONTEXTUALES...")
    df_context = df.copy()
    columnas_requeridas = ['posicion', 'nacionalidad_str', 'club_origen', 'club_destino']
    columnas_faltantes = [col for col in columnas_requeridas if col not in df_context.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas requeridas para features contextuales: {columnas_faltantes}")

    pos_map = {
        'Defensa central': 'Defensa', 'Líbero': 'Defensa', 'Lateral izquierdo': 'Defensa', 'Lateral derecho': 'Defensa', 
        'Pivote': 'Mediocampista', 'Mediocentro': 'Mediocampista', 'Mediocentro ofensivo': 'Mediocampista', 
        'Interior izquierdo': 'Mediocampista', 'Interior derecho': 'Mediocampista', 
        'Extremo izquierdo': 'Delantero', 'Extremo derecho': 'Delantero', 'Mediapunta': 'Delantero', 
        'Segundo delantero': 'Delantero', 'Delantero centro': 'Delantero'
    }
    
    df_context['posicion_agrupada'] = df_context['posicion'].map(pos_map)
    df_context = df_context[df_context['posicion_agrupada'].isin(['Defensa', 'Mediocampista', 'Delantero'])].copy()
    logger.info(f"  - Posiciones agrupadas. Registros restantes tras filtrar porteros: {len(df_context)}")
    
    df_context = pd.get_dummies(df_context, columns=['posicion_agrupada'], prefix='pos', dtype=int)
    
    if 'pos_Defensa' in df_context.columns:
        df_context = df_context.drop(columns=['pos_Defensa'])
        logger.info("  - Dummies de posición creadas (baseline: Defensa).")
    
    nacionalidades_validas = ['Perú', 'Argentina', 'Colombia', 'Uruguay']
    df_context['nacionalidad_agrupada'] = df_context['nacionalidad_str'].apply(lambda x: x if x in nacionalidades_validas else 'Otras')
    df_context = pd.get_dummies(df_context, columns=['nacionalidad_agrupada'], prefix='nac', dtype=int)
    if 'nac_Otras' in df_context.columns:
        df_context = df_context.drop(columns=['nac_Otras'])
        logger.info("  - Dummies de nacionalidad creadas (baseline: Otras).")
    
    df_context['contexto_equipo_top'] = df_context['club_destino'].isin(Config.EQUIPOS_TOP4).astype(int)
    df_context['proviene_liga_extranjera'] = (~df_context['club_origen'].isin(Config.CLUBES_LIGA1)).astype(int)
    
    def es_club_grande(nombre_club):
        if pd.isna(nombre_club): return 0
        nombre = str(nombre_club).lower()
        for keywords in Config.CLUBES_GRANDES_KEYWORDS.values():
            if any(kw in nombre for kw in keywords): return 1
        return 0
    
    df_context['proviene_club_grande'] = df_context['club_origen'].apply(es_club_grande)
    logger.info("  - Features de contexto de club creadas.")
    logger.info("Ingeniería de características CONTEXTUALES completada.")
    return df_context

def seleccionar_features_finales(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando selección final de características...")
    
    columnas_numericas_originales = [
        'totalPasses', 'accuratePasses', 'totalShots', 'shotsOnTarget', 'penaltyGoals', 
        'penaltiesTaken', 'started', 'goals', 'assists', 'shotsOffTarget', 'blockedShots', 
        'keyPasses', 'bigChancesCreated', 'bigChancesMissed', 'successfulDribbles', 
        'penaltiesWon', 'offsides', 'tackles', 'interceptions', 'clearances', 
        'dribbledPast', 'penaltiesCommitted', 'fouls', 'wasFouled', 'aerialDuelsWon', 
        'groundDuelsWon', 'accurateFinalThirdPasses', 'accurateLongBalls', 
        'accurateCrosses', 'possessionLost', 'dispossessed', 'yellowCards', 'redCards',
        'appearances'
    ]
    
    columnas_contextuales_originales = ['posicion', 'nacionalidad_str', 'club_origen', 'club_destino']

    columnas_engineered_redundantes = ['totalShots_p90', 'appearances_p90']

    features_baja_relevancia = [
        'penaltiesWon_p90',
        'penaltiesCommitted_p90',
        'redCards_p90'
    ]
    
    features_multicolineales = ['startPercentage']

    columnas_complejas_incompatibles = [
        'possessionLost_p90',
        'dispossessed_p90',
        'tackles_p90',
        'interceptions_p90',
        'groundDuelsWon_p90',
        'bigChancesCreated_p90',
        'bigChancesMissed_p90'
    ]

    columnas_a_eliminar = (
        columnas_numericas_originales + 
        columnas_contextuales_originales + 
        columnas_engineered_redundantes +
        features_baja_relevancia +
        features_multicolineales +
        columnas_complejas_incompatibles
    )

    columnas_existentes_a_eliminar = [col for col in columnas_a_eliminar if col in df.columns]
    
    df_final = df.drop(columns=columnas_existentes_a_eliminar)
    
    features_baja_relevancia_eliminadas = [col for col in features_baja_relevancia if col in df.columns]
    
    logger.info(f"  - Columnas originales eliminadas: {len(columnas_numericas_originales + columnas_contextuales_originales)}")
    logger.info(f"  - Features redundantes eliminadas: {len(columnas_engineered_redundantes)}")
    logger.info(f"  - Features de baja relevancia eliminadas: {len(features_baja_relevancia_eliminadas)}")
    logger.info(f"  - Features multicolineales eliminadas: {len(features_multicolineales)}")
    logger.info(f"  - Features incompatibles (Cobertura 2da División) eliminadas: {len(columnas_complejas_incompatibles)}")
    
    if 'pos_Delantero' in df_final.columns and 'pos_Mediocampista' in df_final.columns:
        logger.info(f"  ✅ Dummies de posición (pos_Delantero, pos_Mediocampista) mantenidas para modelo.")
    else:
        logger.warning(f"  ⚠️ Dummies de posición NO encontradas.")
    
    auxiliares = ['tm_id', 'ss_id', 'nombre_jugador', 'season']
    features_finales = [c for c in df_final.columns if c not in auxiliares]
    
    logger.info(f"  - Total de features finales: {len(features_finales)}")
    logger.info("Selección de características completada.")
    
    return df_final