import pandas as pd
from pathlib import Path
import logging
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def unificar_datos_brutos():
    logger.info("=" * 100)
    logger.info("INICIO - UNIFICACIÓN DE DATOS BRUTOS")
    logger.info("=" * 100)
    
    # Directorio de entrada y salida
    INPUT_DIR = Path("datos_salida/brutos")
    OUTPUT_FILE = INPUT_DIR / "datos_brutos_merged.csv"
    
    # Verificar que el directorio existe
    if not INPUT_DIR.exists():
        logger.error(f"❌ El directorio {INPUT_DIR} no existe.")
        return
    
    # Buscar todos los archivos CSV
    archivos_csv = sorted(INPUT_DIR.glob("fichajes_brutos_*.csv"))
    
    if not archivos_csv:
        logger.error(f"❌ No se encontraron archivos CSV en {INPUT_DIR}")
        logger.info("   Patrón esperado: fichajes_brutos_YYYY_YYYY.csv")
        return
    
    logger.info(f"📂 Archivos encontrados: {len(archivos_csv)}")
    for archivo in archivos_csv:
        logger.info(f"   - {archivo.name}")
    
    # Lista para almacenar DataFrames
    lista_df = []
    total_registros = 0
    
    # Leer cada archivo y agregar columna 'season'
    for archivo in archivos_csv:
        logger.info(f"\n📄 Procesando: {archivo.name}")
        
        try:
            # Leer CSV
            df_temp = pd.read_csv(archivo, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
            registros_antes = len(df_temp)
            logger.info(f"   ✅ Registros cargados: {registros_antes}")
            
            # Extraer temporada del nombre del archivo
            # Ejemplo: fichajes_brutos_2017_2018.csv → 2017_2018
            temporada = archivo.stem.replace('fichajes_brutos_', '')
            df_temp['season'] = temporada
            logger.info(f"   ✅ Temporada asignada: {temporada}")
            
            # Verificar columnas esperadas
            columnas_esperadas = [
                'tm_id', 'ss_id', 'ss_season_id', 'nombre_jugador', 'edad', 
                'nacionalidad_str', 'posicion', 'club_origen', 'minutesPlayed', 
                'appearances', 'started', 'goals', 'totalShots', 'shotsOnTarget', 
                'shotsOffTarget', 'blockedShots', 'assists', 'keyPasses', 
                'bigChancesCreated', 'bigChancesMissed', 'successfulDribbles', 
                'penaltiesWon', 'penaltiesTaken', 'penaltyGoals', 'offsides', 
                'tackles', 'interceptions', 'clearances', 'dribbledPast', 
                'penaltiesCommitted', 'fouls', 'wasFouled', 'aerialDuelsWon', 
                'groundDuelsWon', 'totalPasses', 'accuratePasses', 
                'accurateFinalThirdPasses', 'accurateLongBalls', 'accurateCrosses', 
                'possessionLost', 'dispossessed', 'yellowCards', 'redCards'
            ]
            
            columnas_faltantes = set(columnas_esperadas) - set(df_temp.columns)
            if columnas_faltantes:
                logger.warning(f"   ⚠️ Columnas faltantes: {columnas_faltantes}")
            
            # Agregar a la lista
            lista_df.append(df_temp)
            total_registros += registros_antes
            
        except Exception as e:
            logger.error(f"   ❌ Error al procesar {archivo.name}: {e}")
            continue
    
    if not lista_df:
        logger.error("❌ No se pudo cargar ningún archivo CSV.")
        return
    
    # Concatenar todos los DataFrames
    logger.info("\n" + "=" * 100)
    logger.info("🔗 Concatenando todos los archivos...")
    df_unificado = pd.concat(lista_df, ignore_index=True)
    
    logger.info(f"✅ Total de registros unificados: {len(df_unificado)}")
    logger.info(f"✅ Total de columnas: {df_unificado.shape[1]}")
    
    # Verificar distribución por temporada
    logger.info("\n📊 Distribución de registros por temporada:")
    distribucion = df_unificado['season'].value_counts().sort_index()
    for temporada, count in distribucion.items():
        logger.info(f"   {temporada}: {count} registros")
    
    # Guardar archivo unificado
    logger.info(f"\n💾 Guardando archivo unificado en: {OUTPUT_FILE}")
    df_unificado.to_csv(OUTPUT_FILE, sep=Config.CSV_SEPARATOR, index=False, encoding=Config.CSV_ENCODING)
    
    logger.info("=" * 100)
    logger.info("✅ UNIFICACIÓN COMPLETADA EXITOSAMENTE")
    logger.info(f"📁 Archivo generado: {OUTPUT_FILE}")
    logger.info(f"📊 Registros totales: {len(df_unificado)}")
    logger.info(f"📊 Columnas totales: {df_unificado.shape[1]}")
    logger.info("=" * 100)

if __name__ == "__main__":
    unificar_datos_brutos()