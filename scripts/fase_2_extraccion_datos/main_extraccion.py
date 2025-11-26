import pandas as pd
import time
import os
import random
import logging
from tqdm import tqdm
import undetected_chromedriver as uc
from config import Config
from pipeline_jugador import obtener_datos_completos_jugador

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def cargar_fichajes_entrada(filepath: str) -> pd.DataFrame:
    try:
        dtype_map = {'tm_id': str, 'ss_id': str, 'ss_tournament_id': str, 'ss_season_id': str}
        df = pd.read_csv(filepath, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING, dtype=dtype_map)
        df.columns = df.columns.str.strip()
        
        required_cols = [
            'tm_id', 'ss_id', 'ss_season_id', 'ss_tournament_id', 
            'nombre_jugador', 'club_destino'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"El archivo de entrada '{filepath}' no contiene las columnas requeridas: {missing_cols}")
            
        return df
    except FileNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error al cargar el archivo de entrada '{filepath}': {e}")
        raise

def cargar_progreso_previo(filepath: str) -> tuple[pd.DataFrame, list]:
    if not os.path.exists(filepath):
        return pd.DataFrame(), []
    
    try:
        logger.info("Detectado archivo de resultados previo. Leyendo para reanudar...")
        df_existente = pd.read_csv(filepath, sep=Config.CSV_SEPARATOR, dtype={'tm_id': str, 'ss_id': str, 'ss_season_id': str})
        
        required_cols = ['tm_id', 'ss_season_id']
        if not all(col in df_existente.columns for col in required_cols):
            logger.warning("Archivo de progreso previo tiene formato incorrecto. Se comenzará desde cero.")
            return pd.DataFrame(), []

        if df_existente.empty:
            logger.info("Archivo de progreso previo está vacío. Se procesará todo.")
            return pd.DataFrame(), []

        df_existente['unique_row_id'] = df_existente['tm_id'].astype(str) + "_" + df_existente['ss_season_id'].astype(str)
        procesados = df_existente['unique_row_id'].unique().tolist()
        return df_existente, procesados
    except Exception as e:
        logger.warning(f"No se pudo leer el archivo de resultados previo ({e}). Se comenzará desde cero.")
        return pd.DataFrame(), []

def filtrar_registros_pendientes(df: pd.DataFrame, procesados: list) -> pd.DataFrame:
    registros_invalidos = df[['tm_id', 'ss_id', 'ss_season_id', 'ss_tournament_id']].isna().any(axis=1).sum()
    if registros_invalidos > 0:
        logger.warning(f"Se descartarán {registros_invalidos} registros con IDs faltantes.")
    
    df_validos = df.dropna(subset=['tm_id', 'ss_id', 'ss_season_id', 'ss_tournament_id']).copy()
    df_validos['unique_row_id'] = df_validos['tm_id'].astype(str) + "_" + df_validos['ss_season_id'].astype(str)
    
    if procesados:
        df_pendientes = df_validos[~df_validos['unique_row_id'].isin(procesados)]
        registros_ya_procesados = len(df_validos) - len(df_pendientes)
        logger.info(f"Se encontraron {registros_ya_procesados} registros ya procesados. Procesando {len(df_pendientes)} restantes.")
    else:
        df_pendientes = df_validos
    
    return df_pendientes

def procesar_jugadores(driver, df_pendientes: pd.DataFrame, anio_fichaje: int) -> list:
    total = len(df_pendientes)
    if total == 0: return []

    tiempo_promedio_s = (Config.SCRAPING_MIN_DELAY + Config.SCRAPING_MAX_DELAY) / 2
    tiempo_estimado_min = (total * tiempo_promedio_s) / 60
    logger.info(f"Tiempo estimado de procesamiento para esta temporada: ~{tiempo_estimado_min:.1f} minutos.")
    
    datos_nuevos = []
    
    for idx, (_, jugador) in enumerate(tqdm(df_pendientes.iterrows(), total=total, desc=f"Procesando Fichajes {anio_fichaje}")):
        try:
            tm_id = str(jugador['tm_id']).split('.')[0]
            ss_id = str(jugador['ss_id']).split('.')[0]
            ss_season = str(jugador['ss_season_id']).split('.')[0]
            ss_tournament = str(jugador['ss_tournament_id']).split('.')[0]

            datos_jugador = obtener_datos_completos_jugador(
                driver=driver, tm_player_id=tm_id, tm_target_club=jugador['club_destino'],
                ss_player_id=ss_id, ss_season_id=ss_season, ss_tournament_id=ss_tournament,
                anio_fichaje=anio_fichaje
            )
            
            if datos_jugador:
                datos_jugador['ss_season_id'] = ss_season
                datos_nuevos.append(datos_jugador)
                logger.info(f"ÉXITO: Datos de '{datos_jugador.get('nombre_jugador', 'N/A')}' extraídos.")
        except Exception as e:
            logger.error(f"Error procesando TM_ID {jugador.get('tm_id', 'N/A')}: {e}", exc_info=True)
        
        if idx < total - 1:
            pause_duration = random.uniform(Config.SCRAPING_MIN_DELAY, Config.SCRAPING_MAX_DELAY)
            time.sleep(pause_duration)
    
    return datos_nuevos

def guardar_resultados(datos_nuevos: list, df_existente: pd.DataFrame, output_path: str):
    if not datos_nuevos:
        logger.info("No hay nuevos datos para guardar en esta sesión.")
        return
    
    logger.info(f"Guardando {len(datos_nuevos)} nuevos resultados...")
    df_nuevos = pd.DataFrame(datos_nuevos)
    
    if 'unique_row_id' in df_existente.columns:
        df_existente = df_existente.drop(columns=['unique_row_id'])

    df_final = pd.concat([df_existente, df_nuevos], ignore_index=True) if not df_existente.empty else df_nuevos
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_final.to_csv(output_path, index=False, encoding=Config.CSV_ENCODING, sep=Config.CSV_SEPARATOR, decimal='.')
    
    logger.info(f"Total de {len(df_final)} registros guardados en '{output_path}'")

def recolectar_datos_temporada(driver, temporada_config: dict) -> dict:
    input_file = temporada_config['input_file']
    output_file = temporada_config['output_file']
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Archivo de entrada no encontrado: '{input_file}'.")
    
    df_trabajo = cargar_fichajes_entrada(input_file)
    
    df_existente, jugadores_procesados = cargar_progreso_previo(output_file)
    df_pendientes = filtrar_registros_pendientes(df_trabajo, jugadores_procesados)
    
    if df_pendientes.empty:
        logger.info("Todos los registros de esta temporada ya han sido procesados.")
        num_registros = len(df_existente) if not df_existente.empty else 0
        return {'estado': 'Ya completado', 'registros_extraidos': num_registros}
        
    datos_nuevos = procesar_jugadores(driver, df_pendientes, temporada_config['anio_fichaje'])
    guardar_resultados(datos_nuevos, df_existente, output_file)

    if os.path.exists(output_file):
        df_final = pd.read_csv(output_file, sep=Config.CSV_SEPARATOR)
        registros_totales = len(df_final)
    else:
        registros_totales = 0
        
    return {'estado': 'Exitoso', 'registros_extraidos': registros_totales}

if __name__ == "__main__":
    resumen_temporadas = []
    driver = None
    try:
        logger.info("Iniciando navegador en modo headless para toda la sesión...")
        options = uc.ChromeOptions()
        options.add_argument('--headless')
        driver = uc.Chrome(options=options, version_main=Config.DRIVER_VERSION)
        
        for temp_key, temp_config in Config.TEMPORADAS.items():
            logger.info(f"\n--- INICIANDO EXTRACCIÓN DE DATOS BRUTOS PARA {temp_key} ---")
            
            sufijo_archivo_entrada = f"{temp_config['tm_saison_id']}_{int(temp_config['tm_saison_id']) + 1}"
            temp_config['input_file'] = f"{Config.DIR_ENTRADA}/fichajes_{sufijo_archivo_entrada}.csv"
            temp_config['output_file'] = f"{Config.DIR_SALIDA_BRUTOS}/fichajes_brutos_{temp_config['file_suffix']}.csv"
            
            resultado = {'temporada': temp_key, 'estado': 'No iniciado', 'registros_extraidos': 0}
            
            try:
                resultado_ejecucion = recolectar_datos_temporada(driver, temp_config)
                resultado.update(resultado_ejecucion)
            except Exception as e:
                resultado['estado'] = 'Error'
                logger.error(f"Fallo irrecuperable en la temporada {temp_key}: {e}", exc_info=True)
            
            resumen_temporadas.append(resultado)
            
            if temp_key != list(Config.TEMPORADAS.keys())[-1]:
                pausa_entre_temporadas = random.uniform(15, 30)
                logger.info(f"Pausa de {pausa_entre_temporadas:.1f}s antes de la siguiente temporada.")
                time.sleep(pausa_entre_temporadas)
    finally:
        if driver:
            driver.quit()
            logger.info("Navegador cerrado. Sesión de extracción finalizada.")

    logger.info("\n" + "="*80)
    logger.info(f"{'RESUMEN FINAL DE LA EXTRACCIÓN DE DATOS BRUTOS':^80}")
    logger.info("="*80)
    logger.info(f"{'TEMPORADA':<15} {'ESTADO':<25} {'REGISTROS EXTRAÍDOS':<25}")
    logger.info("-" * 80)
    for r in resumen_temporadas:
        logger.info(f"{r['temporada']:<15} {r['estado']:<25} {r.get('registros_extraidos', 'N/A'):<25}")
    logger.info("="*80)