import pandas as pd
import time
import os
import random
from tqdm import tqdm
import json
import requests
import logging
from pathlib import Path
import sys

# Agregar la raíz del proyecto al sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from config import Config
from core.scraping.transfermarkt import TransfermarktClient
from core.scraping.sofascore import buscar_ss_id_jugador_exacto, encontrar_ids_temporada_previa

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def mapear_ids_ss(df_fichajes: pd.DataFrame) -> pd.DataFrame | None:
    if df_fichajes is None or df_fichajes.empty:
        logger.warning("El DataFrame de entrada está vacío. Omitiendo mapeo.")
        return None
    logger.info(f"Iniciando mapeo de IDs. Registros de entrada: {len(df_fichajes)}")
    df_unicos = df_fichajes.drop_duplicates(subset=['tm_id'], keep='first').copy()
    num_duplicados = len(df_fichajes) - len(df_unicos)
    if num_duplicados > 0:
        logger.info(f"Se eliminaron {num_duplicados} duplicados. Procesando {len(df_unicos)} jugadores únicos.")
    
    ss_ids_encontrados = []
    
    for _, row in tqdm(df_unicos.iterrows(), total=len(df_unicos), desc="Mapeando jugadores"):
        ss_id = buscar_ss_id_jugador_exacto(row['nombre_jugador'])
        ss_ids_encontrados.append(ss_id)
        time.sleep(random.uniform(Config.MIN_DELAY, Config.MAX_DELAY))
        
    df_unicos['ss_id'] = ss_ids_encontrados
    encontrados = df_unicos['ss_id'].notna().sum()
    total = len(df_unicos)
    porcentaje = (encontrados / total) * 100 if total > 0 else 0
    logger.info(f"Mapeo finalizado. Se encontraron {encontrados} de {total} IDs ({porcentaje:.2f}%).")
    df_final = df_unicos.dropna(subset=['ss_id'])
    if df_final.empty: return None
    return df_final


def encontrar_temporada_previa(df_mapeado: pd.DataFrame, anio_fichaje: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if df_mapeado is None or df_mapeado.empty:
        logger.warning("El DataFrame de entrada está vacío. Omitiendo búsqueda.")
        return None, None
    logger.info(f"Iniciando búsqueda de temporada previa. Año de fichaje ref: {anio_fichaje}")
    lista_resultados = []
    
    for _, jugador in tqdm(df_mapeado.iterrows(), total=len(df_mapeado), desc="Buscando historial"):
        time.sleep(random.uniform(Config.MIN_DELAY, Config.MAX_DELAY))
        ss_season_id, ss_tournament_id = encontrar_ids_temporada_previa(jugador['ss_id'], anio_fichaje)
        registro_actual = jugador.to_dict()
        registro_actual['ss_season_id_previo'] = ss_season_id
        registro_actual['ss_tournament_id_previo'] = ss_tournament_id
        lista_resultados.append(registro_actual)
    df_final = pd.DataFrame(lista_resultados)
    df_exitosos = df_final.dropna(subset=['ss_season_id_previo', 'ss_tournament_id_previo']).copy()
    df_fallidos = df_final[df_final['ss_season_id_previo'].isna()].copy()
    encontrados = len(df_exitosos)
    total = len(df_final)
    porcentaje = (encontrados / total) * 100 if total > 0 else 0
    logger.info(f"Búsqueda finalizada. Se encontró temp. previa para {encontrados} de {total} jugadores ({porcentaje:.2f}%).")
    return df_exitosos if not df_exitosos.empty else None, df_fallidos if not df_fallidos.empty else None


def recolectar_ids_temporada(config_temporada: dict) -> dict:
    temporada_id = config_temporada['tm_saison_id']
    anio_fichaje = config_temporada['anio_fichaje']
    output_file = config_temporada['output_file']

    if os.path.exists(output_file):
        df_existente = pd.read_csv(output_file, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
        logger.info(f"El archivo para la temporada {temporada_id} ya existe. Omitiendo recolección.")
        return {'estado': 'Ya completado', 'registros': len(df_existente)}

    logger.info("Extrayendo fichajes base desde Transfermarkt...")
    df_tm = TransfermarktClient.obtener_fichajes(temporada_id)
    if df_tm is None: return {'estado': 'Error TM', 'registros': 0}

    logger.info("Mapeando con IDs de SofaScore...")
    df_mapeado = mapear_ids_ss(df_tm)
    if df_mapeado is None: return {'estado': 'Error Mapeo SS', 'registros': 0}

    logger.info("Buscando IDs de la temporada previa en SofaScore...")
    df_exitosos, df_fallidos = encontrar_temporada_previa(df_mapeado, anio_fichaje)
    if df_exitosos is None: return {'estado': 'Error Historial SS', 'registros': 0}

    if df_fallidos is not None and not df_fallidos.empty:
        logger.warning(f"Guardando {len(df_fallidos)} registros fallidos para revisión manual...")
        fallidos_file = output_file.replace('.csv', '_FALLIDOS.csv')
        df_fallidos.to_csv(fallidos_file, index=False, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)

    logger.info(f"Guardando {len(df_exitosos)} registros exitosos...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_exitosos.to_csv(output_file, index=False, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)

    return {'estado': 'Exitoso', 'registros': len(df_exitosos)}


if __name__ == "__main__":
    logger.info("==================================================")
    logger.info("INICIANDO PIPELINE DE RECOLECCIÓN DE IDs")
    logger.info("==================================================")

    resumen = []
    
    for temporada_name, temporada_config in Config.TEMPORADAS.items():
        logger.info(f"\nProcesando: {temporada_name}")
        temporada_config['output_file'] = f"{Config.DIR_SALIDA}/fichajes_{temporada_config['tm_saison_id']}_{temporada_config['anio_fichaje']}.csv"
        
        try:
            resultado = recolectar_ids_temporada(temporada_config)
            resumen.append({'Temporada': temporada_name, 'Estado': resultado['estado'], 'Registros': resultado.get('registros', 0)})
        except Exception as e:
            logger.error(f"Fallo irrecuperable en {temporada_name}: {e}")
            resumen.append({'Temporada': temporada_name, 'Estado': 'Fallo Irrecuperable', 'Registros': 0})
            
        if temporada_name != list(Config.TEMPORADAS.keys())[-1]:
            logger.info("Pausa entre temporadas...")
            time.sleep(10)

    logger.info("\n==================================================")
    logger.info("RESUMEN DE RECOLECCIÓN")
    logger.info("==================================================")
    df_resumen = pd.DataFrame(resumen)
    print(df_resumen.to_string(index=False))