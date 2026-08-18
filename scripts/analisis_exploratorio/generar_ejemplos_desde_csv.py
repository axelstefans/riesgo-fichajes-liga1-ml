import pandas as pd
import json
import logging
from pathlib import Path
import time
import random
from tqdm import tqdm
from config import Config
import requests
from bs4 import BeautifulSoup
from core.scraping.sofascore import get_player_statistics_by_tournament

# --- Importamos funciones de procesamiento que sí son reutilizables ---
from pipeline_preprocesamiento import corregir_anomalia_started_appearances

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
ARCHIVO_ENTRADA = Path(Config.DIR_ENTRADA) / "jugadores_para_ejemplos.csv"
ASSETS_DIR = Path("web_streamlit/assets")
OUTPUT_JSON = ASSETS_DIR / "jugadores_ejemplo.json"

# =============================================================================
# FUNCIONES DE SCRAPING (AUTOCONTENIDAS)
# =============================================================================

def _get_data_point_tm(soup, label_text: str) -> str | None:
    data_items = soup.select("ul.data-header__items > li.data-header__label")
    for item in data_items:
        if label_text in item.get_text():
            content_span = item.find("span", class_="data-header__content")
            if content_span:
                return content_span.get_text(strip=True)
    return None

def _obtener_datos_tm_simplificado(tm_player_id: str) -> dict | None:
    try:
        tm_headers = {"User-Agent": Config.USER_AGENT}
        tm_player_url = f"{Config.TM_BASE_URL}/player/profil/spieler/{tm_player_id}"
        response_profile = requests.get(tm_player_url, headers=tm_headers, timeout=Config.REQUEST_TIMEOUT)
        response_profile.raise_for_status()
        soup = BeautifulSoup(response_profile.content, "html.parser")
        
        birth_info = _get_data_point_tm(soup, "F. Nacim./Edad:")
        edad = int(birth_info.split("(")[1].replace(")", "").strip()) if birth_info and "(" in birth_info else 0
        
        nationality_str = _get_data_point_tm(soup, "Nacionalidad:")
        player_position = _get_data_point_tm(soup, "Posición:")

        if not all([nationality_str, player_position, edad > 0]):
            raise ValueError("No se pudieron extraer todos los datos biográficos de Transfermarkt.")
        
        return {
            "edad": edad,
            "nacionalidad_str": nationality_str.strip(),
            "posicion": player_position.strip()
        }
    except Exception as e:
        logger.error(f"Error en Transfermarkt (simplificado) para TM_ID {tm_player_id}: {e}")
        return None



# =============================================================================
# AGREGACIÓN
# =============================================================================

def agregar_stats_por_ejemplo(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Iniciando agregación por jugador y anio_evaluacion...")
    columnas_a_sumar = [
        'minutesPlayed', 'appearances', 'started', 'goals', 'totalShots',
        'shotsOnTarget', 'shotsOffTarget', 'blockedShots', 'assists',
        'keyPasses', 'bigChancesCreated', 'bigChancesMissed',
        'successfulDribbles', 'penaltiesWon', 'penaltiesTaken',
        'penaltyGoals', 'offsides', 'tackles', 'interceptions',
        'clearances', 'dribbledPast', 'penaltiesCommitted', 'fouls',
        'wasFouled', 'aerialDuelsWon', 'groundDuelsWon', 'totalPasses',
        'accuratePasses', 'accurateFinalThirdPasses', 'accurateLongBalls',
        'accurateCrosses', 'possessionLost', 'dispossessed',
        'yellowCards', 'redCards'
    ]
    columnas_metadata = [
        'tm_id', 'ss_id', 'nombre_jugador', 'club_origen', 'club_destino',
        'season', 'posicion', 'nacionalidad_str', 'edad'
    ]
    
    columnas_meta_existentes = [col for col in columnas_metadata if col in df.columns]

    agg_dict = {
        **{col: 'sum' for col in columnas_a_sumar if col in df.columns},
        **{col: 'first' for col in columnas_meta_existentes}
    }
    
    if 'tm_id' in agg_dict:
        del agg_dict['tm_id']
    if 'anio_evaluacion' in agg_dict:
        del agg_dict['anio_evaluacion']

    df_agregado = df.groupby(['tm_id', 'anio_evaluacion']).agg(agg_dict).reset_index()

    df_agregado['season'] = df_agregado['anio_evaluacion'].apply(
        lambda y: f"{int(y)}_{int(y)+1}"
    )
    logger.info(f"Agregación completada. Registros consolidados: {len(df_agregado)}")
    return df_agregado

# =============================================================================
# ORQUESTADOR PRINCIPAL (SIN CACHÉ)
# =============================================================================

def generar_json_ejemplos():
    logger.info("="*50 + " INICIO GENERACIÓN DE EJEMPLOS " + "="*50)
    try:
        df_input = pd.read_csv(
            ARCHIVO_ENTRADA,
            sep=Config.CSV_SEPARATOR,
            dtype=str
        )
        if 'anio_evaluacion' not in df_input.columns:
            raise ValueError("El archivo de entrada debe contener la columna 'anio_evaluacion'.")
    except Exception as e:
        logger.error(f"❌ FATAL: Error al leer o validar '{ARCHIVO_ENTRADA}': {e}")
        return

    logger.info(f"📂 Total de registros en CSV: {len(df_input)}")

    datos_crudos_completos = []

    try:
        for _, jugador in tqdm(
            df_input.iterrows(),
            total=len(df_input),
            desc="Extrayendo datos"
        ):
            try:
                ss_data = get_player_statistics_by_tournament(
                    str(jugador['ss_id']).split('.')[0],
                    str(jugador['ss_season_id']).split('.')[0],
                    str(jugador['ss_tournament_id']).split('.')[0]
                )
                if not ss_data:
                    logger.warning(f"  - ⚠️ Sin datos SofaScore para SS_ID {jugador['ss_id']}")
                    continue

                registro_completo = {**jugador.to_dict(), **ss_data}

                tm_data = _obtener_datos_tm_simplificado(str(jugador['tm_id']).split('.')[0])
                if tm_data:
                    registro_completo.update(tm_data)

                datos_crudos_completos.append(registro_completo)
                time.sleep(random.uniform(1, 2.5))

            except Exception as e:
                logger.error(f"  - ❌ Error procesando SS_ID {jugador.get('ss_id', 'N/A')}: {e}")
    except Exception as general_e:
        logger.error(f"Error general durante extracción: {general_e}")

    logger.info("Iniciando procesamiento de todos los datos extraídos...")

    if not datos_crudos_completos:
        logger.error("No hay datos válidos para procesar. Abortando.")
        return

    df_extraido = pd.DataFrame(datos_crudos_completos)
    df_agregado = agregar_stats_por_ejemplo(df_extraido)
    df_corregido = corregir_anomalia_started_appearances(df_agregado)

    lista_ejemplos_final = []
    for _, jugador_data in df_corregido.iterrows():
        raw_dict = jugador_data.to_dict()
        label = f"{raw_dict.get('nombre_jugador', 'N/A')} ({raw_dict.get('anio_evaluacion', 'N/A')})"
        lista_ejemplos_final.append({"label": label, "datos_crudos": raw_dict})
        logger.info(f"  - ✅ Empaquetado: {label}")

    if lista_ejemplos_final:
        ASSETS_DIR.mkdir(parents=True, exist_ok=True)
        output_data = {"jugadores_ejemplo": lista_ejemplos_final}
        with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=4, ensure_ascii=False)
        logger.info(
            f"\n💾 Archivo JSON generado con {len(lista_ejemplos_final)} ejemplos consolidados."
        )
    else:
        logger.warning("No se pudo generar ningún ejemplo válido.")

    logger.info("="*50 + " FIN GENERACIÓN DE EJEMPLOS " + "="*50)

if __name__ == "__main__":
    generar_json_ejemplos()
