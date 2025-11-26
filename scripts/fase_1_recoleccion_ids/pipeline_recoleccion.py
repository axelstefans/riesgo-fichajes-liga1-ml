import pandas as pd
import time
import os
import random
from tqdm import tqdm
import json
import requests
import unicodedata
import logging

import undetected_chromedriver as uc
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException

from bs4 import BeautifulSoup


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Config:
    USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36'
    PAGE_TIMEOUT = 30
    MIN_DELAY = 15
    MAX_DELAY = 30


def obtener_fichajes_tm(temporada_id: str, driver_version: int) -> pd.DataFrame | None:
    url_exacta = f"https://www.transfermarkt.es/liga-1-clausura/transfers/wettbewerb/TDeC/plus/?saison_id={temporada_id}"
    driver = None
    try:
        options = uc.ChromeOptions()
        options.add_argument(f'user-agent={Config.USER_AGENT}')
        driver = uc.Chrome(headless=True, options=options, version_main=driver_version)
        driver.get(url_exacta)
        WebDriverWait(driver, Config.PAGE_TIMEOUT).until(EC.presence_of_element_located((By.CLASS_NAME, "box")))
        soup = BeautifulSoup(driver.page_source, "html.parser")

        fichajes = []
        cajas_clubes = soup.find_all("div", class_="box")
        for caja in cajas_clubes:
            h2 = caja.find("h2", class_="content-box-headline")
            if not h2: continue
            links_h2 = h2.find_all("a")
            if len(links_h2) < 2: continue
            club_destino = links_h2[1].text.strip()
            header_altas = caja.find("th", string="Altas")
            if not header_altas: continue
            tabla_altas = header_altas.find_parent("table")
            if not tabla_altas: continue
            filas = tabla_altas.find("tbody").find_all("tr")
            for fila in filas:
                celdas = fila.find_all("td")
                if len(celdas) != 9: continue
                coste_cell_text = celdas[8].get_text(strip=True)
                if "Fin de cesión" in coste_cell_text: continue
                club_origen_cell = celdas[7]
                club_origen = club_origen_cell.find("a").get("title") if club_origen_cell.find("a") else "Libre"
                if " II" in club_origen or " U2" in club_origen: continue
                jugador_cell = celdas[0]
                link_jugador = jugador_cell.find("a", href=True)
                if not link_jugador: continue
                nombre_jugador = link_jugador.get("title", "").strip()
                tm_id = link_jugador["href"].split("/")[-1]
                fichajes.append({"nombre_jugador": nombre_jugador, "tm_id": tm_id, "club_origen": club_origen, "club_destino": club_destino})
        
        if fichajes:
            return pd.DataFrame(fichajes)
        else:
            return None
    except Exception as e:
        logger.error(f"Error en obtener_fichajes_tm para temporada {temporada_id}: {e}")
        return None
    finally:
        if driver:
            driver.quit()


def _normalize_string(s: str) -> str:
    return "".join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()


def _buscar_ss_id_jugador(driver, nombre_jugador: str) -> str | None:
    nombres_a_buscar = [nombre_jugador]
    partes_nombre = nombre_jugador.split()
    if len(partes_nombre) > 2:
        nombre_simplificado = f"{partes_nombre[0]} {partes_nombre[-1]}"
        nombres_a_buscar.append(nombre_simplificado)
    for nombre in nombres_a_buscar:
        try:
            search_url = f"https://api.sofascore.com/api/v1/search/all?q={requests.utils.quote(nombre)}"
            driver.get(search_url)
            pre_content = driver.find_element(By.TAG_NAME, 'pre').text
            results = json.loads(pre_content).get('results', [])
            for result in results:
                if result.get('type') == 'player':
                    player_entity = result.get('entity', {})
                    player_id = player_entity.get('id')
                    nombre_sofascore = player_entity.get('name', '')
                    if player_id and _normalize_string(partes_nombre[-1]) in _normalize_string(nombre_sofascore):
                        return str(player_id)
        except (NoSuchElementException, json.JSONDecodeError): continue
        except Exception: continue
    return None


def mapear_ids_ss(df_fichajes: pd.DataFrame, driver_version: int) -> pd.DataFrame | None:
    if df_fichajes is None or df_fichajes.empty:
        logger.warning("El DataFrame de entrada está vacío. Omitiendo mapeo.")
        return None
    logger.info(f"Iniciando mapeo de IDs. Registros de entrada: {len(df_fichajes)}")
    df_unicos = df_fichajes.drop_duplicates(subset=['tm_id'], keep='first').copy()
    num_duplicados = len(df_fichajes) - len(df_unicos)
    if num_duplicados > 0:
        logger.info(f"Se eliminaron {num_duplicados} duplicados. Procesando {len(df_unicos)} jugadores únicos.")
    ss_ids_encontrados = []
    logger.info("Iniciando instancia de navegador para Sofascore...")
    driver = None
    try:
        driver = uc.Chrome(headless=True, version_main=driver_version)
        for _, row in tqdm(df_unicos.iterrows(), total=len(df_unicos), desc="Mapeando jugadores"):
            ss_id = _buscar_ss_id_jugador(driver, row['nombre_jugador'])
            ss_ids_encontrados.append(ss_id)
            time.sleep(random.uniform(Config.MIN_DELAY, Config.MAX_DELAY))
    finally:
        if driver:
            driver.quit()
            logger.info("Cerrando navegador.")
    df_unicos['ss_id'] = ss_ids_encontrados
    encontrados = df_unicos['ss_id'].notna().sum()
    total = len(df_unicos)
    porcentaje = (encontrados / total) * 100 if total > 0 else 0
    logger.info(f"Mapeo finalizado. Se encontraron {encontrados} de {total} IDs ({porcentaje:.2f}%).")
    df_final = df_unicos.dropna(subset=['ss_id'])
    if df_final.empty: return None
    return df_final


def _encontrar_ids_temporada_previa(driver, ss_player_id: str, anio_fichaje: int):
    try:
        historial_url = f"https://api.sofascore.com/api/v1/player/{ss_player_id}/statistics/seasons"
        driver.get(historial_url)
        pre_content = driver.find_element(By.TAG_NAME, 'pre').text
        data = json.loads(pre_content)
        anio_completo_previo = str(anio_fichaje - 1)
        anio_corto_previo = f"{(anio_fichaje - 1) % 100:02d}/{(anio_fichaje) % 100:02d}"
        candidatos_temporada_previa = []
        for tournament_season_group in data.get('uniqueTournamentSeasons', []):
            unique_tournament = tournament_season_group.get('uniqueTournament', {})
            for season in tournament_season_group.get('seasons', []):
                year_field = season.get('year', '')
                if year_field == anio_completo_previo or year_field == anio_corto_previo:
                    season_id = season.get('id')
                    tournament_id = unique_tournament.get('id')
                    if not season_id or not tournament_id: continue
                    stats_url = f"https://api.sofascore.com/api/v1/player/{ss_player_id}/unique-tournament/{tournament_id}/season/{season_id}/statistics/overall"
                    driver.get(stats_url)
                    time.sleep(random.uniform(1, 2))
                    try:
                        stats_content = driver.find_element(By.TAG_NAME, 'pre').text
                        stats_data = json.loads(stats_content)
                        appearances = stats_data.get('statistics', {}).get('appearances', 0)
                        if appearances > 0:
                            candidatos_temporada_previa.append({'season_id': str(season_id), 'tournament_id': str(tournament_id), 'appearances': appearances})
                    except (NoSuchElementException, json.JSONDecodeError): continue
        if not candidatos_temporada_previa: return None, None
        mejor_candidato = max(candidatos_temporada_previa, key=lambda x: x['appearances'])
        return mejor_candidato['season_id'], mejor_candidato['tournament_id']
    except Exception:
        return None, None


def encontrar_temporada_previa(df_mapeado: pd.DataFrame, anio_fichaje: int, driver_version: int) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    if df_mapeado is None or df_mapeado.empty:
        logger.warning("El DataFrame de entrada está vacío. Omitiendo búsqueda.")
        return None, None
    logger.info(f"Iniciando búsqueda de temporada previa. Año de fichaje ref: {anio_fichaje}")
    lista_resultados = []
    logger.info("Iniciando instancia de navegador para Sofascore...")
    driver = None
    try:
        driver = uc.Chrome(headless=True, version_main=driver_version)
        for _, jugador in tqdm(df_mapeado.iterrows(), total=len(df_mapeado), desc="Buscando historial"):
            time.sleep(random.uniform(Config.MIN_DELAY, Config.MAX_DELAY))
            ss_season_id, ss_tournament_id = _encontrar_ids_temporada_previa(driver, jugador['ss_id'], anio_fichaje)
            registro_actual = jugador.to_dict()
            registro_actual['ss_season_id_previo'] = ss_season_id
            registro_actual['ss_tournament_id_previo'] = ss_tournament_id
            lista_resultados.append(registro_actual)
    finally:
        if driver:
            driver.quit()
            logger.info("Cerrando navegador.")
    df_final = pd.DataFrame(lista_resultados)
    df_exitosos = df_final.dropna(subset=['ss_season_id_previo', 'ss_tournament_id_previo']).copy()
    df_fallidos = df_final[df_final['ss_season_id_previo'].isna()].copy()
    encontrados = len(df_exitosos)
    total = len(df_final)
    porcentaje = (encontrados / total) * 100 if total > 0 else 0
    logger.info(f"Búsqueda finalizada. Se encontró temp. previa para {encontrados} de {total} jugadores ({porcentaje:.2f}%).")
    return df_exitosos if not df_exitosos.empty else None, df_fallidos if not df_fallidos.empty else None