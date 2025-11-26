import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import logging
import pandas as pd
import numpy as np
from selenium.webdriver.common.by import By
from config import Config 

logger = logging.getLogger(__name__)

def _get_data_point_tm(soup, label_text: str) -> str | None:
    data_items = soup.select("ul.data-header__items > li.data-header__label")
    for item in data_items:
        if label_text in item.get_text():
            content_span = item.find("span", class_="data-header__content")
            if content_span:
                return content_span.get_text(strip=True)
    return None

def _obtener_datos_transfermarkt(tm_player_id: str, tm_target_club: str, anio_fichaje: int) -> dict | None:
    try:
        tm_target_club_api = Config.MAPEO_CLUBES.get(tm_target_club, tm_target_club)
        tm_headers = {"User-Agent": Config.USER_AGENT}
        
        # ==========================
        # PERFIL DEL JUGADOR
        # ==========================
        tm_player_url = f"{Config.TM_BASE_URL}/player/profil/spieler/{tm_player_id}"
        response_profile = requests.get(tm_player_url, headers=tm_headers, timeout=Config.REQUEST_TIMEOUT)
        response_profile.raise_for_status()
        soup = BeautifulSoup(response_profile.content, "html.parser")

        # Nombre del jugador
        nombre_jugador = ""
        nombre_h1_tag = soup.find("h1", class_="data-header__headline-wrapper")
        if nombre_h1_tag:
            name_container = nombre_h1_tag.find("div", class_="data-header__headline-container")
            if name_container:
                nombre_jugador = name_container.get_text(strip=True)
            else:
                nombre_jugador = nombre_h1_tag.get_text(separator=" ", strip=True).split('#')[0].strip()

        if not nombre_jugador and soup.find("title"):
            nombre_jugador = soup.find("title").get_text().split(" - ")[0].strip()
        if not nombre_jugador:
            raise ValueError("No se pudo extraer el nombre del jugador.")

        # Fecha de nacimiento
        birth_info = _get_data_point_tm(soup, "F. Nacim./Edad:")
        if not birth_info:
            raise ValueError("Campo 'F. Nacim./Edad:' no encontrado.")
        birth_date_obj = datetime.strptime(
            birth_info.split("(")[0].strip(), "%d/%m/%Y"
        )

        # Nacionalidad
        nationality_str = _get_data_point_tm(soup, "Nacionalidad:")
        if not nationality_str:
            raise ValueError("Campo 'Nacionalidad:' no encontrado.")
        
        # Posición
        player_position = _get_data_point_tm(soup, "Posición:")
        if not player_position:
            raise ValueError("Campo 'Posición:' no encontrado.")
        
        # ==========================
        # HISTORIAL DE TRANSFERENCIAS
        # ==========================
        tm_transfer_api_url = f"{Config.TM_BASE_URL}/ceapi/transferHistory/list/{tm_player_id}"
        response_transfers = requests.get(tm_transfer_api_url, headers=tm_headers, timeout=Config.REQUEST_TIMEOUT)
        response_transfers.raise_for_status()
        transfer_data = response_transfers.json().get("transfers", [])

        if not transfer_data:
            raise ValueError("El historial de transferencias está vacío.")

        df_transfers = pd.DataFrame(transfer_data)
        if df_transfers.empty:
            raise ValueError("El DataFrame de transferencias está vacío.")

        # club destino y fecha
        df_transfers["clubTo"] = df_transfers["to"].apply(
            lambda x: x.get("clubName", "") if isinstance(x, dict) else ""
        )
        df_transfers["date_dt"] = pd.to_datetime(
            df_transfers["date"], format="%d/%m/%Y", errors="coerce"
        )
        df_transfers = df_transfers.dropna(subset=["date_dt"])
        df_transfers["year"] = df_transfers["date_dt"].dt.year

        # === Filtrar por club destino ===
        base_df = df_transfers[df_transfers["clubTo"] == tm_target_club_api]

        if base_df.empty:
            logger.warning(
                f"[TM {tm_player_id}] No se encontró fichaje exacto a '{tm_target_club_api}'. "
                f"Activando fallback de búsqueda parcial..."
            )
            primer_nombre_club = tm_target_club_api.split()[0]
            base_df = df_transfers[
                df_transfers["clubTo"].str.contains(primer_nombre_club, case=False, na=False)
            ]

        if base_df.empty:
            raise ValueError(
                f"Fallo total: No se encontró ningún fichaje para '{tm_target_club_api}' en el historial."
            )

        # === Ajustar por anio_fichaje (rango [anio-1, anio+1]) ===
        if anio_fichaje:
            mask_year = base_df["year"].between(anio_fichaje - 1, anio_fichaje + 1)
            candidatos = base_df[mask_year]
            if candidatos.empty:
                candidatos = base_df
        else:
            candidatos = base_df

        # Primer fichaje al club dentro de los candidatos (más antiguo)
        target_transfer = candidatos.sort_values("date_dt", ascending=True).iloc[0]

        # Club de origen en ese fichaje
        from_info = target_transfer["from"]
        if isinstance(from_info, dict):
            club_origen = from_info.get("clubName", "")
        else:
            club_origen = ""

        signing_date_obj = target_transfer["date_dt"]

        # ==========================
        # CÁLCULO DE LA EDAD
        # ==========================
        # Si tenemos anio_fichaje, usamos una fecha de referencia en ese año
        if anio_fichaje:
            # Por ejemplo: 1 de julio del año de fichaje (aprox. inicio de temporada)
            fecha_ref = datetime(anio_fichaje, 7, 1)
        else:
            # Fallback: usamos la fecha real del fichaje
            fecha_ref = signing_date_obj

        age_at_ref = (
            fecha_ref.year
            - birth_date_obj.year
            - ((fecha_ref.month, fecha_ref.day) < (birth_date_obj.month, birth_date_obj.day))
        )

        logger.debug(
            f"[TM {tm_player_id}] club_destino={tm_target_club_api}, "
            f"club_origen={club_origen}, fecha_ref={fecha_ref.date()}, "
            f"birth_date={birth_date_obj.date()}, edad={age_at_ref}"
        )

        return {
            "nombre_jugador": nombre_jugador,
            "edad": int(age_at_ref),
            "nacionalidad_str": nationality_str,
            "posicion": player_position,
            "club_origen": club_origen,
        }

    except Exception as e:
        logger.error(f"Error en Transfermarkt para TM_ID {tm_player_id}: {e}")
        return None

def _safe_float_convert(value, default=0.0) -> float:
    if value is None: 
        return default
    try:
        return float(str(value).replace(',', '.'))
    except (ValueError, TypeError):
        return default

def _obtener_datos_sofascore(driver, ss_player_id: str, ss_season_id: str, ss_tournament_id: str) -> dict | None:
    sofascore_api_url = f"{Config.SS_BASE_URL}/player/{ss_player_id}/unique-tournament/{ss_tournament_id}/season/{ss_season_id}/statistics/overall"
    try:
        driver.get(sofascore_api_url)
        json_content = driver.find_element(By.TAG_NAME, "pre").text
        data = json.loads(json_content)

        if 'statistics' not in data or not data['statistics']:
            logger.warning(f"No se encontraron estadísticas en SofaScore para SS_ID {ss_player_id}, Season {ss_season_id}.")
            return None

        stats = data['statistics']

        return {
            # --- Generales y de Disponibilidad (3) ---
            "minutesPlayed": int(stats.get("minutesPlayed", 0)),
            "appearances": int(stats.get("appearances", 0)),
            "started": int(stats.get("matchesStarted", 0)),

            # --- Ofensivas (14) ---
            "goals": int(stats.get("goals", 0)),
            "totalShots": int(stats.get("totalShots", 0)),
            "shotsOnTarget": int(stats.get("shotsOnTarget", 0)),
            "shotsOffTarget": int(stats.get("shotsOffTarget", 0)),
            "blockedShots": int(stats.get("blockedShots", 0)),
            "assists": int(stats.get("assists", 0)),
            "keyPasses": int(stats.get("keyPasses", 0)),
            "bigChancesCreated": int(stats.get("bigChancesCreated", 0)),
            "bigChancesMissed": int(stats.get("bigChancesMissed", 0)),
            "successfulDribbles": int(stats.get("successfulDribbles", 0)),
            "penaltiesWon": int(stats.get("penaltyWon", 0)),
            "penaltiesTaken": int(stats.get("penaltiesTaken", 0)),
            "penaltyGoals": int(stats.get("penaltyGoals", 0)),
            "offsides": int(stats.get("offsides", 0)),

            # --- Defensivas y Físicas (9) ---
            "tackles": int(stats.get("tackles", 0)),
            "interceptions": int(stats.get("interceptions", 0)),
            "clearances": int(stats.get("clearances", 0)),
            "dribbledPast": int(stats.get("dribbledPast", 0)),
            "penaltiesCommitted": int(stats.get("penaltyConceded", 0)),
            "fouls": int(stats.get("fouls", 0)),
            "wasFouled": int(stats.get("wasFouled", 0)),
            "aerialDuelsWon": int(stats.get("aerialDuelsWon", 0)),
            "groundDuelsWon": int(stats.get("groundDuelsWon", 0)),

            # --- Técnicas y de Distribución (7) ---
            "totalPasses": int(stats.get("totalPasses", 0)),
            "accuratePasses": int(stats.get("accuratePasses", 0)),
            "accurateFinalThirdPasses": int(stats.get("accurateFinalThirdPasses", 0)),
            "accurateLongBalls": int(stats.get("accurateLongBalls", 0)),
            "accurateCrosses": int(stats.get("accurateCrosses", 0)),
            "possessionLost": int(stats.get("possessionLost", 0)),
            "dispossessed": int(stats.get("dispossessed", 0)),
            
            # --- Disciplina (2) ---
            "yellowCards": int(stats.get("yellowCards", 0)),
            "redCards": int(stats.get("redCards", 0)),
        }
    except Exception as e:
        logger.error(f"Error en SofaScore para SS_ID {ss_player_id}: {e}")
        return None

def obtener_datos_completos_jugador(driver, tm_player_id: str, tm_target_club: str, ss_player_id: str, ss_season_id: str, ss_tournament_id: str, anio_fichaje: int) -> dict | None:
    tm_data = _obtener_datos_transfermarkt(tm_player_id, tm_target_club, anio_fichaje)
    if not tm_data:
        return None

    ss_data = _obtener_datos_sofascore(driver, ss_player_id, ss_season_id, ss_tournament_id)

    final_data = {
        "tm_id": tm_player_id,
        "ss_id": ss_player_id,
        "ss_season_id": ss_season_id,
        "nombre_jugador": tm_data.get("nombre_jugador"),
        "edad": tm_data.get("edad"),
        "nacionalidad_str": tm_data.get("nacionalidad_str"), 
        "posicion": tm_data.get("posicion"),
        "club_origen": tm_data.get("club_origen"),
    }

    # --- LISTA SINCRONIZADA CON LAS 35 MÉTRICAS ---
    ss_fields = [
        # Generales y de Disponibilidad
        "minutesPlayed", "appearances", "started",
        # Ofensivas
        "goals", "totalShots", "shotsOnTarget", "shotsOffTarget", "blockedShots",
        "assists", "keyPasses", "bigChancesCreated", "bigChancesMissed",
        "successfulDribbles", "penaltiesWon", "penaltiesTaken", "penaltyGoals", "offsides",
        # Defensivas y Físicas
        "tackles", "interceptions", "clearances", "dribbledPast", "penaltiesCommitted",
        "fouls", "wasFouled", "aerialDuelsWon", "groundDuelsWon",
        # Técnicas y de Distribución
        "totalPasses", "accuratePasses", "accurateFinalThirdPasses", "accurateLongBalls",
        "accurateCrosses", "possessionLost", "dispossessed",
        # Disciplina
        "yellowCards", "redCards"
    ]

    if ss_data:
        final_data.update(ss_data)
    else:
        for field in ss_fields:
            final_data[field] = np.nan

    return final_data