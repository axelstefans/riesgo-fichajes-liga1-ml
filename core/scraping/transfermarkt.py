import logging
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup
from scrapling.fetchers import Fetcher
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class TMRetryableError(Exception):
    pass


def on_tm_retry_error(retry_state: Any) -> None:
    logger.error(
        f"❌ Fallo en Transfermarkt tras {retry_state.attempt_number} intentos: {retry_state.outcome.exception()}"
    )


class TransfermarktClient:
    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry_error_callback=on_tm_retry_error,
    )
    def _fetch_con_reintento(url: str) -> Any:
        page = Fetcher.get(url, impersonate="chrome110", timeout=30)
        if page.status != 200:
            logger.error(f"Error HTTP {page.status} en Transfermarkt")
            if page.status in [429, 500, 502, 503, 504]:
                raise TMRetryableError(f"HTTP {page.status}")
            return None
        return page

    @staticmethod
    def obtener_fichajes(temporada_id: str) -> pd.DataFrame | None:
        url_exacta = f"https://www.transfermarkt.es/liga-1-clausura/transfers/wettbewerb/TDeC/plus/?saison_id={temporada_id}"
        logger.info(f"🌐 Extrayendo Transfermarkt: {url_exacta}")

        try:
            # Scrapling impersonates Chrome natively to bypass anti-bot
            page = TransfermarktClient._fetch_con_reintento(url_exacta)

            if not page:
                return None

            # Reutilizamos exactamente la misma lógica de parseo original con bs4
            soup = BeautifulSoup(page.text, "html.parser")
            fichajes = []

            cajas_clubes = soup.find_all("div", class_="box")
            for caja in cajas_clubes:
                h2 = caja.find("h2", class_="content-box-headline")
                if not h2:
                    continue
                links_h2 = h2.find_all("a")
                if len(links_h2) < 2:
                    continue
                club_destino = links_h2[1].text.strip()

                header_altas = caja.find("th", string="Altas")
                if not header_altas:
                    continue
                tabla_altas = header_altas.find_parent("table")
                if not tabla_altas:
                    continue

                filas = tabla_altas.find("tbody").find_all("tr")
                for fila in filas:
                    celdas = fila.find_all("td")
                    if len(celdas) != 9:
                        continue
                    coste_cell_text = celdas[8].get_text(strip=True)
                    if "Fin de cesión" in coste_cell_text:
                        continue

                    club_origen_cell = celdas[7]
                    club_origen = (
                        club_origen_cell.find("a").get("title")
                        if club_origen_cell.find("a")
                        else "Libre"
                    )
                    if " II" in club_origen or " U2" in club_origen:
                        continue

                    jugador_cell = celdas[0]
                    link_jugador = jugador_cell.find("a", href=True)
                    if not link_jugador:
                        continue

                    nombre_jugador = link_jugador.get("title", "").strip()
                    tm_id = link_jugador["href"].split("/")[-1]
                    fichajes.append(
                        {
                            "nombre_jugador": nombre_jugador,
                            "tm_id": tm_id,
                            "club_origen": club_origen,
                            "club_destino": club_destino,
                        }
                    )

            return pd.DataFrame(fichajes) if fichajes else None

        except Exception as e:
            logger.error(
                f"Error extrayendo fichajes TM para temporada {temporada_id}: {e}"
            )
            return None
