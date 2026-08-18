import pandas as pd
import logging
from scrapling.fetchers import Fetcher
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class TransfermarktClient:
    @staticmethod
    def obtener_fichajes(temporada_id: str) -> pd.DataFrame | None:
        url_exacta = f"https://www.transfermarkt.es/liga-1-clausura/transfers/wettbewerb/TDeC/plus/?saison_id={temporada_id}"
        logger.info(f"🌐 Extrayendo Transfermarkt: {url_exacta}")
        
        try:
            # Scrapling impersonates Chrome natively to bypass anti-bot
            page = Fetcher.get(url_exacta, impersonate="chrome110", timeout=30)
            
            if page.status != 200:
                logger.error(f"Error HTTP {page.status} en Transfermarkt")
                return None
                
            # Reutilizamos exactamente la misma lógica de parseo original con bs4
            soup = BeautifulSoup(page.text, "html.parser")
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
                    fichajes.append({
                        "nombre_jugador": nombre_jugador, "tm_id": tm_id, 
                        "club_origen": club_origen, "club_destino": club_destino
                    })
            
            return pd.DataFrame(fichajes) if fichajes else None
            
        except Exception as e:
            logger.error(f"Error extrayendo fichajes TM para temporada {temporada_id}: {e}")
            return None
