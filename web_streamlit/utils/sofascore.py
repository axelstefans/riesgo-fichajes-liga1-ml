# utils/sofascore.py
from curl_cffi import requests
import logging
import json
import time
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN ---
IMPERSONATE_VER = "chrome110"
HEADERS = {
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9"
}

# Países donde SUMAMOS (Apertura + Clausura)
PAISES_SUMA = [
    "Argentina", "Mexico", "Colombia", "Uruguay", "Paraguay", "Chile", "Bolivia", "Ecuador"
]

# Categorías internacionales a ignorar
CATEGORIAS_INTERNACIONALES = [
    "South America", "World", "Europe", "International", "CONCACAF", "Asia", "Africa", "FIFA"
]

def _hacer_peticion_segura(url, descripcion):
    try:
        print(f"🌐 {descripcion}: {url}")
        response = requests.get(url, headers=HEADERS, impersonate=IMPERSONATE_VER, timeout=15)
        if response.status_code == 404: return None
        if response.status_code != 200:
            print(f"❌ Error HTTP {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        print(f"❌ Error conexión: {e}")
        return None

def _descargar_imagen_segura(url):
    """
    Descarga la imagen como bytes para evitar bloqueos de Hotlink en el navegador.
    """
    try:
        print(f"📸 Descargando foto: {url}")
        response = requests.get(url, headers=HEADERS, impersonate=IMPERSONATE_VER, timeout=10)
        if response.status_code == 200:
            return response.content # Devolvemos los bytes de la imagen
    except Exception as e:
        print(f"⚠️ No se pudo descargar la imagen: {e}")
    return None

def buscar_jugador_sofascore(nombre_query):
    url = f"https://api.sofascore.com/api/v1/search/all?q={nombre_query}&page=0"
    data = _hacer_peticion_segura(url, "Buscando")
    resultados = []
    if data:
        for res in data.get('results', []):
            if res.get('type') == 'player':
                resultados.append(res.get('entity', {}))
        print(f"   ✅ Encontrados {len(resultados)} jugadores.")
    return resultados

def _obtener_perfil_completo(player_id):
    url = f"https://api.sofascore.com/api/v1/player/{player_id}"
    data = _hacer_peticion_segura(url, "Perfil")
    return data.get('player', {}) if data else {}

def _obtener_ids_desde_ultimos_partidos(player_id):
    """PLAN B: Eventos recientes."""
    url = f"https://api.sofascore.com/api/v1/player/{player_id}/events/last/0"
    data = _hacer_peticion_segura(url, "Plan B (Eventos)")
    
    candidatos = []
    if data and 'events' in data:
        vistos = set()
        for evt in data['events']:
            t = evt.get('tournament', {})
            ut = t.get('uniqueTournament', {})
            s = evt.get('season', {})
            if ut and s:
                key = (ut.get('id'), s.get('id'))
                if key not in vistos:
                    vistos.add(key)
                    candidatos.append({
                        'season_id': s.get('id'),
                        'tournament_id': ut.get('id'),
                        'name': ut.get('name'),
                        'category': ut.get('category', {}).get('name', 'Unknown'),
                        'year_str': s.get('year', '0'),
                        'timestamp': evt.get('startTimestamp', 0)
                    })
    candidatos.sort(key=lambda x: x['timestamp'], reverse=True)
    return candidatos

def _filtrar_ligas_principales(lista_stats):
    """Filtra para quedarse SOLO con Ligas (quita Copas)."""
    if not lista_stats: return []
    
    keywords_liga = ["Liga", "Primera", "Apertura", "Clausura", "Torneo", "Serie", "Brasileirão", "Campeonato", "Tournoi", "Bundesliga", "Premier"]
    keywords_copa = ["Copa Argentina", "Copa MX", "Copa Chile", "Copa Colombia", "Copa BetPlay", "Copa Bicentenario", "Supercopa", "Cup", "Pokal", "Trophy"]

    ligas_filtradas = []
    for s in lista_stats:
        nombre = s.get('src_tournament', '')
        es_liga = any(k in nombre for k in keywords_liga)
        es_copa = any(k in nombre for k in keywords_copa)
        
        if "Copa de la Liga" in nombre:
            es_copa = False
            es_liga = True

        if es_liga and not es_copa:
            ligas_filtradas.append(s)
    
    if not ligas_filtradas and lista_stats:
        return lista_stats
    return ligas_filtradas

def _sumar_estadisticas(lista_stats):
    if not lista_stats: return {}
    acumulado = defaultdict(float)
    metricas = [
        "minutesPlayed", "appearances", "matchesStarted", "goals", "assists", 
        "yellowCards", "redCards", "totalShots", "shotsOnTarget", "shotsOffTarget", 
        "blockedShots", "fouls", "wasFouled", "offsides", "totalPasses", "accuratePasses", 
        "accurateFinalThirdPasses", "keyPasses", "accurateLongBalls", "accurateCrosses", 
        "successfulDribbles", "dribbledPast", "aerialDuelsWon", "clearances", 
        "interceptions", "tackles", "penaltiesTaken", "penaltyGoals"
    ]
    print(f"   ➕ Sumando {len(lista_stats)} torneos...")
    for stats in lista_stats:
        for key in metricas:
            acumulado[key] += stats.get(key, 0)
    return dict(acumulado)

def obtener_stats_sofascore(player_id):
    # 1. Perfil
    perfil = _obtener_perfil_completo(player_id)
    
    # 2. Temporadas
    url_seasons = f"https://api.sofascore.com/api/v1/player/{player_id}/seasons"
    data_seasons = _hacer_peticion_segura(url_seasons, "Temporadas")
    
    fuente_datos = []
    if data_seasons and 'uniqueTournamentSeasons' in data_seasons:
        for t in data_seasons['uniqueTournamentSeasons']:
            u_id = t.get('uniqueTournament', {}).get('id')
            u_name = t.get('uniqueTournament', {}).get('name')
            u_cat = t.get('uniqueTournament', {}).get('category', {}).get('name', 'Unknown')
            for s in t.get('seasons', []):
                fuente_datos.append({
                    'year_str': s.get('year', '0'), 'season_id': s['id'], 
                    'tournament_id': u_id, 'name': u_name, 'category': u_cat
                })
    else:
        print("⚠️ Falló lista. Activando Plan B...")
        fuente_datos = _obtener_ids_desde_ultimos_partidos(player_id) or []

    # Agrupar por AÑO
    anios_map = defaultdict(list)
    for item in fuente_datos:
        y_str = item['year_str']
        try:
            if "26" in y_str: yi = 2026 
            elif "25" in y_str: yi = 2025
            elif "24" in y_str: yi = 2024
            else: yi = int(y_str[:4])
        except: yi = 0
        
        if yi > 2020:
            anios_map[yi].append(item)

    backup_internacional = None

    # 3. Procesar AÑOS
    for anio in sorted(anios_map.keys(), reverse=True):
        print(f"🔍 Analizando año: {anio}")
        torneos = anios_map[anio]
        
        # Filtro duplicados
        torneos_unicos = {}
        for t in torneos:
            key = (t['tournament_id'], t['season_id'])
            if key not in torneos_unicos:
                torneos_unicos[key] = t
        
        lista_torneos = list(torneos_unicos.values())
        stats_recolectadas = []
        team_data = {}

        for t in lista_torneos:
            url_st = f"https://api.sofascore.com/api/v1/player/{player_id}/unique-tournament/{t['tournament_id']}/season/{t['season_id']}/statistics/overall"
            time.sleep(0.1)
            d = _hacer_peticion_segura(url_st, f"Bajando {t['name']}")
            if d:
                s = d.get('statistics', {})
                if s.get('minutesPlayed', 0) > 0:
                    print(f"      ✅ {t['name']} ({t['category']}): {s['minutesPlayed']} min")
                    s['src_tournament'] = t['name']
                    s['src_category'] = t['category']
                    stats_recolectadas.append(s)
                    if 'team' in d: team_data = d['team']

        if not stats_recolectadas: continue

        # 4. FILTRO DE LIGAS DOMÉSTICAS
        torneos_locales = [s for s in stats_recolectadas if s['src_category'] not in CATEGORIAS_INTERNACIONALES]
        
        if torneos_locales:
            ligas_puras = _filtrar_ligas_principales(torneos_locales)
            
            pais_ref = torneos_locales[0]['src_category']
            stats_finales = {}

            if pais_ref in PAISES_SUMA:
                # ARGENTINA: Suma Apertura + Clausura
                print(f"      🔹 País '{pais_ref}'. Sumando Ligas Puras.")
                stats_finales = _sumar_estadisticas(ligas_puras)
            else:
                # PERÚ: Elige la mejor Liga
                print(f"      🔹 País '{pais_ref}'. Seleccionando Liga Principal.")
                mejor_torneo = max(ligas_puras, key=lambda x: x.get('minutesPlayed', 0))
                print(f"      🏆 Seleccionado: {mejor_torneo.get('src_tournament')} ({mejor_torneo.get('minutesPlayed')} min)")
                stats_finales = mejor_torneo

            return {
                "stats": stats_finales,
                "team": team_data,
                "profile": perfil
            }
        
        else:
            print(f"      ⚠️ Solo torneos internacionales. Guardando backup...")
            mejor_inter = max(stats_recolectadas, key=lambda x: x.get('minutesPlayed', 0))
            if not backup_internacional or mejor_inter.get('minutesPlayed', 0) > backup_internacional['stats'].get('minutesPlayed', 0):
                 backup_internacional = {"stats": mejor_inter, "team": team_data, "profile": perfil}
            continue

    if backup_internacional:
        print("⚠️ No se encontraron ligas domésticas. Usando datos internacionales.")
        return backup_internacional

    print("❌ No se encontraron estadísticas.")
    return None

def mapear_sofascore_a_app(data_raw, profile_search):
    if not data_raw: return {}
    stats = data_raw.get('stats', {})
    team = data_raw.get('team', {})
    profile = data_raw.get('profile', profile_search)
    
    pid = profile.get('id')
    
    # --- DESCARGA DE FOTO ---
    foto_url = f"https://api.sofascore.app/api/v1/player/{pid}/image"
    foto_data = _descargar_imagen_segura(foto_url) # Intentamos descargar los bytes
    
    # Si falló la descarga, usamos la URL como fallback (aunque probablemente falle en el navegador también)
    imagen_final = foto_data if foto_data else foto_url

    pais = profile.get('country', {}).get('name', 'Otro')
    if pais == "Peru": pais = "Perú"

    return {
        "nombre_jugador": profile.get('name', 'Desconocido'),
        "imagen_url": imagen_final, # Ahora enviamos los BYTES o la URL
        "edad": _calcular_edad(profile.get('dateOfBirthTimestamp')),
        "posicion": _traducir_posicion(profile.get('position', '')),
        "nacionalidad_str": pais,
        "club_origen": team.get('name', 'Sin Club'),
        "minutesPlayed": int(stats.get('minutesPlayed', 0)),
        "appearances": int(stats.get('appearances', 0)),
        "yellowCards": int(stats.get('yellowCards', 0)),
        "fouls": int(stats.get('fouls', 0)),
        "wasFouled": int(stats.get('wasFouled', 0)),
        "goals": int(stats.get('goals', 0)),
        "assists": int(stats.get('assists', 0)),
        "penaltyGoals": int(stats.get('penaltyGoals', 0)),
        "penaltiesTaken": int(stats.get('penaltiesTaken', 0)),
        "totalShots": int(stats.get('totalShots', 0)),
        "shotsOnTarget": int(stats.get('shotsOnTarget', 0)),
        "shotsOffTarget": int(stats.get('shotsOffTarget', 0)),
        "blockedShots": int(stats.get('blockedShots', 0)),
        "keyPasses": int(stats.get('keyPasses', 0)),
        "successfulDribbles": int(stats.get('successfulDribbles', 0)),
        "offsides": int(stats.get('offsides', 0)),
        "totalPasses": int(stats.get('totalPasses', 0)),
        "accuratePasses": int(stats.get('accuratePasses', 0)),
        "accurateFinalThirdPasses": int(stats.get('accurateFinalThirdPasses', 0)),
        "accurateLongBalls": int(stats.get('accurateLongBalls', 0)),
        "accurateCrosses": int(stats.get('accurateCrosses', 0)),
        "aerialDuelsWon": int(stats.get('aerialDuelsWon', 0)),
        "dribbledPast": int(stats.get('dribbledPast', 0)),
        "clearances": int(stats.get('clearances', 0))
    }

def _traducir_posicion(pos):
    return {"F": "Delantero", "M": "Mediocampista", "D": "Defensa", "G": "Portero"}.get(pos, "Mediocampista")

def _calcular_edad(ts):
    if not ts: return 25
    dt = datetime.fromtimestamp(ts)
    return datetime.now().year - dt.year