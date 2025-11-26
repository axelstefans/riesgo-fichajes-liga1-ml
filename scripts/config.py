import logging

class Config:
    DEBUG_RECORD_LIMIT = None

    DIR_ENTRADA = "datos_entrada"
    DIR_SALIDA_BRUTOS = "datos_salida/brutos"
    DIR_SALIDA_PROCESADOS = "datos_salida/preprocesados"
    DIR_SALIDA_FINALES = "datos_salida/finales"
    DIR_CACHE = "datos_salida/cache"
    DIR_DEBUG = "datos_salida/debug"

    CSV_SEPARATOR = ';'
    CSV_ENCODING = 'utf-8-sig'

    TM_BASE_URL = "https://www.transfermarkt.es"
    SS_BASE_URL = "https://api.sofascore.com/api/v1"
    DRIVER_VERSION = 136
    USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
    REQUEST_TIMEOUT = 20

    SCRAPING_MIN_DELAY = 9
    SCRAPING_MAX_DELAY = 14

    ID_LIGA_1 = 406

    TEMPORADAS = {
        "2024/25": {"tm_saison_id": "2024", "anio_fichaje": 2024, "file_suffix": "2024_2025"},
        "2023/24": {"tm_saison_id": "2023", "anio_fichaje": 2023, "file_suffix": "2023_2024"},
        "2022/23": {"tm_saison_id": "2022", "anio_fichaje": 2022, "file_suffix": "2022_2023"},
        "2021/22": {"tm_saison_id": "2021", "anio_fichaje": 2021, "file_suffix": "2021_2022"},
        "2020/21": {"tm_saison_id": "2020", "anio_fichaje": 2020, "file_suffix": "2020_2021"},
        "2019/20": {"tm_saison_id": "2019", "anio_fichaje": 2019, "file_suffix": "2019_2020"},
        "2018/19": {"tm_saison_id": "2018", "anio_fichaje": 2018, "file_suffix": "2018_2019"},
        "2017/18": {"tm_saison_id": "2017", "anio_fichaje": 2017, "file_suffix": "2017_2018"},
    }

    MAPEO_CLUBES = {
        "Universitario de Deportes": "Universitario", "Club Sporting Cristal": "Sport. Cristal",
        "Club Alianza Lima": "Alianza Lima", "FBC Melgar": "FBC Melgar",
        "Universidad César Vallejo": "César Vallejo", "Asociación Deportiva Tarma": "AD Tarma",
        "Club Cienciano": "Cienciano", "Cusco FC": "Cusco FC", "Deportivo Garcilaso": "Dep. Garcilaso",
        "Sport Huancayo": "Sport Huancayo", "CD Unión Comercio": "Unión Comercio",
        "CD Los Chankas": "Los Chankas", "Alianza Atlético Sullana": "Alianza Atl.",
        "Universidad Técnica de Cajamarca": "UTC", "Carlos A. Mannucci": "CA Mannucci",
        "Sport Boys Association": "Sport Boys", "Club Atlético Grau": "Atlético Grau",
        "Comerciantes Unidos": "Comerciantes", "Deportivo Municipal": "Dep. Municipal",
        "Deportivo Binacional": "Dep. Binacional", "Academia Deportiva Cantolao": "AD Cantolao",
        "FC Carlos Stein": "Carlos Stein", "Universidad San Martín de Porres": "U. San Martín",
        "Ayacucho FC": "Ayacucho FC", "Rosario FC": "Rosario FC", "Real Garcilaso": "Real Garcilaso",
        "Pirata FC": "Pirata FC", "Alianza Universidad": "Alianza UDH", "Deportivo Llacuabamba": "Llacuabamba",
        "Club Juan Pablo II College": "Juan Pablo II"
    }

    PAISES_SUDAMERICANOS_TOP = ["Argentina", "Brasil", "Uruguay", "Colombia"]
    PAISES_SUDAMERICANOS_OTROS = ["Paraguay", "Ecuador", "Chile", "Bolivia", "Venezuela"]

    CLUBES_LIGA1 = [
        "Club Alianza Lima",
        "Universidad Técnica de Cajamarca",
        "Carlos A. Mannucci",
        "Club Cienciano",
        "Sport Boys Association",
        "Cusco FC",
        "Ayacucho FC",
        "Sport Huancayo",
        "FBC Melgar",
        "Universitario de Deportes",
        "Club Atlético Grau",
        "Alianza Atlético Sullana",
        "Deportivo Garcilaso",
        "Deportivo Municipal",
        "Asociación Deportiva Tarma",
        "Alianza Universidad",
        "Deportivo Binacional",
        "Universidad César Vallejo",
        "Club Sporting Cristal",
        "Academia Deportiva Cantolao",
        "FC Carlos Stein",
        "Comerciantes Unidos",
        "CD Unión Comercio",
        "CD Los Chankas",
        "Universidad San Martín de Porres",
        "Real Garcilaso",
        "Club Juan Pablo II College",
        "Pirata FC",
        "Deportivo Llacuabamba",
        "Rosario FC",
    ]

    EQUIPOS_TOP4 = [
        "Club Alianza Lima",
        "Universitario de Deportes",
        "Club Sporting Cristal",
        "FBC Melgar",
    ]

    CLUBES_GRANDES_KEYWORDS = {
        # Argentina
        "Boca Juniors": ["boca", "c.a. boca", "club atletico boca"],
        "River Plate": ["river", "c.a. river"],
        "Racing Club": ["racing"],
        "San Lorenzo": ["san lorenzo"],
        "Independiente": ["independiente", "club atletico independiente"],
        "Velez Sarsfield": ["velez", "vélez"],

        # Brasil
        "Flamengo": ["flamengo"],
        "Palmeiras": ["palmeiras"],
        "Corinthians": ["corinthians"],
        "Sao Paulo": ["sao paulo", "são paulo", "spfc"],
        "Santos": ["santos fc"],
        "Atletico Mineiro": ["atletico mineiro", "atlético mineiro", "galo"],
        "Gremio": ["gremio", "grêmio"],
        "Internacional": ["internacional", "sport club internacional"],
        "Fluminense": ["fluminense"],

        # Uruguay
        "Penarol": ["peñarol", "penarol", "club atletico penarol"],
        "Nacional (URU)": ["club nacional de football", "nacional (uru)", "nacional"],

        # Colombia
        "Atletico Nacional": ["atletico nacional", "atlético nacional"],
        "Millonarios": ["millonarios"],
        "America de Cali": ["america de cali", "américa de cali"],
        "Junior": ["atlético junior", "junior"],
        "Deportivo Cali": ["deportivo cali"],
        "Deportes Tolima": ["deportes tolima", "tolima"],

        # Chile
        "Colo-Colo": ["colo-colo", "colo colo", "csd colo-colo"],
        "Universidad de Chile": ["universidad de chile", "u de chile"],
        "Universidad Catolica": ["universidad catolica", "u catolica"],

        # Ecuador
        "Barcelona SC": ["barcelona sc", "barcelona guayaquil"],
        "LDU Quito": ["ldu", "liga de quito", "ldu quito"],
        "Independiente del Valle": ["independiente del valle", "idv"],
        "Emelec": ["emelec"],

        # Paraguay
        "Olimpia": ["olimpia"],
        "Cerro Porteno": ["cerro porteno", "cerro porteño"],
        "Libertad": ["libertad"],

        # Perú (Los 4 grandes, por si el origen es un préstamo interno)
        "Club Alianza Lima": ["alianza lima", "club alianza"],
        "Universitario de Deportes": ["universitario de deportes", "universitario"],
        "Club Sporting Cristal": ["sporting cristal", "sport. cristal", "club sporting"],
        "FBC Melgar": ["fbc melgar", "melgar"],

        # Mexico
        "Club America (MEX)": ["club america", "américa (mex)", "america mexico"],
        "Chivas Guadalajara": ["chivas", "guadalajara"],
        "Tigres UANL": ["tigres", "uanl"],
        "Monterrey": ["monterrey", "rayados"],
        "Cruz Azul": ["cruz azul"],
        "Pumas UNAM": ["pumas", "unam"]
    }

def _validar_configuracion():
    sufijos_temporadas = {t["file_suffix"] for t in Config.TEMPORADAS.values()}
    if not sufijos_temporadas:
        raise ValueError("TEMPORADAS vacío.")
    dirs = [Config.DIR_ENTRADA, Config.DIR_SALIDA_BRUTOS, Config.DIR_SALIDA_PROCESADOS, Config.DIR_SALIDA_FINALES, Config.DIR_CACHE, Config.DIR_DEBUG]
    if any(not d for d in dirs):
        raise ValueError("Directorios clave no definidos.")

_validar_configuracion()
