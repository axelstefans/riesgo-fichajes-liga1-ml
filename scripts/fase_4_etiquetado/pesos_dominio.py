# scripts/pesos_dominio.py
PESOS_BASE = {
    # Ofensivos
    "goals_p90": 2.0,
    "assists_p90": 1.5,
    "keyPasses_p90": 2.0,
    "shotsOnTarget_p90": 1.8,
    "successfulDribbles_p90": 1.5,
    
    # Pases
    "accurateFinalThirdPasses_p90": 1.2,
    "accurateCrosses_p90": 1.0,
    "accurateLongBalls_p90": 0.8,
    "accuratePassesPercentage": 1.0,
    
    # Conversión (REDUCIDO) ✅
    "goalConversionPercentage": 0.8,      # ✅ Reducido de 1.5 a 0.8
    "penaltyConversionPercentage": 0.5,
    
    # Defensivos
    "aerialDuelsWon_p90": 1.2,
    "clearances_p90": 0.8,
    
    # Otros
    "wasFouled_p90": 0.5,
    "minutesPlayed": 1.0,
    
    # Negativos (AJUSTADOS) ✅
    "shotsOffTarget_p90": -0.8,
    "offsides_p90": -1.0,
    "dribbledPast_p90": -1.0,
    "blockedShots_p90": 0.0,          # ✅ Cambiado de -0.3 a 0.0
    "yellowCards_p90": -0.5,
    "fouls_p90": -0.8,
    
    # Contextuales
    "contexto_equipo_top": 0.3,
    "proviene_liga_extranjera": -0.2,
    "proviene_club_grande": 0.2,
    
    # Nacionalidad
    "nac_Perú": 0.3,
    "nac_Argentina": 0.1,
    "nac_Colombia": 0.1,
    "nac_Uruguay": 0.1,
    
    # Dummies
    "pos_Delantero": 0.0,
    "pos_Mediocampista": 0.0,
    
    # Edad (CORREGIDO) ✅
    "edad": -0.2,  # ✅ Cambiado de 0.0 a -0.2 (edad alta = ligero riesgo)
}

AJUSTES_DELANTERO = {
    "goals_p90": +0.8,
    "shotsOnTarget_p90": +0.5,
    "goalConversionPercentage": +0.5,
    "successfulDribbles_p90": +0.2,
    "aerialDuelsWon_p90": +0.2,
    "offsides_p90": -0.5,
    "clearances_p90": -0.5,
    "accuratePassesPercentage": -0.3,
}

AJUSTES_MEDIO = {
    "keyPasses_p90": +0.6,
    "accuratePassesPercentage": +0.5,
    "accurateFinalThirdPasses_p90": +0.4,
    "clearances_p90": -0.3,
}

AJUSTES_DEFENSA = {
    "clearances_p90": +0.7,
    "aerialDuelsWon_p90": +0.5,
    "dribbledPast_p90": -0.8,
    "fouls_p90": -0.2,
    "goals_p90": -0.5,
    "keyPasses_p90": -0.5,
    "successfulDribbles_p90": -0.5,
    "shotsOnTarget_p90": -0.8,
}

def get_pesos_por_posicion(posicion_grupo: str) -> dict:
    base = PESOS_BASE.copy()
    pos = (posicion_grupo or "").strip().lower()
    
    if pos == "delantero":
        ajustes = AJUSTES_DELANTERO
    elif pos == "mediocampista":
        ajustes = AJUSTES_MEDIO
    else:
        ajustes = AJUSTES_DEFENSA
        
    for k, v in ajustes.items():
        base[k] = base.get(k, 0.0) + v
            
    return base