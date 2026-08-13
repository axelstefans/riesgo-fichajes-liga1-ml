"""
core/constants.py
Centralized constants for the Riesgo Fichajes ML pipeline and Streamlit app.
"""

# ========== POSITIONAL MAPPINGS ==========
POS_MAP = {
    'Delantero': 'Delantero',
    'Mediocampista': 'Mediocampista',
    'Defensa': 'Defensa',
    'Defensa central': 'Defensa',
    'Líbero': 'Defensa',
    'Lateral izquierdo': 'Defensa',
    'Lateral derecho': 'Defensa',
    'Pivote': 'Mediocampista',
    'Mediocentro': 'Mediocampista',
    'Mediocentro ofensivo': 'Mediocampista',
    'Interior izquierdo': 'Mediocampista',
    'Interior derecho': 'Mediocampista',
    'Extremo izquierdo': 'Delantero',
    'Extremo derecho': 'Delantero',
    'Mediapunta': 'Delantero',
    'Segundo delantero': 'Delantero',
    'Delantero centro': 'Delantero'
}

# ========== FEATURE ENGINEERING COLUMNS ==========
COLUMNAS_NUMERICAS_ORIGINALES = [
    'totalPasses', 'accuratePasses', 'totalShots', 'shotsOnTarget', 'penaltyGoals', 
    'penaltiesTaken', 'started', 'goals', 'assists', 'shotsOffTarget', 'blockedShots', 
    'keyPasses', 'bigChancesCreated', 'bigChancesMissed', 'successfulDribbles', 
    'penaltiesWon', 'offsides', 'tackles', 'interceptions', 'clearances', 
    'dribbledPast', 'penaltiesCommitted', 'fouls', 'wasFouled', 'aerialDuelsWon', 
    'groundDuelsWon', 'accurateFinalThirdPasses', 'accurateLongBalls', 
    'accurateCrosses', 'possessionLost', 'dispossessed', 'yellowCards', 'redCards',
    'appearances'
]

COLUMNAS_CONTEXTUALES_ORIGINALES = ['posicion', 'nacionalidad_str', 'club_origen', 'club_destino']

COLUMNAS_ENGINEERED_REDUNDANTES = ['totalShots_p90', 'appearances_p90']

FEATURES_BAJA_RELEVANCIA = [
    'penaltiesWon_p90',
    'penaltiesCommitted_p90',
    'redCards_p90'
]

FEATURES_MULTICOLINEALES = ['startPercentage']

COLUMNAS_COMPLEJAS_INCOMPATIBLES = [
    'possessionLost_p90',
    'dispossessed_p90',
    'tackles_p90',
    'interceptions_p90',
    'groundDuelsWon_p90',
    'bigChancesCreated_p90',
    'bigChancesMissed_p90'
]

# The master list of raw and redundant columns to drop before model training
COLUMNAS_A_ELIMINAR = (
    COLUMNAS_NUMERICAS_ORIGINALES + 
    COLUMNAS_CONTEXTUALES_ORIGINALES + 
    COLUMNAS_ENGINEERED_REDUNDANTES +
    FEATURES_BAJA_RELEVANCIA +
    FEATURES_MULTICOLINEALES +
    COLUMNAS_COMPLEJAS_INCOMPATIBLES
)
