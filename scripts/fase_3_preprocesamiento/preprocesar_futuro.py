import pandas as pd
import numpy as np
import os
import sys
import logging

# Ajuste de rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config

# Importamos la función matemática de tu pipeline
try:
    sys.path.append(os.path.join(os.getcwd(), 'scripts', 'fase_3_preprocesamiento'))
    from pipeline_preprocesamiento import crear_features_numericas
except ImportError:
    logging.error("❌ No se pudo importar pipeline_preprocesamiento.py")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def limpiar_columnas_segun_logica_usuario(df):
    """
    Replica EXACTAMENTE tu lógica de eliminación de columnas
    para que el dataset futuro quede limpio y ligero.
    """
    logger.info("🧹 Eliminando columnas crudas e irrelevantes...")
    
    # 1. Columnas Crudas (Raw) -> Se van porque ya tendremos los _p90
    columnas_numericas_originales = [
        'totalPasses', 'accuratePasses', 'totalShots', 'shotsOnTarget', 'penaltyGoals', 
        'penaltiesTaken', 'started', 'goals', 'assists', 'shotsOffTarget', 'blockedShots', 
        'keyPasses', 'bigChancesCreated', 'bigChancesMissed', 'successfulDribbles', 
        'penaltiesWon', 'offsides', 'tackles', 'interceptions', 'clearances', 
        'dribbledPast', 'penaltiesCommitted', 'fouls', 'wasFouled', 'aerialDuelsWon', 
        'groundDuelsWon', 'accurateFinalThirdPasses', 'accurateLongBalls', 
        'accurateCrosses', 'possessionLost', 'dispossessed', 'yellowCards', 'redCards',
        'appearances'
    ]
    
    # 2. Columnas Contextuales del futuro (No las necesitamos, usaremos las del pasado)
    # Nota: En el bruto no suelen venir, pero por si acaso.
    columnas_contextuales_originales = ['posicion', 'nacionalidad_str', 'club_origen', 'club_destino']

    # 3. Redundantes
    columnas_engineered_redundantes = ['totalShots_p90', 'appearances_p90']

    # 4. Baja Relevancia
    features_baja_relevancia = [
        'penaltiesWon_p90',
        'penaltiesCommitted_p90',
        'redCards_p90'
    ]
    
    # 5. Multicolineales
    features_multicolineales = ['startPercentage']

    # 6. Complejas Incompatibles
    columnas_complejas_incompatibles = [
        'possessionLost_p90',
        'dispossessed_p90',
        'tackles_p90',
        'interceptions_p90',
        'groundDuelsWon_p90',
        'bigChancesCreated_p90',
        'bigChancesMissed_p90'
    ]

    # Unimos todas las listas de eliminación
    columnas_a_eliminar = (
        columnas_numericas_originales + 
        columnas_contextuales_originales + 
        columnas_engineered_redundantes +
        features_baja_relevancia +
        features_multicolineales +
        columnas_complejas_incompatibles
    )

    # Ejecutamos el borrado seguro (solo si existen)
    existentes = [c for c in columnas_a_eliminar if c in df.columns]
    df_final = df.drop(columns=existentes)
    
    logger.info(f"🗑️ Se eliminaron {len(existentes)} columnas.")
    return df_final

def main():
    logger.info("🛠️ PREPROCESANDO FUTURO (VERSIÓN LIMPIA)")

    # Rutas
    input_path = os.path.join(Config.DIR_SALIDA_BRUTOS, "rendimiento_posterior_bruto.csv")
    output_path = os.path.join(Config.DIR_SALIDA_PROCESADOS, "rendimiento_posterior_preprocesado.csv")

    if not os.path.exists(input_path):
        logging.error(f"❌ No existe: {input_path}")
        return

    # 1. Cargar
    df = pd.read_csv(input_path, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
    
    # 2. Limpieza de Tipos
    cols_id = ['tm_id', 'ss_id', 'season_origen', 'season_futuro_id', 'tipo_dato']
    for col in df.columns:
        if col not in cols_id:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # 3. Crear Features Matemáticas (p90, %, etc.)
    # Esto usa tu función validada
    df_calculado = crear_features_numericas(df)

    # 4. Eliminar Basura (Usando tus listas exactas)
    df_limpio = limpiar_columnas_segun_logica_usuario(df_calculado)

    # 5. Guardar
    df_limpio.to_csv(output_path, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING, index=False)
    
    logger.info(f"✅ Futuro limpio guardado: {output_path}")
    logger.info(f"📊 Columnas finales: {len(df_limpio.columns)}")

if __name__ == "__main__":
    main()