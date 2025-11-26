import pandas as pd
import numpy as np
import logging
from pathlib import Path
from config import Config
import pipeline_preprocesamiento as pp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_pipeline_completo():
    """
    Orquesta todo el proceso de preprocesamiento con el ORDEN LÓGICO CORRECTO:
    Limpieza -> Agregación -> Filtros Post-Agregación -> Feature Engineering.
    """
    logger.info("="*50 + " INICIO PIPELINE PREPROCESAMIENTO " + "="*50)
    
    try:
        # --- 1. Definición de Rutas ---
        dir_brutos = Path(Config.DIR_SALIDA_BRUTOS)
        dir_entrada = Path(Config.DIR_ENTRADA)
        dir_procesados = Path(Config.DIR_SALIDA_PROCESADOS)
        dir_procesados.mkdir(parents=True, exist_ok=True)
        output_path = dir_procesados / "fichajes_preprocesados.csv"

        # --- 2. Carga, Merge y Transformación ---
        df_bruto = pp.cargar_y_unificar_datos_brutos(dir_brutos)
        
        logger.info(f"Iniciando merge de datos de contexto desde: '{dir_entrada}'")
        archivos_entrada = sorted(dir_entrada.glob("fichajes_*.csv"))
        if not archivos_entrada:
            raise FileNotFoundError(f"No se encontraron archivos de entrada en {dir_entrada}.")
        df_contexto_list = [pd.read_csv(f, sep=Config.CSV_SEPARATOR, dtype={'tm_id': str}) for f in archivos_entrada]
        df_contexto_unificado = pd.concat(df_contexto_list, ignore_index=True)
        df_contexto_to_merge = df_contexto_unificado[['tm_id', 'ss_season_id', 'club_destino']].copy()
        
        df_bruto['tm_id'] = df_bruto['tm_id'].astype(str)
        df_bruto['ss_season_id'] = df_bruto['ss_season_id'].astype(str)
        df_contexto_to_merge['tm_id'] = df_contexto_to_merge['tm_id'].astype(str)
        df_contexto_to_merge['ss_season_id'] = df_contexto_to_merge['ss_season_id'].astype(str)
        
        df_merged = pd.merge(df_bruto, df_contexto_to_merge, on=['tm_id', 'ss_season_id'], how='left')
        logger.info("Merge completado.")

        df_pre_limpio = pp.limpiar_filas(df_merged, minutos_threshold=0)
        df_agregado = pp.agregar_stats_por_temporada(df_pre_limpio)
        df_corregido = pp.corregir_anomalia_started_appearances(df_agregado)
        
        registros_antes_umbral = len(df_corregido)
        df_post_limpio = df_corregido[df_corregido['minutesPlayed'] >= 450].copy()
        eliminados_umbral = registros_antes_umbral - len(df_post_limpio)
        logger.info(f"Limpieza POST-agregación: Se eliminaron {eliminados_umbral} registros con menos de 450 minutos.")
        
        df_procesado = pp.crear_features_numericas(df_post_limpio)
        df_procesado = pp.crear_features_contextuales(df_procesado)
        df_procesado = pp.seleccionar_features_finales(df_procesado)

        # --- 3. Pasos Finales (Reordenar y Ordenar) ---
        metadata_cols = [c for c in ['tm_id', 'ss_id', 'nombre_jugador', 'season'] if c in df_procesado.columns]
        feature_cols = sorted([c for c in df_procesado.columns if c not in metadata_cols])
        df_reordenado = df_procesado[metadata_cols + feature_cols]
        
        logger.info("Ordenando el dataset final por 'season' y 'tm_id'...")
        df_final = df_reordenado.sort_values(by=['season', 'tm_id'], ascending=[True, True]).reset_index(drop=True)

        # --- 4. Guardado Final ---
        df_final.to_csv(output_path, sep=Config.CSV_SEPARATOR, index=False, decimal='.')
        
        logger.info("="*50 + " FIN PIPELINE PREPROCESAMIENTO " + "="*50)
        logger.info(f"✅ Pipeline completado. Guardado en: {output_path}")
        logger.info(f"📊 Dimensiones finales: {df_final.shape[0]} filas x {df_final.shape[1]} columnas")

    except Exception as e:
        logger.error("❌ Error fatal durante el pipeline.", exc_info=True)

if __name__ == "__main__":
    run_pipeline_completo()