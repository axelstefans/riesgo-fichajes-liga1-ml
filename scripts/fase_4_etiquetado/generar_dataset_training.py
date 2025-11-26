# scripts/generar_dataset_training.py
import pandas as pd
import numpy as np
import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config

try:
    sys.path.append(os.path.join(os.getcwd(), 'scripts', 'fase_4_etiquetado'))
    from pesos_dominio import get_pesos_por_posicion
except ImportError:
    logging.error("❌ Error importando pesos_dominio.")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DEL JUEZ ---
MIN_MINUTOS_JUGADOS = 450
PERCENTIL_RIESGO = 0.40
TEMPORADA_INICIAL_MINIMA = '2018_2019'  # ✅ NUEVO: Excluir 2017_2018

def reconstruir_posicion(row):
    if row.get('pos_Delantero') == 1: return 'Delantero'
    elif row.get('pos_Mediocampista') == 1: return 'Mediocampista'
    return 'Defensa'

def calcular_zscores_hibrido(df, features_pesos):
    df_z = df.copy()
    
    for feat in features_pesos:
        col_target = None
        col_futuro = f"{feat}_futuro"
        
        if col_futuro in df_z.columns:
            col_target = col_futuro
        elif feat == 'shots_p90':
            if 'totalShots_p90_futuro' in df_z.columns:
                col_target = 'totalShots_p90_futuro'
            elif 'shots_p90_futuro' in df_z.columns:
                col_target = 'shots_p90_futuro'
        elif feat in df_z.columns:
            col_target = feat
            
        if col_target:
            mean = df_z[col_target].mean()
            std = df_z[col_target].std()
            if std == 0: std = 1
            df_z[f"z_{feat}"] = (df_z[col_target] - mean) / std
            
    return df_z

def main():
    logger.info("🚀 INICIANDO FUSIÓN Y ETIQUETADO PREDICTIVO (EXPANDING WINDOW)")

    # 1. Rutas
    path_pasado = os.path.join(Config.DIR_SALIDA_PROCESADOS, "fichajes_preprocesados.csv")
    path_futuro = os.path.join(Config.DIR_SALIDA_PROCESADOS, "rendimiento_posterior_preprocesado.csv")
    path_final = os.path.join(Config.DIR_SALIDA_FINALES, "dataset_entrenamiento_final.csv")

    # 2. Cargar
    df_pasado = pd.read_csv(path_pasado, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
    df_futuro = pd.read_csv(path_futuro, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)

    for df in [df_pasado, df_futuro]:
        df['tm_id'] = df['tm_id'].astype(str)
    df_pasado['season'] = df_pasado['season'].astype(str)
    df_futuro['season_origen'] = df_futuro['season_origen'].astype(str)

    logger.info(f"📥 Pasado: {len(df_pasado)} | Futuro: {len(df_futuro)}")

    # 3. MERGE
    df_merged = pd.merge(
        df_pasado,
        df_futuro,
        left_on=['tm_id', 'season'],
        right_on=['tm_id', 'season_origen'],
        how='inner',
        suffixes=('', '_futuro') 
    )
    logger.info(f"🔗 Registros unidos: {len(df_merged)}")

    # ✅ NUEVO: Filtrar temporadas con datos insuficientes
    registros_antes = len(df_merged)
    df_merged = df_merged[df_merged['season'] >= TEMPORADA_INICIAL_MINIMA].copy()
    registros_despues = len(df_merged)
    registros_excluidos = registros_antes - registros_despues
    
    logger.info(f"⚠️  Excluidas temporadas < {TEMPORADA_INICIAL_MINIMA}: {registros_excluidos} registros")
    logger.info(f"✅ Registros restantes: {registros_despues}")

    # ✅ Ordenar temporalmente
    orden_temporal = [
        '2018_2019', '2019_2020', '2020_2021', '2021_2022',
        '2022_2023', '2023_2024', '2024_2025'
    ]
    df_merged['season'] = pd.Categorical(df_merged['season'], categories=orden_temporal, ordered=True)
    df_merged = df_merged.sort_values('season').reset_index(drop=True)
    logger.info("✅ Dataset ordenado temporalmente")

    # 4. ETIQUETADO
    df_merged['posicion_grupo'] = df_merged.apply(reconstruir_posicion, axis=1)
    df_merged['riesgo_fichaje'] = np.nan
    df_merged['score_futuro'] = 0.0

    # REGLA A: INACTIVIDAD
    col_minutos_fut = 'minutesPlayed_futuro' if 'minutesPlayed_futuro' in df_merged.columns else 'minutesPlayed'
    
    if col_minutos_fut not in df_merged.columns:
        logger.error("❌ No se encuentra minutesPlayed_futuro")
        return

    mask_noplay = df_merged[col_minutos_fut] < MIN_MINUTOS_JUGADOS
    df_merged.loc[mask_noplay, 'riesgo_fichaje'] = 1
    logger.info(f"📉 Inactividad (<{MIN_MINUTOS_JUGADOS}min): {mask_noplay.sum()}")

    # REGLA B: RENDIMIENTO (EXPANDING WINDOW)
    df_activos = df_merged[~mask_noplay].copy()
    
    if not df_activos.empty:
        # Calcular scores
        for pos in ['Defensa', 'Mediocampista', 'Delantero']:
            mask_pos = df_activos['posicion_grupo'] == pos
            if not mask_pos.any(): continue
            
            subset = df_activos[mask_pos].copy()
            pesos = get_pesos_por_posicion(pos)
            subset_z = calcular_zscores_hibrido(subset, list(pesos.keys()))
            
            scores = pd.Series(0.0, index=subset.index)
            for feat, weight in pesos.items():
                z_col = f"z_{feat}"
                if z_col in subset_z.columns:
                    scores += subset_z[z_col] * weight
            
            df_merged.loc[subset.index, 'score_futuro'] = scores

        # ✅ Expanding window por posición
        logger.info("⏳ Calculando umbrales históricos (expanding window)...")
        
        for pos in ['Defensa', 'Mediocampista', 'Delantero']:
            mask_pos_activos = (df_merged['posicion_grupo'] == pos) & (~mask_noplay)
            
            if mask_pos_activos.sum() == 0:
                continue
            
            indices_pos = df_merged[mask_pos_activos].index
            
            # Expanding quantile
            umbral_expandido = (
                df_merged.loc[indices_pos, 'score_futuro']
                .expanding()
                .quantile(PERCENTIL_RIESGO)
            )
            
            # Etiquetar
            es_riesgo_alto = df_merged.loc[indices_pos, 'score_futuro'] <= umbral_expandido
            df_merged.loc[indices_pos, 'riesgo_fichaje'] = es_riesgo_alto.astype(int)
            
            logger.info(f"   ✅ {pos}: {mask_pos_activos.sum()} etiquetados (expanding)")

    # 5. LIMPIEZA
    cols_pasado = df_pasado.columns.tolist()
    if 'riesgo_fichaje' in cols_pasado: 
        cols_pasado.remove('riesgo_fichaje')
    
    cols_finales = cols_pasado + ['riesgo_fichaje']
    df_final = df_merged[cols_finales].copy()
    
    # 6. Guardar
    df_final.to_csv(path_final, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING, index=False)
    
    conteo = df_final['riesgo_fichaje'].value_counts()
    logger.info("="*60)
    logger.info(f"✅ DATASET FINAL: {path_final}")
    logger.info(f"📊 Total: {len(df_final)} (excluidos {registros_excluidos} de 2017_2018)")
    logger.info(f"   🔴 Riesgo Alto (1): {conteo.get(1.0, 0)}")
    logger.info(f"   🟢 Riesgo Bajo (0): {conteo.get(0.0, 0)}")
    logger.info("="*60)

if __name__ == "__main__":
    main()