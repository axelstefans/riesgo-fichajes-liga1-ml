# scripts/verificar_balance.py
import pandas as pd
import logging
from pathlib import Path

# --- Configuración ---
# Asegúrate de que apunte a tu dataset final (el de 450min)
INPUT_FILE = "datos_salida/finales/dataset_etiquetado.csv"

# Usar la columna binaria de riesgo, no el score continuo
TARGET_COL = "riesgo_fichaje"

# Temporada que usarás como TEST (se excluye del train)
TEST_SEASON = "2024_2025"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("verificar_balance")

def main():
    filepath = Path(INPUT_FILE)
    if not filepath.exists():
        logger.error(f"¡ERROR! No se encontró el archivo '{INPUT_FILE}'")
        return

    # Lee el CSV separado por ';'
    df = pd.read_csv(filepath, sep=";", encoding="utf-8-sig")
    logger.info(f"Dataset cargado con {len(df)} filas y {len(df.columns)} columnas.")
    
    # Chequeo rápido de columnas
    logger.info(f"Columnas disponibles: {list(df.columns)}")
    
    if "season" not in df.columns:
        logger.error("¡ERROR! No se encontró la columna 'season' en el dataset.")
        return

    if TARGET_COL not in df.columns:
        logger.error(f"¡ERROR! No se encontró la columna target '{TARGET_COL}' en el dataset.")
        return

    # Aislar el set de entrenamiento (todo menos la última temporada)
    df_train = df[df["season"] != TEST_SEASON].copy()
    
    if df_train.empty:
        logger.error(
            "¡ERROR! El set de entrenamiento quedó vacío. "
            f"Revisa el valor de TEST_SEASON ('{TEST_SEASON}') y los valores reales de 'season'."
        )
        logger.info(f"Valores únicos en season: {df['season'].unique()}")
        return
    
    logger.info(
        f"📂 Verificando el balance del SET DE ENTRENAMIENTO "
        f"(Total: {len(df_train)} filas, temporada test excluida: {TEST_SEASON})"
    )
    
    # Aseguramos que la columna target no tenga NaNs
    df_train = df_train[~df_train[TARGET_COL].isna()].copy()
    if df_train.empty:
        logger.error(f"¡ERROR! Todas las filas del train tienen '{TARGET_COL}' nulo.")
        return

    # Calcular y mostrar el balance
    balance = df_train[TARGET_COL].value_counts().sort_index()
    total = len(df_train)
    
    # Asumimos problema binario 0 / 1
    clase_0_n = balance.get(0, 0)
    clase_1_n = balance.get(1, 0)
    
    clase_0_pct = (clase_0_n / total) * 100 if total > 0 else 0
    clase_1_pct = (clase_1_n / total) * 100 if total > 0 else 0

    logger.info("=" * 80)
    logger.info("⚖️ RESULTADO DEL BALANCE DE CLASES (en Train Set) ⚖️")
    logger.info(f"   Distribución completa de '{TARGET_COL}':")
    logger.info(f"\n{balance}\n")
    logger.info(f"   Clase 0 (Bajo Riesgo): {clase_0_n} registros ({clase_0_pct:.1f}%)")
    logger.info(f"   Clase 1 (Alto Riesgo): {clase_1_n} registros ({clase_1_pct:.1f}%)")
    logger.info("=" * 80)
    
    if abs(clase_0_pct - clase_1_pct) < 5:
        logger.info(
            "✅ Veredicto: El dataset está BALANCEADO. "
            "NO se debe usar 'class_weight' o 'scale_pos_weight'."
        )
    else:
        logger.info(
            "⚠️ Veredicto: El dataset está DESBALANCEADO. "
            "SÍ se debe usar 'class_weight' o 'scale_pos_weight'."
        )

if __name__ == "__main__":
    main()
