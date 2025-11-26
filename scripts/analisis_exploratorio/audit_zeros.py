import pandas as pd
import logging
from pathlib import Path
from config import Config

# Configuración de logs
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

INPUT_FILE = Path("datos_salida/preprocesados/fichajes_preprocesados.csv")
OUTPUT_FILE = Path("reports/reporte_ceros.txt")

def main():
    logger.info("=" * 80)
    logger.info("🕵️‍♂️ AUDITORÍA DE CEROS (ZERO-INFLATION CHECK)")
    logger.info("=" * 80)

    if not INPUT_FILE.exists():
        logger.error(f"❌ No se encontró el archivo: {INPUT_FILE}")
        return

    df = pd.read_csv(INPUT_FILE, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
    
    # Seleccionar solo columnas numéricas (excluyendo IDs y metadatos)
    cols_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    cols_excluir = ['tm_id', 'ss_id', 'edad', 'minutesPlayed', 'season']
    cols_analisis = [c for c in cols_numericas if c not in cols_excluir]

    resultados = []

    for col in cols_analisis:
        num_ceros = (df[col] == 0).sum()
        total = len(df)
        pct_ceros = (num_ceros / total) * 100
        
        resultados.append({
            'Feature': col,
            'Ceros': num_ceros,
            'Porcentaje': pct_ceros
        })

    # Crear DataFrame de resultados y ordenar
    df_res = pd.DataFrame(resultados).sort_values('Porcentaje', ascending=False)

    # Imprimir en consola
    logger.info(f"\n📊 ANÁLISIS DE {len(cols_analisis)} VARIABLES NUMÉRICAS:\n")
    print(df_res.to_string(index=False, formatters={'Porcentaje': '{:.1f}%'.format}))

    # Guardar reporte
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("REPORTE DE PREVALENCIA DE CEROS\n")
        f.write("="*50 + "\n")
        f.write(df_res.to_string(index=False, formatters={'Porcentaje': '{:.1f}%'.format}))
    
    logger.info(f"\n💾 Reporte guardado en: {OUTPUT_FILE}")

    # Interpretación automática rápida
    top_feature = df_res.iloc[0]['Feature']
    top_val = df_res.iloc[0]['Porcentaje']
    
    logger.info("-" * 80)
    logger.info("💡 INTERPRETACIÓN RÁPIDA:")
    logger.info(f"   La variable con más ceros es '{top_feature}' ({top_val:.1f}%).")
    if top_val > 90:
        logger.info("   ⚠️ ALERTA: Más del 90% son ceros. Considera si esta variable aporta valor al modelo.")
        logger.info("      (Ej: penaltyGoals_p90 es normal que sea alto en ceros, pero accurateCrosses no tanto).")
    else:
        logger.info("   ✅ Los niveles de ceros parecen razonables para métricas de fútbol.")

if __name__ == "__main__":
    main()