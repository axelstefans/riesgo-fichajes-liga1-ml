import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook")

INPUT_FILE = Path("datos_salida/preprocesados/fichajes_preprocesados.csv")
OUTPUT_DIR = Path("reports/EDA_preprocesado")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLORES_POSICION = {
    'Defensa': '#FF6B6B',
    'Mediocampista': '#4ECDC4',
    'Delantero': '#45B7D1'
}
ORDEN_POSICION = ['Defensa', 'Mediocampista', 'Delantero']

def cargar_datos():
    logger.info(f"📂 Cargando datos desde: {INPUT_FILE}")
    try:
        df = pd.read_csv(INPUT_FILE, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
        logger.info(f"✅ Datos cargados: {len(df)} registros, {df.shape[1]} columnas")
        
        if 'pos_Delantero' in df.columns and 'pos_Mediocampista' in df.columns:
            df['posicion_original'] = 'Defensa'
            df.loc[df['pos_Delantero'] == 1, 'posicion_original'] = 'Delantero'
            df.loc[df['pos_Mediocampista'] == 1, 'posicion_original'] = 'Mediocampista'
            logger.info("  ✅ Columna 'posicion_original' reconstruida desde dummies para EDA.")
        else:
            logger.error("❌ No se encontraron dummies de posición.")
            return None
            
        return df
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return None

def generar_estadisticas_descriptivas(df):
    logger.info("Generando estadísticas descriptivas...")
    output_file = OUTPUT_DIR / "02_estadisticas_descriptivas.txt"
    
    exclude_from_stats = ['tm_id', 'ss_id']
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_from_stats]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("ESTADÍSTICAS DESCRIPTIVAS - DATASET PREPROCESADO\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"Total de registros: {len(df)}\n")
        f.write(f"Total de features: {len(numeric_cols)}\n")
        f.write(f"Ratio: {len(df) / len(numeric_cols):.2f}\n\n")
        
        f.write("LISTA DE FEATURES:\n")
        f.write("-" * 100 + "\n")
        for i, feat in enumerate(sorted(numeric_cols), 1):
            f.write(f"{i:2d}. {feat}\n")
        f.write("\n")
        
        f.write("ESTADÍSTICAS:\n")
        f.write("-" * 100 + "\n")
        f.write(df[numeric_cols].describe().to_string())
        f.write("\n\n")
        
        f.write("DISTRIBUCIÓN POR POSICIÓN:\n")
        f.write("-" * 100 + "\n")
        dist_pos = df['posicion_original'].value_counts().sort_index()
        for pos, count in dist_pos.items():
            f.write(f"{pos:15s}: {count:4d} ({(count/len(df)*100):5.1f}%)\n")
        f.write("\n")
        
        f.write("="*100 + "\n")
    
    logger.info(f"✅ Reporte guardado en: {output_file}")

def grafico_1_matriz_correlacion(df):
    logger.info("Generando matriz de correlación...")
    
    cols_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_corr = [c for c in cols_numericas if c not in ['tm_id', 'ss_id']]
    
    logger.info(f"  Total de features: {len(cols_corr)}")
    logger.info(f"  Mostrando top 20 por varianza...")
    variances = df[cols_corr].var().sort_values(ascending=False)
    cols_seleccionadas = variances.head(20).index.tolist()

    corr = df[cols_seleccionadas].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    plt.figure(figsize=(14, 12))
    
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", 
                cmap='RdYlGn_r', center=0, 
                square=True, linewidths=0.5, 
                cbar_kws={"shrink": 0.8},
                annot_kws={"size": 9})
    
    plt.title('Matriz de Correlación (Top 20 Features por Varianza)', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_matriz_correlacion_completa.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    pares_altos = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) > 0.80:
                pares_altos.append((corr.columns[i], corr.columns[j], corr.iloc[i, j]))
    
    if pares_altos:
        logger.warning(f"  ⚠️ {len(pares_altos)} pares con |r| > 0.80:")
        for f1, f2, r in pares_altos:
            logger.warning(f"     {f1} ↔ {f2}: {r:.3f}")
    else:
        logger.info(f"  ✅ Sin correlaciones > 0.80")

def grafico_2_histogramas_p90(df):
    logger.info("Generando histogramas...")
    
    cols_plot = [
        'goals_p90', 'assists_p90', 'clearances_p90',
        'shotsOnTarget_p90', 'keyPasses_p90', 'aerialDuelsWon_p90',
        'yellowCards_p90', 'fouls_p90', 'accurateCrosses_p90'
    ]
    cols_plot = [c for c in cols_plot if c in df.columns]
    
    n_rows = (len(cols_plot) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten()
    
    for i, col in enumerate(cols_plot):
        axes[i].hist(df[col], bins=30, color='#4ECDC4', edgecolor='white', alpha=0.8)
        
        median_val = df[col].median()
        axes[i].axvline(median_val, color='red', linestyle='--', linewidth=2)
        
        axes[i].set_title(col.replace('_p90', '').replace('_', ' ').title(), fontsize=12)
        axes[i].set_ylabel('Frecuencia')
        axes[i].grid(axis='y', alpha=0.3)
    
    for i in range(len(cols_plot), len(axes)):
        fig.delaxes(axes[i])

    plt.suptitle('Distribución de Features por 90 Minutos', fontsize=14, y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_histogramas_features_p90.png", dpi=300, bbox_inches='tight')
    plt.close()

def crear_boxplot_generico(df, metrica, titulo, archivo):
    logger.info(f"Generando boxplot: {metrica}...")
    
    if metrica not in df.columns:
        logger.warning(f"⚠️ {metrica} no existe. Saltando.")
        return

    plt.figure(figsize=(10, 6))
    
    sns.boxplot(data=df, x='posicion_original', y=metrica, 
                order=ORDEN_POSICION, palette=COLORES_POSICION,
                showfliers=True, linewidth=1.5)
    
    plt.title(titulo, fontsize=13, pad=12)
    plt.ylabel(metrica.replace('_p90', '').replace('_', ' ').title(), fontsize=11)
    plt.xlabel('Posición', fontsize=11)
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / archivo, dpi=300, bbox_inches='tight')
    plt.close()

def grafico_6_minutos(df):
    logger.info("Generando distribución de minutos...")
    
    if 'minutesPlayed' not in df.columns:
        return
    
    plt.figure(figsize=(10, 5))
    
    plt.hist(df['minutesPlayed'], bins=40, color='#4ECDC4', edgecolor='white', alpha=0.8)
    
    plt.axvline(450, color='red', linestyle='--', linewidth=2, label='Filtro (450 min)')
    plt.axvline(df['minutesPlayed'].median(), color='green', linestyle='--', linewidth=2, 
               label=f"Mediana ({df['minutesPlayed'].median():.0f} min)")

    plt.title('Distribución de Minutos Jugados', fontsize=13)
    plt.xlabel('Minutos Totales')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_distribucion_minutesPlayed.png", dpi=300, bbox_inches='tight')
    plt.close()

def main():
    logger.info("="*80)
    logger.info("🚀 EDA PREPROCESADO")
    logger.info("="*80)
    
    df = cargar_datos()
    if df is None:
        return

    generar_estadisticas_descriptivas(df)
    grafico_1_matriz_correlacion(df)
    grafico_2_histogramas_p90(df)
    
    crear_boxplot_generico(df, 'goals_p90', 
                           'Goles por 90 min (Por Posición)', 
                           "04_boxplot_goals_p90_por_posicion.png")
    
    crear_boxplot_generico(df, 'assists_p90', 
                           'Asistencias por 90 min (Por Posición)', 
                           "05_boxplot_assists_p90_por_posicion.png")
    
    crear_boxplot_generico(df, 'clearances_p90', 
                           'Despejes por 90 min (Por Posición)', 
                           "06_boxplot_clearances_p90_por_posicion.png")
    
    grafico_6_minutos(df)
    
    logger.info("="*80)
    logger.info(f"✅ COMPLETADO. Gráficos en: {OUTPUT_DIR}")
    logger.info("="*80)

if __name__ == "__main__":
    main()