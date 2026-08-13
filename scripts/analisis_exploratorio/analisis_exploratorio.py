import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import Config
from core.constants import POS_MAP

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

INPUT_FILE = Path("datos_salida/brutos/datos_brutos_merged.csv")
OUTPUT_DIR = Path("reports/EDA")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def cargar_datos():
    logger.info(f"Cargando datos desde: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, sep=Config.CSV_SEPARATOR, encoding=Config.CSV_ENCODING)
    logger.info(f"✅ Datos cargados: {len(df)} registros, {df.shape[1]} columnas")
    return df

def generar_estadisticas_descriptivas(df):
    logger.info("Generando estadísticas descriptivas...")
    
    output_file = OUTPUT_DIR / "estadisticas_descriptivas.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("ESTADÍSTICAS DESCRIPTIVAS - DATASET CRUDO\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total de registros: {len(df)}\n")
        f.write(f"Total de columnas: {df.shape[1]}\n\n")
        
        f.write("INFORMACIÓN GENERAL DEL DATASET:\n")
        f.write("-" * 80 + "\n")
        from io import StringIO
        buffer = StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())
        
        f.write("VALORES NULOS POR COLUMNA:\n")
        f.write("-" * 80 + "\n")
        nulos = df.isnull().sum()
        nulos_pct = (nulos / len(df) * 100).round(2)
        nulos_df = pd.DataFrame({'Nulos': nulos, 'Porcentaje': nulos_pct})
        nulos_df = nulos_df[nulos_df['Nulos'] > 0].sort_values('Nulos', ascending=False)
        if len(nulos_df) > 0:
            f.write(nulos_df.to_string())
        else:
            f.write("No hay valores nulos en el dataset.\n")
        f.write("\n\n")
        
        f.write("ESTADÍSTICAS DESCRIPTIVAS - MÉTRICAS NUMÉRICAS:\n")
        f.write("-" * 80 + "\n")
        metricas_clave = [
            'minutesPlayed', 'appearances', 'started', 'goals', 'assists',
            'totalShots', 'shotsOnTarget', 'totalPasses', 'accuratePasses',
            'tackles', 'interceptions', 'clearances', 'yellowCards', 'redCards'
        ]
        metricas_disponibles = [m for m in metricas_clave if m in df.columns]
        f.write(df[metricas_disponibles].describe().to_string())
        f.write("\n\n")
        
        f.write("DISTRIBUCIÓN DE VARIABLES CATEGÓRICAS:\n")
        f.write("-" * 80 + "\n")
        
        if 'posicion' in df.columns:
            f.write("\nDISTRIBUCIÓN POR POSICIÓN:\n")
            pos_counts = df['posicion'].value_counts()
            pos_pct = (pos_counts / len(df) * 100).round(2)
            pos_df = pd.DataFrame({'Cantidad': pos_counts, 'Porcentaje': pos_pct})
            f.write(pos_df.to_string())
            f.write("\n")
        
        if 'nacionalidad_str' in df.columns:
            f.write("\nTOP 10 NACIONALIDADES:\n")
            nac_counts = df['nacionalidad_str'].value_counts().head(10)
            nac_pct = (nac_counts / len(df) * 100).round(2)
            nac_df = pd.DataFrame({'Cantidad': nac_counts, 'Porcentaje': nac_pct})
            f.write(nac_df.to_string())
            f.write("\n")
        
        if 'season' in df.columns:
            f.write("\nDISTRIBUCIÓN POR TEMPORADA:\n")
            season_counts = df['season'].value_counts().sort_index()
            season_pct = (season_counts / len(df) * 100).round(2)
            season_df = pd.DataFrame({'Cantidad': season_counts, 'Porcentaje': season_pct})
            f.write(season_df.to_string())
            f.write("\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("FIN DEL REPORTE\n")
        f.write("=" * 80 + "\n")
    
    logger.info(f"✅ Estadísticas guardadas en: {output_file}")

def grafico_1_distribucion_minutos(df):
    logger.info("Generando Gráfico 1: Distribución de minutesPlayed...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(df['minutesPlayed'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(df['minutesPlayed'].mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {df["minutesPlayed"].mean():.0f} min')
    ax.axvline(df['minutesPlayed'].median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {df["minutesPlayed"].median():.0f} min')
    ax.axvline(450, color='orange', linestyle='--', linewidth=2, label='Umbral filtrado: 450 min')
    
    ax.set_xlabel('Minutos Jugados', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Minutos Jugados (Dataset Crudo)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "01_distribucion_minutos_jugados.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")

def grafico_2_boxplot_metricas_posicion(df):
    logger.info("Generando Gráfico 2: Boxplot de métricas ofensivas por posición...")
    
    if 'posicion' not in df.columns:
        logger.warning("⚠️ Columna 'posicion' no encontrada. Saltando gráfico 2.")
        return
    
    df_plot = df.copy()
    df_plot['posicion_agrupada'] = df_plot['posicion'].map(POS_MAP)
    df_plot = df_plot[df_plot['posicion_agrupada'].notna()]
    
    metricas = ['goals', 'assists', 'totalShots']
    metricas_disponibles = [m for m in metricas if m in df_plot.columns]
    
    fig, axes = plt.subplots(1, len(metricas_disponibles), figsize=(16, 6))
    
    if len(metricas_disponibles) == 1:
        axes = [axes]
    
    for idx, metrica in enumerate(metricas_disponibles):
        sns.boxplot(data=df_plot, x='posicion_agrupada', y=metrica, ax=axes[idx], palette='Set2')
        axes[idx].set_xlabel('Posición', fontsize=11, fontweight='bold')
        axes[idx].set_ylabel(metrica.replace('_', ' ').title(), fontsize=11, fontweight='bold')
        axes[idx].set_title(f'Distribución de {metrica.replace("_", " ").title()} por Posición', fontsize=12, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "02_boxplot_metricas_por_posicion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")

def grafico_3_top_nacionalidades(df):
    logger.info("Generando Gráfico 3: Top 10 nacionalidades...")
    
    if 'nacionalidad_str' not in df.columns:
        logger.warning("⚠️ Columna 'nacionalidad_str' no encontrada. Saltando gráfico 3.")
        return
    
    top_10 = df['nacionalidad_str'].value_counts().head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.barh(range(len(top_10)), top_10.values, color='coral', edgecolor='black')
    ax.set_yticks(range(len(top_10)))
    ax.set_yticklabels(top_10.index)
    ax.set_xlabel('Cantidad de Jugadores', fontsize=12, fontweight='bold')
    ax.set_title('Top 10 Nacionalidades en el Dataset', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (nac, count) in enumerate(top_10.items()):
        pct = (count / len(df)) * 100
        ax.text(count + 5, i, f'{count} ({pct:.1f}%)', va='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "03_top_10_nacionalidades.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")

def grafico_4_registros_temporada(df):
    logger.info("Generando Gráfico 4: Registros por temporada...")
    
    if 'season' not in df.columns:
        logger.warning("⚠️ Columna 'season' no encontrada. Saltando gráfico 4.")
        return
    
    season_counts = df['season'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(range(len(season_counts)), season_counts.values, color='mediumseagreen', edgecolor='black', alpha=0.8)
    ax.set_xticks(range(len(season_counts)))
    ax.set_xticklabels(season_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Temporada', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cantidad de Registros', fontsize=12, fontweight='bold')
    ax.set_title('Distribución de Registros por Temporada', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(season_counts.values):
        ax.text(i, count + 5, str(count), ha='center', fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "04_registros_por_temporada.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")

def grafico_5_matriz_correlacion(df):
    logger.info("Generando Gráfico 5: Matriz de correlación...")
    
    metricas_correlacion = [
        'goals', 'assists', 'totalShots', 'shotsOnTarget', 'totalPasses',
        'accuratePasses', 'tackles', 'interceptions', 'minutesPlayed', 'started'
    ]
    
    metricas_disponibles = [m for m in metricas_correlacion if m in df.columns]
    
    if len(metricas_disponibles) < 2:
        logger.warning("⚠️ No hay suficientes métricas numéricas. Saltando gráfico 5.")
        return
    
    df_corr = df[metricas_disponibles].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(df_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=0.5, cbar_kws={'label': 'Correlación'}, ax=ax)
    
    ax.set_title('Matriz de Correlación - Métricas Clave (Datos Crudos)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "05_matriz_correlacion.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")

def grafico_6_deteccion_outliers(df):
    logger.info("Generando Gráfico 6: Detección de outliers...")
    
    metricas_outliers = ['goals', 'assists', 'totalShots', 'tackles', 'minutesPlayed', 'yellowCards']
    metricas_disponibles = [m for m in metricas_outliers if m in df.columns]
    
    if len(metricas_disponibles) == 0:
        logger.warning("⚠️ No hay métricas disponibles para detección de outliers. Saltando gráfico 6.")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for idx, metrica in enumerate(metricas_disponibles):
        if idx < len(axes):
            sns.boxplot(y=df[metrica], ax=axes[idx], color='lightblue')
            axes[idx].set_ylabel(metrica.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            axes[idx].set_title(f'Detección de Outliers: {metrica.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[idx].grid(axis='y', alpha=0.3)
    
    for idx in range(len(metricas_disponibles), len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Detección de Outliers en Métricas Clave', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    output_path = OUTPUT_DIR / "06_deteccion_outliers.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")

def main():
    logger.info("=" * 80)
    logger.info("INICIO - ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    logger.info("=" * 80)
    
    df = cargar_datos()
    
    generar_estadisticas_descriptivas(df)
    
    grafico_1_distribucion_minutos(df)
    grafico_2_boxplot_metricas_posicion(df)
    grafico_3_top_nacionalidades(df)
    grafico_4_registros_temporada(df)
    grafico_5_matriz_correlacion(df)
    grafico_6_deteccion_outliers(df)
    
    logger.info("=" * 80)
    logger.info("✅ ANÁLISIS EXPLORATORIO COMPLETADO")
    logger.info(f"📁 Todos los gráficos guardados en: {OUTPUT_DIR}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()