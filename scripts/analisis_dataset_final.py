import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from config import Config

# =====================================================
# CONFIGURACIÓN
# =====================================================

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)
sns.set_palette("husl")

INPUT_FILE = Path("datos_salida/finales/dataset_entrenamiento_final.csv")
OUTPUT_DIR = Path("reports/EDA_final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_BAJO_RIESGO = '#27AE60'
COLOR_ALTO_RIESGO = '#E74C3C'
COLORES_CLASES = [COLOR_BAJO_RIESGO, COLOR_ALTO_RIESGO]

# =====================================================
# FUNCIONES AUXILIARES
# =====================================================

def cargar_datos():
    """Carga el dataset final de entrenamiento."""
    logger.info(f"📂 Cargando datos desde: {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"❌ No se encontró el archivo: {INPUT_FILE}\n"
            f"Ejecuta primero: python scripts/fase_4_etiquetado/generar_dataset_training.py"
        )
    
    df = pd.read_csv(INPUT_FILE, sep=';', encoding='utf-8')
    logger.info(f"✅ Datos cargados: {len(df)} registros, {df.shape[1]} columnas")
    
    if 'riesgo_fichaje' not in df.columns:
        raise ValueError("❌ El dataset no contiene la columna 'riesgo_fichaje'")
    
    balance = df['riesgo_fichaje'].value_counts()
    logger.info(f"📊 Balance de clases:")
    logger.info(f"   - Bajo Riesgo (0): {balance.get(0, 0)} ({balance.get(0, 0)/len(df)*100:.1f}%)")
    logger.info(f"   - Alto Riesgo (1): {balance.get(1, 0)} ({balance.get(1, 0)/len(df)*100:.1f}%)")
    
    return df


def generar_reporte_estadistico(df):
    """Genera reporte estadístico descriptivo."""
    logger.info("📝 Generando reporte estadístico...")
    
    output_file = OUTPUT_DIR / "00_reporte_estadistico.txt"
    
    exclude_cols = ['tm_id', 'ss_id', 'riesgo_fichaje']
    numeric_features = [
        col for col in df.select_dtypes(include=[np.number]).columns 
        if col not in exclude_cols
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("REPORTE ESTADÍSTICO - DATASET FINAL DE ENTRENAMIENTO\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"📊 INFORMACIÓN GENERAL\n")
        f.write("-" * 100 + "\n")
        f.write(f"Total de observaciones: {len(df)}\n")
        f.write(f"Total de features predictoras: {len(numeric_features)}\n")
        f.write(f"Ratio observaciones/features: {len(df)/len(numeric_features):.2f}\n\n")
        
        f.write(f"🎯 DISTRIBUCIÓN DE LA VARIABLE OBJETIVO\n")
        f.write("-" * 100 + "\n")
        balance = df['riesgo_fichaje'].value_counts()
        f.write(f"Bajo Riesgo (0): {balance.get(0, 0):4d} ({balance.get(0, 0)/len(df)*100:5.2f}%)\n")
        f.write(f"Alto Riesgo (1): {balance.get(1, 0):4d} ({balance.get(1, 0)/len(df)*100:5.2f}%)\n\n")
        
        f.write(f"📋 LISTA DE FEATURES PREDICTORAS\n")
        f.write("-" * 100 + "\n")
        for i, feat in enumerate(sorted(numeric_features), 1):
            f.write(f"{i:2d}. {feat}\n")
        f.write("\n")
        
        f.write(f"📊 ESTADÍSTICAS DESCRIPTIVAS POR CLASE\n")
        f.write("-" * 100 + "\n")
        f.write("\nBAJO RIESGO (0):\n")
        f.write(df[df['riesgo_fichaje'] == 0][numeric_features].describe().to_string())
        f.write("\n\nALTO RIESGO (1):\n")
        f.write(df[df['riesgo_fichaje'] == 1][numeric_features].describe().to_string())
        f.write("\n\n")
        
        f.write("=" * 100 + "\n")
    
    logger.info(f"✅ Reporte guardado: {output_file}")


# =====================================================
# GRÁFICO 1: DISTRIBUCIÓN DEL TARGET
# =====================================================

def grafico_1_balance_clases(df):
    """Balance de clases - Distribución de la variable objetivo."""
    logger.info("📊 Generando Gráfico 1: Balance de clases...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    target_counts = df['riesgo_fichaje'].value_counts().sort_index()
    labels = ['Bajo Riesgo (0)', 'Alto Riesgo (1)']
    colors = COLORES_CLASES
    
    bars = ax.bar(labels, target_counts.values, color=colors, 
                   edgecolor='black', linewidth=2, alpha=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}\n({height/len(df)*100:.1f}%)',
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    
    ax.axhline(len(df)/2, color='gray', linestyle='--', 
               linewidth=1.5, alpha=0.5, label='Balance perfecto (50%)')
    
    ax.set_ylabel('Cantidad de Observaciones', fontsize=13, fontweight='bold')
    ax.set_xlabel('Clase de Riesgo de Fichaje', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribución de la Variable Objetivo: Riesgo de Fichaje\n'
        '(Balance de Clases en el Dataset de Entrenamiento)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(target_counts.values) * 1.15)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "01_balance_clases.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")


# =====================================================
# GRÁFICO 2: BOXPLOT fouls_p90
# =====================================================

def grafico_2_boxplot_fouls(df):
    """
    Boxplot de faltas por 90 minutos por clase.
    Feature #1 más discriminativa (Cohen's d=0.47).
    """
    logger.info("📊 Generando Gráfico 2: Boxplot fouls_p90...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    data_to_plot = [
        df[df['riesgo_fichaje'] == 0]['fouls_p90'],
        df[df['riesgo_fichaje'] == 1]['fouls_p90']
    ]
    
    bp = ax.boxplot(data_to_plot, 
                    tick_labels=['Bajo Riesgo', 'Alto Riesgo'],
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='yellow', 
                                  markeredgecolor='black', markersize=8))
    
    for patch, color in zip(bp['boxes'], COLORES_CLASES):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    
    for i, clase in enumerate([0, 1]):
        datos_clase = df[df['riesgo_fichaje'] == clase]['fouls_p90']
        media = datos_clase.mean()
        mediana = datos_clase.median()
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, 
                f'Media: {media:.2f}\nMediana: {mediana:.2f}',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Faltas Cometidas por 90 Minutos', fontsize=13, fontweight='bold')
    ax.set_xlabel('Clase de Riesgo de Fichaje', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribución de Faltas por 90 Minutos según Clase de Riesgo\n'
        '(Feature #1: Cohen\'s d=0.47, p<0.001)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "02_boxplot_fouls_p90.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")


# =====================================================
# GRÁFICO 3: BARRAS AGRUPADAS contexto_equipo_top
# =====================================================

def grafico_3_barras_contexto(df):
    """
    Barras agrupadas de contexto_equipo_top por clase.
    Feature #2 más discriminativa (Cohen's d=0.38).
    """
    logger.info("📊 Generando Gráfico 3: Barras contexto_equipo_top...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calcular proporciones
    crosstab = pd.crosstab(df['contexto_equipo_top'], df['riesgo_fichaje'], normalize='columns') * 100
    
    x = np.arange(2)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, crosstab[0], width, 
                   label='Bajo Riesgo', color=COLOR_BAJO_RIESGO, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, crosstab[1], width, 
                   label='Alto Riesgo', color=COLOR_ALTO_RIESGO, 
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Agregar valores
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Porcentaje dentro de cada Clase (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Va a Equipo Top 4 de Liga 1', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribución de Destino (Equipo Top) por Clase de Riesgo\n'
        '(Feature #2: Cohen\'s d=0.38, p<0.001)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_xticks(x)
    ax.set_xticklabels(['No (0)', 'Sí (1)'])
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "03_barras_contexto_equipo_top.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")


# =====================================================
# GRÁFICO 4: HISTOGRAMA SUPERPUESTO minutesPlayed
# =====================================================

def grafico_4_histograma_minutos(df):
    """
    Histogramas superpuestos de minutos jugados.
    Feature #10 (Cohen's d=0.27).
    """
    logger.info("📊 Generando Gráfico 4: Histograma minutesPlayed...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    bajo_riesgo = df[df['riesgo_fichaje'] == 0]['minutesPlayed']
    alto_riesgo = df[df['riesgo_fichaje'] == 1]['minutesPlayed']
    
    ax.hist(bajo_riesgo, bins=35, alpha=0.65, label='Bajo Riesgo', 
            color=COLOR_BAJO_RIESGO, edgecolor='black', linewidth=0.5)
    ax.hist(alto_riesgo, bins=35, alpha=0.65, label='Alto Riesgo', 
            color=COLOR_ALTO_RIESGO, edgecolor='black', linewidth=0.5)
    
    ax.axvline(bajo_riesgo.mean(), color=COLOR_BAJO_RIESGO, 
               linestyle='--', linewidth=2.5, 
               label=f'Media Bajo Riesgo: {bajo_riesgo.mean():.0f} min')
    ax.axvline(alto_riesgo.mean(), color=COLOR_ALTO_RIESGO, 
               linestyle='--', linewidth=2.5, 
               label=f'Media Alto Riesgo: {alto_riesgo.mean():.0f} min')
    
    ax.axvline(450, color='orange', linestyle=':', linewidth=2, 
               label='Umbral mínimo (450 min)', alpha=0.7)
    
    ax.set_xlabel('Minutos Jugados en la Temporada', fontsize=13, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribución de Minutos Jugados por Clase de Riesgo\n'
        '(Histogramas Superpuestos - Cohen\'s d=0.27, p<0.001)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.legend(fontsize=11, loc='upper right', framealpha=0.95)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "04_histograma_minutos.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")


# =====================================================
# GRÁFICO 5: BOXPLOT accurateCrosses_p90
# =====================================================

def grafico_5_boxplot_crosses(df):
    """
    Boxplot de centros acertados por 90 minutos.
    Feature #3 más discriminativa (Cohen's d=0.34).
    """
    logger.info("📊 Generando Gráfico 5: Boxplot accurateCrosses_p90...")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    data_to_plot = [
        df[df['riesgo_fichaje'] == 0]['accurateCrosses_p90'],
        df[df['riesgo_fichaje'] == 1]['accurateCrosses_p90']
    ]
    
    bp = ax.boxplot(data_to_plot, 
                    tick_labels=['Bajo Riesgo', 'Alto Riesgo'],
                    patch_artist=True,
                    widths=0.6,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='yellow', 
                                  markeredgecolor='black', markersize=8))
    
    for patch, color in zip(bp['boxes'], COLORES_CLASES):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_linewidth(2)
    
    for i, clase in enumerate([0, 1]):
        datos_clase = df[df['riesgo_fichaje'] == clase]['accurateCrosses_p90']
        media = datos_clase.mean()
        mediana = datos_clase.median()
        ax.text(i + 1, ax.get_ylim()[1] * 0.95, 
                f'Media: {media:.2f}\nMediana: {mediana:.2f}',
                ha='center', fontsize=10, bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    ax.set_ylabel('Centros Acertados por 90 Minutos', fontsize=13, fontweight='bold')
    ax.set_xlabel('Clase de Riesgo de Fichaje', fontsize=13, fontweight='bold')
    ax.set_title(
        'Distribución de Centros Acertados por 90 Min según Clase de Riesgo\n'
        '(Feature #3: Cohen\'s d=0.34, p<0.001)',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "05_boxplot_accurateCrosses_p90.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")


# =====================================================
# GRÁFICO 6: MATRIZ DE CORRELACIÓN
# =====================================================

def grafico_6_matriz_correlacion(df):
    """
    Matriz de correlación del target con las top features.
    """
    logger.info("📊 Generando Gráfico 6: Matriz de correlación...")
    
    exclude_cols = ['tm_id', 'ss_id']
    numeric_features = [
        col for col in df.select_dtypes(include=[np.number]).columns 
        if col not in exclude_cols and col != 'riesgo_fichaje'
    ]
    
    correlaciones = df[numeric_features + ['riesgo_fichaje']].corr()['riesgo_fichaje'].abs()
    correlaciones = correlaciones.sort_values(ascending=False)
    
    top_12_features = correlaciones[1:13].index.tolist()
    
    logger.info(f"   Top 12 features más correlacionadas con el target:")
    for i, feat in enumerate(top_12_features, 1):
        corr_val = correlaciones[feat]
        logger.info(f"      {i:2d}. {feat:40s}: {corr_val:.4f}")
    
    features_con_target = top_12_features + ['riesgo_fichaje']
    matriz_corr = df[features_con_target].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    sns.heatmap(matriz_corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8, "label": "Correlación"},
                vmin=-1, vmax=1, ax=ax, annot_kws={"size": 9})
    
    ax.set_title(
        'Matriz de Correlación: Top 12 Features vs Target\n'
        '(Características Más Predictivas del Riesgo de Fichaje)',
        fontsize=14, fontweight='bold', pad=20
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / "06_matriz_correlacion_target.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"✅ Gráfico guardado: {output_path}")


# =====================================================
# FUNCIÓN PRINCIPAL
# =====================================================

def main():
    """Ejecuta el análisis exploratorio completo."""
    logger.info("=" * 100)
    logger.info("🚀 INICIO - ANÁLISIS EXPLORATORIO DE DATOS (DATASET FINAL)")
    logger.info("=" * 100)
    
    try:
        df = cargar_datos()
        
        generar_reporte_estadistico(df)
        
        grafico_1_balance_clases(df)
        grafico_2_boxplot_fouls(df)
        grafico_3_barras_contexto(df)
        grafico_4_histograma_minutos(df)
        grafico_5_boxplot_crosses(df)
        grafico_6_matriz_correlacion(df)
        
        logger.info("=" * 100)
        logger.info("✅ ANÁLISIS EXPLORATORIO COMPLETADO EXITOSAMENTE")
        logger.info(f"📁 Todos los gráficos guardados en: {OUTPUT_DIR}")
        logger.info(f"📊 Total de gráficos generados: 6")
        logger.info("")
        logger.info("🎯 FEATURES GRAFICADAS (CON EVIDENCIA ESTADÍSTICA):")
        logger.info("   1. fouls_p90              (d=0.47, p<0.001) - MUY BUENA")
        logger.info("   2. contexto_equipo_top    (d=0.38, p<0.001) - BUENA")
        logger.info("   3. accurateCrosses_p90    (d=0.34, p<0.001) - BUENA")
        logger.info("   4. minutesPlayed          (d=0.27, p<0.001) - REGULAR")
        logger.info("=" * 100)
        
    except Exception as e:
        logger.error(f"❌ Error durante la ejecución: {e}")
        raise


if __name__ == "__main__":
    main()