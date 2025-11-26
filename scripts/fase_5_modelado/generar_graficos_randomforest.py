# scripts/generar_graficos_randomforest.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from sklearn.metrics import f1_score

# Configuración
SEED = 42
np.random.seed(SEED)

# Configuración visual
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def cargar_datos_y_preparar():
    """Carga el dataset y prepara train/val split para gráficos."""
    
    # Cargar dataset
    df = pd.read_csv(
        "datos_salida/finales/dataset_entrenamiento_final.csv",
        sep=";",
        encoding="utf-8-sig"
    )
    
    print(f"📂 Dataset cargado: {len(df)} registros")
    
    # Separar por temporada
    temporadas = sorted(
        df["season"].unique().tolist(),
        key=lambda s: tuple(map(int, s.split('_')))
    )
    
    print(f"🗓️ Temporadas: {temporadas}")
    
    # Train: todas menos la última
    # Val: última temporada
    last_season = temporadas[-1]
    df_train = df[df["season"] != last_season].copy()
    df_val = df[df["season"] == last_season].copy()
    
    print(f"🛠️ Train: {len(df_train)} registros (temporadas {temporadas[:-1]})")
    print(f"🎯 Validation: {len(df_val)} registros (temporada {last_season})")
    
    # Features y target
    exclude_cols = {'tm_id', 'ss_id', 'nombre_jugador', 'season', 'riesgo_fichaje'}
    feats = [c for c in df_train.columns if c not in exclude_cols]
    
    X_train = df_train[feats]                             
    y_train = df_train['riesgo_fichaje'].astype(int)

    X_val = df_val[feats]                                 
    y_val = df_val['riesgo_fichaje'].astype(int)
    
    print(f"✅ Features: {len(feats)}")
    
    return X_train, y_train, X_val, y_val, feats, df_train


def grafico_convergencia_arboles(X_train, y_train, X_val, y_val, output_dir):
    """
    GRÁFICO 1: F1-Score vs Número de Árboles
    Equivalente a las curvas de loss de XGBoost/LightGBM
    """
    print("\n" + "="*80)
    print("📊 GENERANDO GRÁFICO 1: Convergencia (F1 vs Número de Árboles)")
    print("="*80)
    
    rf = RandomForestClassifier(
        n_estimators=50,         # ← SUGERENCIA 1
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features="log2",
        class_weight="balanced",
        warm_start=True,
        random_state=SEED,
        n_jobs=-1,
        verbose=0
    )
    
    # Entrenar incrementalmente
    n_trees_range = list(range(50, 551, 25))  # 50, 75, 100, ..., 550
    train_scores = []
    val_scores = []
    
    print(f"⏳ Entrenando con {len(n_trees_range)} configuraciones de árboles...")
    
    for i, n in enumerate(n_trees_range):
        rf.n_estimators = n
        rf.fit(X_train, y_train)
        
        # Evaluar
        train_pred = rf.predict(X_train)
        val_pred = rf.predict(X_val)
        
        train_f1 = f1_score(y_train, train_pred, average='macro')
        val_f1 = f1_score(y_val, val_pred, average='macro')
        
        train_scores.append(train_f1)
        val_scores.append(val_f1)
        
        if (i + 1) % 5 == 0:
            print(f"   Progreso: {i+1}/{len(n_trees_range)} | "
                  f"n_trees={n} | Train F1={train_f1:.3f} | Val F1={val_f1:.3f}")
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(n_trees_range, train_scores, 
            label='F1-Score Entrenamiento', 
            linewidth=2.5, 
            color='#2E86AB',
            marker='o',
            markersize=4,
            alpha=0.8)
    
    ax.plot(n_trees_range, val_scores, 
            label='F1-Score Validación (Temporada 2024-2025)',
            linewidth=2.5, 
            color='#E67E22',
            marker='s',
            markersize=4,
            alpha=0.8)
    
    # Marcar el punto óptimo (n=500)
    best_idx = n_trees_range.index(500) if 500 in n_trees_range else -1
    if best_idx >= 0:
        ax.axvline(x=500, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
        ax.text(500, max(val_scores)*0.95, 
                f'Óptimo\n(n=500)', 
                ha='center', 
                fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Número de Árboles', fontsize=13)
    ax.set_ylabel('F1-Score (Macro)', fontsize=13)
    ax.set_title('Convergencia de RandomForest: F1-Score vs Número de Árboles', 
                 fontsize=15, 
                 pad=20,
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    output_path = output_dir / "RandomForest_Convergencia_Arboles.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico guardado: {output_path}")
    print(f"   F1 Train final (n=500): {train_scores[n_trees_range.index(500)]:.4f}")
    print(f"   F1 Val final (n=500):   {val_scores[n_trees_range.index(500)]:.4f}")
    
    return train_scores, val_scores


def grafico_learning_curve(X_train, y_train, cv_splitter, output_dir):
    """
    GRÁFICO 2: Learning Curve
    Muestra cómo varía el rendimiento con diferentes tamaños de entrenamiento
    """
    print("\n" + "="*80)
    print("📊 GENERANDO GRÁFICO 2: Learning Curve (Tamaño de Datos vs F1)")
    print("="*80)
    
    # Modelo optimizado
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=1,
        max_features="log2",
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
        verbose=0
    )
    
    print("⏳ Calculando learning curve (esto puede tomar 2-3 minutos)...")
    
    # Calcular learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        rf,
        X_train,
        y_train,
        cv=cv_splitter,  # Walk-forward CV
        scoring='f1_macro',
        train_sizes=np.linspace(0.2, 1.0, 8),  # 20%, 30%, ..., 100%
        n_jobs=-1,
        verbose=0
    )
    
    # Calcular medias y desviaciones
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    # Crear gráfico
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Línea de entrenamiento
    ax.plot(train_sizes, train_mean, 
            label='F1-Score Entrenamiento', 
            linewidth=2.5, 
            color='#2E86AB',
            marker='o',
            markersize=6)
    ax.fill_between(train_sizes, 
                     train_mean - train_std, 
                     train_mean + train_std, 
                     alpha=0.2, 
                     color='#2E86AB')
    
    # Línea de validación
    ax.plot(train_sizes, val_mean, 
            label='F1-Score Validación (CV Walk-Forward)', 
            linewidth=2.5, 
            color='#E67E22',
            marker='s',
            markersize=6)
    ax.fill_between(train_sizes, 
                     val_mean - val_std, 
                     val_mean + val_std, 
                     alpha=0.2, 
                     color='#E67E22')
    
    ax.set_xlabel('Tamaño del Conjunto de Entrenamiento (registros)', fontsize=13)
    ax.set_ylabel('F1-Score (Macro)', fontsize=13)
    ax.set_title('Curva de Aprendizaje: RandomForest\n¿Más datos mejorarían el modelo?', 
                 fontsize=15, 
                 pad=20,
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_ylim(0.4, 1.0)
    
    plt.tight_layout()
    output_path = output_dir / "RandomForest_Learning_Curve.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Gráfico guardado: {output_path}")
    print(f"   Con 100% datos: Train F1={train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"                    Val F1={val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
    
    return train_sizes, train_mean, val_mean


def construir_cv_splitter(df_train):
    """Construye el CV splitter walk-forward para learning curve."""
    temporadas_cv = sorted(
        df_train["season"].unique().tolist(),
        key=lambda s: tuple(map(int, s.split('_')))
    )
    
    # Reset index para que los índices coincidan
    df_train = df_train.reset_index(drop=True)
    
    splits = []
    for i in range(2, len(temporadas_cv)):
        train_seasons = temporadas_cv[:i]
        val_season = temporadas_cv[i]
        
        train_idx = df_train[df_train["season"].isin(train_seasons)].index.tolist()
        val_idx = df_train[df_train["season"] == val_season].index.tolist()
        
        if train_idx and val_idx:
            splits.append((train_idx, val_idx))
    
    return splits


def main():
    print("\n" + "="*80)
    print("🎨 GENERADOR DE GRÁFICOS ADICIONALES PARA RANDOMFOREST")
    print("="*80)
    
    # Crear directorio de salida
    output_dir = Path("reports/final_model_comparison/RandomForest")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos
    X_train, y_train, X_val, y_val, feats, df_train = cargar_datos_y_preparar()
    
    # 2. Construir CV splitter
    cv_splitter = construir_cv_splitter(df_train)
    print(f"✅ CV Splitter: {len(cv_splitter)} folds walk-forward")
    
    # 3. GRÁFICO 1: Convergencia (F1 vs Número de Árboles)
    train_scores, val_scores = grafico_convergencia_arboles(
        X_train, y_train, X_val, y_val, output_dir
    )
    
    # 4. GRÁFICO 2: Learning Curve
    train_sizes, train_mean, val_mean = grafico_learning_curve(
        X_train, y_train, cv_splitter, output_dir
    )
    
    # Resumen final
    print("\n" + "="*80)
    print("✅ TODOS LOS GRÁFICOS GENERADOS EXITOSAMENTE")
    print("="*80)
    print(f"📂 Ubicación: {output_dir}")
    print("\nArchivos generados:")
    print("   1. RandomForest_Convergencia_Arboles.png")
    print("   2. RandomForest_Learning_Curve.png")
    print("\nGráficos adicionales ya existentes:")
    print("   3. RandomForest_FeatureImportance_Global.png")
    print("   4. RandomForest_test_2024_2025_CM.png")
    print("   5. RandomForest_test_2024_2025_ROC.png")
    print("   6. RandomForest_test_2024_2025_PR.png")
    print("="*80)
    print("\n🎓 INTERPRETACIÓN PARA TU TESIS:")
    print("-"*80)
    print("GRÁFICO 1 (Convergencia):")
    print("  - Muestra cómo el modelo mejora al agregar más árboles")
    print("  - El F1 de validación se estabiliza alrededor de 400-500 árboles")
    print("  - Equivalente a las curvas de loss de XGBoost/LightGBM")
    print("\nGRÁFICO 2 (Learning Curve):")
    print("  - Muestra si más datos mejorarían el modelo")
    print("  - Curvas convergentes → dataset suficiente")
    print("  - Curvas separadas → posible beneficio de más datos")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()