# scripts/exportar_modelo.py

import warnings
warnings.filterwarnings("ignore")

import json
from pathlib import Path
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
SEED = 42
np.random.seed(SEED)

MODELO_NOMBRE = "RandomForest"
CSV_PATH = "datos_salida/finales/dataset_entrenamiento_final.csv"
TARGET_COL = "riesgo_fichaje"
OUTDIR_PATH = "model_artifacts/produccion_randomforest"

# ✅ Hiperparámetros OPTIMIZADOS de RandomForest
HYPERPARAMS_RF = {
    "n_estimators": 500,
    "max_depth": 12,
    "min_samples_split": 4,
    "min_samples_leaf": 1,
    "max_features": "log2",
    "class_weight": "balanced",
    "random_state": SEED,
    "n_jobs": -1,
    "verbose": 0,
}

logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ✅ LIMPIO: Sin score_ponderado
EXCLUDE_COLS_BASE = {
    "tm_id",
    "ss_id",
    "nombre_jugador",
    "season",
    "riesgo_fichaje",
}


# =========================
# SELECCIÓN DE FEATURES
# =========================
def seleccionar_features(df: pd.DataFrame, target: str) -> list[str]:
    """
    Selecciona features para el modelo (consistente con entrenar_modelos_comparativo.py).
    """
    excluir = set(EXCLUDE_COLS_BASE) | {target}
    feats = [c for c in df.columns if c not in excluir]

    logger.info(f"📋 Features seleccionadas: {len(feats)}")
    logger.info(f"   Excluidas: {sorted(excluir & set(df.columns))}")

    # Validaciones
    if target in feats:
        raise ValueError(f"⚠️ ALERTA: '{target}' está en las features (TARGET LEAKAGE)")

    if "pos_Delantero" not in feats or "pos_Mediocampista" not in feats:
        logger.warning("⚠️ Dummies de posición NO encontradas")
    else:
        logger.info("✅ Dummies de posición presentes")

    if len(feats) != 31:
        raise ValueError(f"⚠️ ERROR: Se esperaban 31 features, se encontraron {len(feats)}")
    
    logger.info("✅ Número de features correcto: 31")
    
    return feats


def ordenar_temporadas(vals):
    """Ordena temporadas cronológicamente."""
    return sorted(vals, key=lambda s: tuple(map(int, s.split("_"))))


# =========================
# GRÁFICOS DE DIAGNÓSTICO
# =========================
def plot_confusion_matrix(y_true, y_pred, outdir: Path):
    """Genera matriz de confusión."""
    logger.info("📊 Generando Matriz de Confusión...")
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Bajo Riesgo", "Alto Riesgo"]
    )
    
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(values_format="d", cmap="Greens", colorbar=False, ax=ax)
    ax.set_title(f"{MODELO_NOMBRE}\nMatriz de Confusión (Test Hold-out)", 
                 fontsize=14, 
                 fontweight='bold',
                 pad=15)
    plt.tight_layout()
    
    cm_path = outdir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Matriz de Confusión guardada: {cm_path}")
    
    return {
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }


def plot_roc_curve(y_true, y_prob, outdir: Path):
    """Genera curva ROC."""
    logger.info("📈 Generando Curva ROC...")
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=3, color='#27AE60')
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=2, alpha=0.7)
    
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=12, fontweight='bold')
    ax.set_title(f"{MODELO_NOMBRE}\nCurva ROC (Test Hold-out)", 
                 fontsize=14, 
                 fontweight='bold',
                 pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    roc_path = outdir / "roc_curve.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Curva ROC guardada: {roc_path}")
    
    return {"auc_roc": float(auc)}


def plot_precision_recall_curve(y_true, y_prob, outdir: Path):
    """Genera curva Precision-Recall."""
    logger.info("📉 Generando Curva Precision-Recall...")
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, linewidth=3, color='#27AE60', label="Curva P-R")
    
    ax.set_xlabel("Recall", fontsize=12, fontweight='bold')
    ax.set_ylabel("Precision", fontsize=12, fontweight='bold')
    ax.set_title(f"{MODELO_NOMBRE}\nCurva Precision-Recall (Test Hold-out)", 
                 fontsize=14, 
                 fontweight='bold',
                 pad=15)
    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    
    pr_path = outdir / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Curva Precision-Recall guardada: {pr_path}")


def plot_feature_importance(modelo, feature_names: list, outdir: Path):
    """Genera gráfico de feature importance."""
    logger.info("📊 Generando Feature Importance...")
    
    importances = modelo.feature_importances_
    
    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=df_imp, orient="h", ax=ax, color='#27AE60')
    
    ax.set_title(f"Importancia de Features (Top 20)\n{MODELO_NOMBRE}", 
                 fontsize=14, 
                 fontweight='bold',
                 pad=15)
    ax.set_xlabel("Importancia", fontsize=12, fontweight='bold')
    ax.set_ylabel("Feature", fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    fi_path = outdir / "feature_importance.png"
    plt.savefig(fi_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Feature Importance guardado: {fi_path}")
    
    # Guardar importancias en JSON
    importance_dict = df_imp.set_index('feature')['importance'].to_dict()
    return {k: float(v) for k, v in importance_dict.items()}


def plot_convergence_curve(X_train, y_train, X_test, y_test, outdir: Path):
    """
    Genera curva de convergencia (F1 vs Número de Árboles).
    Equivalente a las learning curves de modelos boosting.
    """
    logger.info("📊 Generando Curva de Convergencia (F1 vs Árboles)...")
    
    rf = RandomForestClassifier(
        max_depth=HYPERPARAMS_RF['max_depth'],
        min_samples_split=HYPERPARAMS_RF['min_samples_split'],
        min_samples_leaf=HYPERPARAMS_RF['min_samples_leaf'],
        max_features=HYPERPARAMS_RF['max_features'],
        class_weight=HYPERPARAMS_RF['class_weight'],
        warm_start=True,
        random_state=SEED,
        n_jobs=-1,
        verbose=0
    )
    
    n_trees_range = list(range(50, HYPERPARAMS_RF['n_estimators'] + 1, 25))
    train_scores = []
    test_scores = []
    
    logger.info(f"   ⏳ Evaluando {len(n_trees_range)} configuraciones...")
    
    for n in n_trees_range:
        rf.n_estimators = n
        rf.fit(X_train, y_train)
        
        train_pred = rf.predict(X_train)
        test_pred = rf.predict(X_test)
        
        train_f1 = f1_score(y_train, train_pred, average='macro')
        test_f1 = f1_score(y_test, test_pred, average='macro')
        
        train_scores.append(train_f1)
        test_scores.append(test_f1)
    
    # Gráfico
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.plot(n_trees_range, train_scores, 
            label='F1-Score Entrenamiento', 
            linewidth=2.5, 
            color='#2E86AB',
            marker='o',
            markersize=4,
            alpha=0.8)
    
    ax.plot(n_trees_range, test_scores, 
            label='F1-Score Test (Hold-out)', 
            linewidth=2.5, 
            color='#27AE60',
            marker='s',
            markersize=4,
            alpha=0.8)
    
    # Marcar el punto óptimo
    best_idx = n_trees_range.index(HYPERPARAMS_RF['n_estimators'])
    ax.axvline(x=HYPERPARAMS_RF['n_estimators'], 
               color='gray', 
               linestyle='--', 
               alpha=0.5, 
               linewidth=1.5)
    ax.text(HYPERPARAMS_RF['n_estimators'], 
            max(test_scores) * 0.95, 
            f'Óptimo\n(n={HYPERPARAMS_RF["n_estimators"]})', 
            ha='center', 
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Número de Árboles', fontsize=13, fontweight='bold')
    ax.set_ylabel('F1-Score (Macro)', fontsize=13, fontweight='bold')
    ax.set_title('Convergencia: F1-Score vs Número de Árboles', 
                 fontsize=14, 
                 fontweight='bold',
                 pad=15)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)
    
    plt.tight_layout()
    conv_path = outdir / "convergence_curve.png"
    plt.savefig(conv_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"   ✅ Curva de Convergencia guardada: {conv_path}")
    logger.info(f"   📊 F1 final (n={HYPERPARAMS_RF['n_estimators']}): "
                f"Train={train_scores[-1]:.4f}, Test={test_scores[-1]:.4f}")
    
    return {
        "convergence_train_f1_final": float(train_scores[-1]),
        "convergence_test_f1_final": float(test_scores[-1]),
        "convergence_optimal_trees": int(HYPERPARAMS_RF['n_estimators']),
    }


# =========================
# ENTRENAMIENTO Y EXPORTACIÓN
# =========================
def entrenar_y_exportar():
    """Función principal de entrenamiento y exportación."""
    
    outdir = Path(OUTDIR_PATH)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("🚀 INICIANDO EXPORTACIÓN DE MODELO RANDOMFOREST")
    logger.info("=" * 80)

    # 1. Cargar dataset
    df = pd.read_csv(CSV_PATH, sep=";", encoding="utf-8-sig")
    logger.info(f"📂 Dataset cargado: {len(df)} filas, {df.shape[1]} columnas")
    logger.info(f"   Ruta: {CSV_PATH}")

    # 2. Validar target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target '{TARGET_COL}' no encontrado en el dataset")
    
    if len(df[TARGET_COL].unique()) < 2:
        raise ValueError(f"Target '{TARGET_COL}' debe tener al menos 2 clases")

    # 3. Seleccionar features
    feats = seleccionar_features(df, TARGET_COL)

    # 4. Definir split temporal
    temporadas = ordenar_temporadas(df["season"].unique().tolist())
    
    if len(temporadas) < 2:
        raise ValueError(
            f"Se requieren ≥2 temporadas; encontradas: {temporadas}"
        )

    test_season = temporadas[-1]
    train_seasons = temporadas[:-1]

    logger.info(f"🗓️ Temporadas disponibles: {temporadas}")
    logger.info(f"🛠️ Train: {train_seasons}")
    logger.info(f"🎯 Test (Hold-out): {test_season}")

    # 5. Preparar datos
    df_train = df[df["season"].isin(train_seasons)].copy()
    df_test = df[df["season"] == test_season].copy()

    X_train = df_train[feats]
    y_train = df_train[TARGET_COL].astype(int)
    X_test = df_test[feats]
    y_test = df_test[TARGET_COL].astype(int)

    logger.info(f"📊 Train: {len(y_train)} registros")
    logger.info(f"📊 Test:  {len(y_test)} registros")
    logger.info(f"📊 Distribución Train: Clase 0={int(sum(y_train == 0))}, "
                f"Clase 1={int(sum(y_train == 1))}")
    logger.info(f"📊 Distribución Test:  Clase 0={int(sum(y_test == 0))}, "
                f"Clase 1={int(sum(y_test == 1))}")

    # 6. Entrenar modelo
    logger.info("\n" + "=" * 80)
    logger.info("🧬 ENTRENANDO MODELO FINAL PARA EXPORTACIÓN")
    logger.info("=" * 80)

    modelo_a_exportar = RandomForestClassifier(**HYPERPARAMS_RF)
    
    logger.info("🔧 Hiperparámetros:")
    for k, v in HYPERPARAMS_RF.items():
        if k not in ['random_state', 'n_jobs', 'verbose']:
            logger.info(f"   {k}: {v}")

    modelo_a_exportar.fit(X_train, y_train)
    logger.info("✅ Entrenamiento completado")

    # 7. Evaluar en test
    logger.info("\n" + "=" * 80)
    logger.info(f"📊 EVALUACIÓN EN TEST (HOLD-OUT) - Temporada {test_season}")
    logger.info("=" * 80)

    y_prob_test = modelo_a_exportar.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    metrics_test = {
        "Accuracy": accuracy_score(y_test, y_pred_test),
        "Precision": precision_score(y_test, y_pred_test, average="macro"),
        "Recall": recall_score(y_test, y_pred_test, average="macro"),
        "F1-Score": f1_score(y_test, y_pred_test, average="macro"),
        "AUC-ROC": roc_auc_score(y_test, y_prob_test),
    }

    logger.info("📊 Métricas en TEST (HOLD-OUT):")
    for k, v in metrics_test.items():
        logger.info(f"   {k}: {v:.4f}")

    # 8. Generar gráficos de diagnóstico
    logger.info("\n" + "=" * 80)
    logger.info("📊 GENERANDO GRÁFICOS DE DIAGNÓSTICO")
    logger.info("=" * 80)

    cm_info = plot_confusion_matrix(y_test, y_pred_test, outdir)
    roc_info = plot_roc_curve(y_test, y_prob_test, outdir)
    plot_precision_recall_curve(y_test, y_prob_test, outdir)
    fi_info = plot_feature_importance(modelo_a_exportar, feats, outdir)
    conv_info = plot_convergence_curve(X_train, y_train, X_test, y_test, outdir)

    # 9. Exportar artefactos
    logger.info("\n" + "=" * 80)
    logger.info("💾 EXPORTANDO ARTEFACTOS")
    logger.info("=" * 80)

    # Modelo
    model_path = outdir / f"{MODELO_NOMBRE.lower()}_model.joblib"
    joblib.dump(modelo_a_exportar, model_path)
    logger.info(f"   ✅ Modelo guardado: {model_path}")

    # Metadatos
    meta = {
        "model_name": MODELO_NOMBRE,
        "version": "1.0",
        "timestamp": pd.Timestamp.now().isoformat(),
        "source_dataset": CSV_PATH,
        "training_seasons": train_seasons,
        "test_season": test_season,
        "hyperparameters": HYPERPARAMS_RF,
        "target_column": TARGET_COL,
        "n_features": len(feats),
        "features_list": feats,
        "training_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "test_metrics": {k: float(v) for k, v in metrics_test.items()},
        "confusion_matrix": cm_info,
        "roc_auc": roc_info,
        "feature_importance_top20": fi_info,
        "convergence_info": conv_info,
        "decision_threshold": 0.5,
    }

    metadata_path = outdir / "metadata.json"
    with open(metadata_path, "w", encoding='utf-8') as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)
    logger.info(f"   ✅ Metadatos guardados: {metadata_path}")

    # Features
    features_path = outdir / "features.txt"
    with open(features_path, "w", encoding='utf-8') as f:
        f.write("\n".join(feats))
    logger.info(f"   ✅ Features guardadas: {features_path}")

    # 10. Resumen final
    logger.info("\n" + "=" * 80)
    logger.info("🎉 EXPORTACIÓN COMPLETADA EXITOSAMENTE")
    logger.info("=" * 80)
    logger.info(f"\n📦 Artefactos generados en: {outdir}")
    logger.info(f"   1. Modelo entrenado: {model_path.name}")
    logger.info(f"   2. Metadatos completos: {metadata_path.name}")
    logger.info(f"   3. Lista de features: {features_path.name}")
    logger.info("   4. Matriz de Confusión: confusion_matrix.png")
    logger.info("   5. Curva ROC: roc_curve.png")
    logger.info("   6. Curva Precision-Recall: precision_recall_curve.png")
    logger.info("   7. Feature Importance: feature_importance.png")
    logger.info("   8. Curva de Convergencia: convergence_curve.png")
    
    logger.info(f"\n📊 Rendimiento en TEST (Temporada {test_season}):")
    for k, v in metrics_test.items():
        logger.info(f"   • {k}: {v:.4f}")
    
    logger.info("\n🎯 Umbral de decisión: 0.5000")
    logger.info("\n" + "=" * 80)
    logger.info("✅ Modelo RandomForest listo para producción")
    logger.info("=" * 80)


def main():
    """Punto de entrada principal."""
    try:
        entrenar_y_exportar()
    except Exception as e:
        logger.error(f"\n❌ ERROR DURANTE LA EXPORTACIÓN: {e}")
        logger.error("=" * 80)
        raise


if __name__ == "__main__":
    main()