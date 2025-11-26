# scripts/entrenar_modelos_comparativo.py
import warnings
warnings.filterwarnings("ignore")

import os
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
import copy

import numpy as np
import pandas as pd
                  
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, 
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


SEED = 42
np.random.seed(SEED)

logger = logging.getLogger("comparativo")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)
                                                          
# ✅ LIMPIO: Solo columnas auxiliares y target
EXCLUDE_COLS_BASE = {
    "tm_id", "ss_id", "nombre_jugador", "season",  
    "riesgo_fichaje"
}

def seleccionar_features(df: pd.DataFrame, target: str) -> list[str]:
    """
    Selecciona features para el modelo, excluyendo auxiliares y target.
    Incluye validaciones de seguridad.
    """
    excluir = set(EXCLUDE_COLS_BASE) | {target}
    feats = [c for c in df.columns if c not in excluir]
    
    logger.info(f"📋 Features seleccionadas: {len(feats)}")
    logger.info(f"   Excluidas: {sorted(excluir & set(df.columns))}")
    
    # ✅ Validación: target no debe estar en features
    if target in feats:
        raise ValueError(f"⚠️ ALERTA: '{target}' está en las features (TARGET LEAKAGE)")
    
    # ✅ Validación: dummies de posición
    if 'pos_Delantero' not in feats or 'pos_Mediocampista' not in feats:
        logger.warning("⚠️ Dummies de posición NO encontradas en features")
    else:
        logger.info("✅ Dummies de posición presentes: pos_Delantero, pos_Mediocampista")
    
    # ✅ Validación: número esperado de features
    if len(feats) != 31:
        logger.warning(f"⚠️ Se esperaban 31 features, se encontraron {len(feats)}")
    else:
        logger.info(f"✅ Número de features correcto: 31")
    
    return feats

def metricas_binarias(y_true, y_prob, thresh=0.5) -> dict:
    """Calcula métricas de clasificación binaria."""
    y_pred = (y_prob >= thresh).astype(int)
    
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan
        
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "AUC-ROC": auc,
    }

def construir_folds_temporales(temporadas_ordenadas: list[str]) -> list[tuple[list[str], str]]:
    """
    Construye folds temporales para walk-forward validation.
    Empieza desde i=2 para tener al menos 2 temporadas de entrenamiento.
    """
    folds = []
    for i in range(2, len(temporadas_ordenadas)): 
        train_seasons = temporadas_ordenadas[:i]
        test_season = temporadas_ordenadas[i]
        folds.append((train_seasons, test_season))
    return folds

def probas_modelo(clf, X_test):
    """Extrae probabilidades del modelo (maneja pipelines y modelos directos)."""
    try:
        # Si es pipeline, extraer el clasificador
        if isinstance(clf, Pipeline):
            clf_step = clf.named_steps.get('clf', clf)
        else:
            clf_step = clf
        
        # Intentar predict_proba
        if hasattr(clf_step, "predict_proba"):
            proba = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf_step, "decision_function"):
            df_val = clf.decision_function(X_test)
            proba = 1 / (1 + np.exp(-df_val))
        else:
            proba = clf.predict(X_test).astype(float)
    except Exception as e:
        logger.warning(f"⚠️ Error en predict_proba: {e}, usando predict")
        proba = clf.predict(X_test).astype(float)
    
    return np.clip(proba, 0, 1)
                                                                   
def crear_modelos() -> dict:
    modelos = {}
    
    # LOGISTIC REGRESSION (igual que antes)
    modelos["LogisticRegression"] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(
            solver="lbfgs",
            random_state=SEED,
            max_iter=1000,
            C=1.0,
            penalty='l2',
            class_weight='balanced'
        ))
    ])

    # 🔹 RANDOM FOREST – versión OPTIMIZADA (GridSearchCV)
    modelos["RandomForest"] = RandomForestClassifier(
        n_estimators=500,         # antes 400
        max_depth=12,            # antes 10
        min_samples_split=4,     # antes 2
        min_samples_leaf=1,      # antes 2
        max_features="log2",     # antes "sqrt"
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED
    )

    # XGBOOST (mantenemos baseline, porque el optimizado rindió peor en TEST)
    if HAS_XGB:
        modelos["XGBoost"] = XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=2.0,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=-1,
            random_state=SEED
        )

    # 🔹 LIGHTGBM – versión OPTIMIZADA (GridSearchCV)
    if HAS_LGBM:
        modelos["LightGBM"] = LGBMClassifier(
            n_estimators=300,          # antes 400
            learning_rate=0.02,        # antes 0.03
            max_depth=8,               # igual que antes
            num_leaves=25,             # igual que antes
            min_child_samples=20,      # igual que antes
            feature_fraction=0.7,      # igual
            bagging_fraction=0.8,      # igual
            bagging_freq=1,            # igual
            reg_alpha=0.0,             # antes 0.1
            reg_lambda=1.0,            # antes 2.0
            objective="binary",
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
            n_jobs=-1
        )
   
    # ADABOOST (igual)
    modelos["AdaBoost"] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', AdaBoostClassifier(
            n_estimators=200,
            learning_rate=0.1,
            random_state=SEED
        ))
    ])

    # GAUSSIAN NB (igual)
    modelos["GaussianNB"] = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GaussianNB())
    ])

    return modelos



COLORES_CM = {
    "LogisticRegression": "Purples",
    "RandomForest": "Greens",
    "XGBoost": "Oranges",
    "LightGBM": "Blues",
    "AdaBoost": "Reds",
    "GaussianNB": "YlOrBr"
}

def plot_per_fold_diagnostics(
    y_true: np.ndarray, y_proba: np.ndarray,
    title_prefix: str, outdir: Path
) -> None:
    """Genera matriz de confusión, curva ROC y Precision-Recall para un fold."""
    plt.close("all")
    y_pred = (y_proba >= 0.5).astype(int)

    modelo_nombre = title_prefix.split("_test_")[0]
    cmap_modelo = COLORES_CM.get(modelo_nombre, "Blues")

    # Confusion Matrix
    try:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, 
            display_labels=["Bajo Riesgo", "Alto Riesgo"]
        )
        fig, ax = plt.subplots(figsize=(7, 6))
        disp.plot(values_format="d", cmap=cmap_modelo, colorbar=False, ax=ax)
        ax.set_title(f"{modelo_nombre}\nMatriz de Confusión (Test={title_prefix.split('_')[-1]})", fontsize=13)
        plt.tight_layout()
        plt.savefig(outdir / f"{title_prefix}_CM.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"No se pudo generar CM para {title_prefix}: {e}")

    # ROC Curve
    try:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2.5, color='#2E86AB')
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=11)
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=11)
        ax.set_title(f"{modelo_nombre}\nCurva ROC (Test={title_prefix.split('_')[-1]})", fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"{title_prefix}_ROC.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"No se pudo generar ROC para {title_prefix}: {e}")

    # Precision-Recall Curve
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, linewidth=2.5, color='#A23B72', label="Curva P-R")
        ax.set_xlabel("Recall", fontsize=11)
        ax.set_ylabel("Precision", fontsize=11)
        ax.set_title(f"{modelo_nombre}\nCurva Precision-Recall (Test={title_prefix.split('_')[-1]})", fontsize=13)
        ax.legend(loc="best", fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"{title_prefix}_PR.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"No se pudo generar PR para {title_prefix}: {e}")


def plot_feature_importance(modelo_entrenado, feature_names: list, model_name: str, outdir: Path):
    """Genera gráfico de feature importance (solo para modelos con coef_ o feature_importances_)."""
    importances = None
    
    # Extraer clasificador si es pipeline
    clf = modelo_entrenado
    if isinstance(modelo_entrenado, Pipeline):
        clf = modelo_entrenado.named_steps.get('clf', modelo_entrenado)
    
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_[0])

    if importances is None:
        logger.info(f"   (Skipping Feature Importance: {model_name} no tiene importances)")
        return

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False).head(15)

    plt.close("all")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    color_map = {
        "LogisticRegression": "#9B59B6",
        "RandomForest": "#27AE60",
        "XGBoost": "#E67E22",
        "LightGBM": "#3498DB",
        "AdaBoost": "#E74C3C",
        "GaussianNB": "#F39C12"
    }
    color = color_map.get(model_name, "#34495E")
    
    sns.barplot(x="importance", y="feature", data=df_imp, orient="h", ax=ax, color=color)
    ax.set_title(f"Importancia de Features (Top 15)\n{model_name}", fontsize=14, pad=15)
    ax.set_xlabel("Importancia", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"{model_name}_FeatureImportance_Global.png", dpi=300)
    plt.close(fig)


def plot_summary_charts(df_summary: pd.DataFrame, df_last_season: pd.DataFrame, outdir: Path):
    """Genera gráficos de resumen comparativo."""
    logger.info("📊 Generando gráficos de resumen final...")
    plt.close("all")

    palette = {
        "LogisticRegression": "#9B59B6",
        "RandomForest": "#27AE60",
        "XGBoost": "#E67E22",
        "LightGBM": "#3498DB",
        "AdaBoost": "#E74C3C",
        "GaussianNB": "#F39C12"
    }

    # 1. Gráfico de F1-Score Promedio (CV)
    try:
        df_summary = df_summary.sort_values("F1-Score", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            x="Modelo", 
            y="F1-Score",  
            data=df_summary, 
            ax=ax, 
            hue="Modelo",
            palette=palette,
            legend=False
        )
        
        ax.set_title("Rendimiento Promedio (Walk-Forward CV, 5 Folds)", fontsize=14, pad=15)
        ax.set_ylabel("F1-Score (Promedio)", fontsize=12)
        ax.set_xlabel("Modelo", fontsize=12)
        ax.set_ylim(0, max(1.0, df_summary["F1-Score"].max() * 1.1))
        ax.grid(axis='y', alpha=0.3)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "01_comparativo_F1_promedio_CV.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        logger.error(f"No se pudo generar gráfico 01 (Promedio): {e}")
        
    # 2. Gráfico de F1-Score en Última Temporada (Test)
    try:
        df_last_season = df_last_season.sort_values("F1-Score", ascending=False)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.barplot(
            x="Modelo", 
            y="F1-Score",  
            data=df_last_season, 
            ax=ax,
            hue="Modelo",
            palette=palette,
            legend=False
        )
        
        test_season = df_last_season['Test_season'].iloc[0]
        ax.set_title(f"Rendimiento en Test Futuro (Test={test_season})", fontsize=14, pad=15)
        ax.set_ylabel("F1-Score", fontsize=12)
        ax.set_xlabel("Modelo", fontsize=12)
        ax.set_ylim(0, max(1.0, df_last_season["F1-Score"].max() * 1.1))
        ax.grid(axis='y', alpha=0.3)
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(outdir / "02_comparativo_F1_ultima_temporada_TEST.png", dpi=300)
        plt.close(fig)
    except Exception as e:
        logger.error(f"No se pudo generar gráfico 02 (Última Temporada): {e}")


def evaluar_modelo_walkforward(modelo_nombre: str, modelo, df: pd.DataFrame, feats: list[str],
                               target: str, temporadas_ordenadas: list[str], outdir: Path):
    """
    Evalúa un modelo usando walk-forward validation temporal.
    """
    results = []
    folds = construir_folds_temporales(temporadas_ordenadas)
    logger.info(f"🗓️ Folds temporales: {len(folds)} folds generados")
    
    ultima_temporada_test = temporadas_ordenadas[-1]

    for train_seasons, test_season in folds:
        df_train = df[df["season"].isin(train_seasons)]
        df_test = df[df["season"] == test_season]

        X_train = df_train[feats].copy()
        y_train = df_train[target].astype(int).values

        X_test = df_test[feats].copy()
        y_test = df_test[target].astype(int).values

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            logger.warning(
                f"⏭️ {modelo_nombre} | Train≤{train_seasons[-1]} → Test={test_season} "
                f"omitido por clases insuficientes"
            )
            continue

        modelo_fold = copy.deepcopy(modelo)
        
        # ✅ CORREGIDO: Manejo de scale_pos_weight para XGBoost (con/sin pipeline)
        if modelo_nombre == "XGBoost":
            neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
            spw = (neg / pos) if pos > 0 else 1.0
            
            if isinstance(modelo_fold, Pipeline):
                modelo_fold.named_steps['clf'].set_params(scale_pos_weight=spw)
            else:
                modelo_fold.set_params(scale_pos_weight=spw)
        
        modelo_fold.fit(X_train, y_train)
        y_prob = probas_modelo(modelo_fold, X_test)

        mets = metricas_binarias(y_test, y_prob, thresh=0.5)
        mets.update({
            "Modelo": modelo_nombre,
            "Train_seasons": "+".join(train_seasons),
            "Test_season": test_season,
            "n_train": len(df_train),
            "n_test": len(df_test)
        })
        results.append(mets)
        logger.info(
            f"✅ {modelo_nombre} | Train≤{train_seasons[-1]} → Test={test_season} | "
            f"F1={mets['F1-Score']:.3f} Acc={mets['Accuracy']:.3f} AUC={mets['AUC-ROC']:.3f}"
        )
        
        # Generar gráficos solo para el fold final
        if test_season == ultima_temporada_test:
            logger.info(f"   -> Generando gráficos para el fold final (Test={test_season})...")
            title_prefix = f"{modelo_nombre}_test_{test_season}"
            plot_per_fold_diagnostics(y_test, y_prob, title_prefix, outdir)

    df_res = pd.DataFrame(results)
    out_model_csv = outdir / f"folds_{modelo_nombre}.csv"
    df_res.to_csv(out_model_csv, index=False, encoding="utf-8-sig", sep=";")
    
    # Feature Importance (entrenar con todos los datos excepto última temporada)
    logger.info(f"   Generando Feature Importance para {modelo_nombre}...")
    X_train_full = df[df["season"].isin(temporadas_ordenadas[:-1])][feats].copy()
    y_train_full = df[df["season"].isin(temporadas_ordenadas[:-1])][target].astype(int).values
    
    logger.info(f"   Entrenando en {len(X_train_full)} registros ({len(temporadas_ordenadas)-1} temporadas)") 

    modelo_final_imp = copy.deepcopy(modelo)
    
    if modelo_nombre == "XGBoost":
        neg, pos = (y_train_full == 0).sum(), (y_train_full == 1).sum()
        spw = (neg / pos) if pos > 0 else 1.0
        
        if isinstance(modelo_final_imp, Pipeline):
            modelo_final_imp.named_steps['clf'].set_params(scale_pos_weight=spw)
        else:
            modelo_final_imp.set_params(scale_pos_weight=spw)
        
    modelo_final_imp.fit(X_train_full, y_train_full)
    plot_feature_importance(modelo_final_imp, feats, modelo_nombre, outdir)

    return results, df_res

def resumir_resultados_por_modelo(resultados_por_modelo: dict, outdir: Path):
    """Genera resumen comparativo de todos los modelos."""
    resumen_rows = []
    ult_rows = []

    todas = pd.concat([df for _, df in resultados_por_modelo.values()], ignore_index=True)
    if todas.empty:
        logger.error("No se generaron resultados para resumir.")
        return
        
    temporadas = sorted(todas["Test_season"].unique())
    if not temporadas:
        logger.error("No hay temporadas de test en los resultados.")
        return
        
    ultima_temporada = temporadas[-1]

    for modelo, df_res in resultados_por_modelo.values():
        if df_res.empty:
            continue
        
        row_mean = {
            "Modelo": modelo,
            "Accuracy": df_res["Accuracy"].mean(),
            "Precision": df_res["Precision"].mean(),
            "Recall": df_res["Recall"].mean(),
            "F1-Score": df_res["F1-Score"].mean(),
            "AUC-ROC": df_res["AUC-ROC"].mean(),
            "F1-Score_std": df_res["F1-Score"].std(ddof=1),
            "AUC-ROC_std": df_res["AUC-ROC"].std(ddof=1),
        }
        resumen_rows.append(row_mean)

        ult = df_res[df_res["Test_season"] == ultima_temporada].copy()
        if not ult.empty:
            u = ult.iloc[0]
            ult_rows.append({
                "Modelo": modelo,
                "Test_season": u["Test_season"],
                "Accuracy": u["Accuracy"],
                "Precision": u["Precision"],
                "Recall": u["Recall"],
                "F1-Score": u["F1-Score"],
                "AUC-ROC": u["AUC-ROC"],
                "n_train": u["n_train"],
                "n_test": u["n_test"]
            })

    df_resumen = pd.DataFrame(resumen_rows).sort_values("F1-Score", ascending=False)
    df_ult = pd.DataFrame(ult_rows).sort_values("F1-Score", ascending=False)

    out_csv = outdir / "resumen_comparativo.csv"
    df_resumen.to_csv(out_csv, index=False, encoding="utf-8-sig", sep=";")

    out_ult = outdir / "resumen_ult_temporada.csv"
    df_ult.to_csv(out_ult, index=False, encoding="utf-8-sig", sep=";")
    
    try:
        plot_summary_charts(df_resumen, df_ult, outdir)
    except Exception as e:
        logger.error(f"No se pudieron generar gráficos de resumen: {e}")

    logger.info("\n" + "="*100)
    logger.info("📊 RESUMEN COMPARATIVO — Walk-Forward CV (Promedio de 5 Folds)")
    logger.info("="*100)
    
    cols_mostrar = ["Modelo", "Accuracy", "Precision", "Recall", "F1-Score", "AUC-ROC"]
    logger.info(df_resumen[cols_mostrar].to_string(index=False))
    logger.info(f"\n💾 Reporte completo guardado en: {out_csv}")

    logger.info("\n" + "="*100)
    logger.info(f"🎯 RENDIMIENTO EN LA ÚLTIMA TEMPORADA (Test={ultima_temporada})")
    logger.info("="*100)
    logger.info(df_ult[cols_mostrar].to_string(index=False))
    logger.info(f"\n💾 Reporte guardado en: {out_ult}")
    logger.info("="*100 + "\n")

    return df_resumen, df_ult

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=str,
        default="datos_salida/finales/dataset_entrenamiento_final.csv",
        help="Ruta al CSV de entrenamiento final"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="riesgo_fichaje",
        help="Columna objetivo (default: riesgo_fichaje)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="reports/final_model_comparison",
        help="Directorio de salida de reportes"
    )
    
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, sep=";", encoding="utf-8-sig")
    logger.info(f"📂 Dataset: {len(df)} filas, {df.shape[1]} columnas | {Path(args.csv).name}")

    if args.target not in df.columns:
        raise ValueError(f"Target '{args.target}' no existe en el dataset.")

    feats = seleccionar_features(df, args.target)
    logger.info(f"🧮 Nº de features para el modelo: {len(feats)}")
                               
    temporadas = sorted(
        df["season"].unique().tolist(),
        key=lambda s: tuple(map(int, s.split("_")))
    )
    
    if len(temporadas) < 3:
        raise ValueError("Se requieren ≥3 temporadas para walk-forward.")
    logger.info(f"🗓️ Temporadas detectadas (orden): {temporadas}")

    modelos = crear_modelos()

    resultados_por_modelo = {}
    for nombre, modelo in modelos.items():
        logger.info(f"\n▶️ Modelo: {nombre}")
        
        model_plot_dir = outdir / nombre
        model_plot_dir.mkdir(parents=True, exist_ok=True)
        
        res_list, df_res = evaluar_modelo_walkforward(
            nombre, modelo, df, feats, args.target, temporadas, model_plot_dir
        )
        resultados_por_modelo[nombre] = (nombre, df_res)

    resumir_resultados_por_modelo(resultados_por_modelo, outdir)


if __name__ == "__main__":
    main()