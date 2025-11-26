# scripts/optimizar_modelo.py
import warnings
warnings.filterwarnings("ignore")

import json
import argparse
import logging
from time import time

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# =========================
# CONFIGURACIÓN GLOBAL
# =========================
SEED = 42
np.random.seed(SEED)

logger = logging.getLogger("optimizar_modelo")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(ch)

EXCLUDE_COLS_BASE = {
    "tm_id", "ss_id", "nombre_jugador", "season", "riesgo_fichaje"
}

# =========================
# SELECCIÓN DE FEATURES
# =========================
def seleccionar_features(df: pd.DataFrame, target: str) -> list[str]:
    """Selecciona features (consistente con entrenar_modelos_comparativo.py)."""
    excluir = set(EXCLUDE_COLS_BASE) | {target}
    feats = [c for c in df.columns if c not in excluir]
    
    logger.info(f"📋 Features seleccionadas: {len(feats)}")
    
    # Validaciones
    if target in feats:
        raise ValueError(f"⚠️ ALERTA: '{target}' en features (TARGET LEAKAGE)")
    
    if 'pos_Delantero' not in feats or 'pos_Mediocampista' not in feats:
        logger.warning("⚠️ Dummies de posición NO encontradas")
    else:
        logger.info("✅ Dummies de posición presentes")
    
    # ✅ Validación de número de features
    if len(feats) != 31:
        logger.warning(f"⚠️ Se esperaban 31 features, encontradas {len(feats)}")
    else:
        logger.info("✅ Número de features correcto: 31")
    
    return feats

# =========================
# CV WALK-FORWARD TEMPORAL
# =========================
def construir_folds_temporales(temporadas_ordenadas: list[str]) -> list[tuple[list[str], str]]:
    """Genera folds walk-forward."""
    folds = []
    for i in range(2, len(temporadas_ordenadas)): 
        train_seasons = temporadas_ordenadas[:i]
        val_season = temporadas_ordenadas[i]
        folds.append((train_seasons, val_season))
    return folds

def crear_splits_indices_manual(df_train: pd.DataFrame, temporadas_folds: list) -> list:
    """
    Crea splits manuales de índices para GridSearchCV.
    
    ✅ CRÍTICO: df_train debe venir YA con índices reseteados desde main()
    """
    logger.info("🛡️ Creando splits Walk-Forward...")
    splits_indices = []
    
    # ✅ NO hacemos reset_index aquí (ya viene reseteado de main)
    
    for train_seasons, val_season in temporadas_folds:
        train_indices = df_train[df_train["season"].isin(train_seasons)].index.tolist()
        val_indices = df_train[df_train["season"] == val_season].index.tolist()

        if not train_indices or not val_indices:
            logger.warning("Fold omitido: datos insuficientes")
            continue
            
        logger.info(
            f"   Fold: TRAIN {train_seasons} ({len(train_indices)}) | "
            f"VAL [{val_season}] ({len(val_indices)})"
        )
        splits_indices.append((train_indices, val_indices))
    
    return splits_indices

# =========================
# MODELOS Y PARRILLAS REALISTAS
# =========================
def get_modelo_y_params(modelo_nombre: str, y_train=None):
    """
    Retorna modelo base y parrilla de hiperparámetros para GridSearchCV.
    Parrillas PEQUEÑAS pero centradas en rangos prometedores.
    """
    
    if modelo_nombre.lower() == 'lightgbm':
        modelo = LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
            n_jobs=-1
        )
        # ✅ Parrilla PEQUEÑA centrada en tu baseline (400, 0.03, etc.)
        param_grid = {
            "n_estimators": [300, 400, 500],           # 3 valores
            "learning_rate": [0.02, 0.03, 0.05],       # 3 valores
            "max_depth": [6, 8, 10],                   # 3 valores
            "num_leaves": [25, 31],                    # 2 valores
            "min_child_samples": [15, 20],             # 2 valores
            "feature_fraction": [0.7, 0.8],            # 2 valores
            "bagging_fraction": [0.7, 0.8],            # 2 valores
            "bagging_freq": [1],                       # 1 valor (fijo)
            "reg_alpha": [0.0, 0.1],                   # 2 valores
            "reg_lambda": [1.0, 2.0],                  # 2 valores
        }
        # Total: 3×3×3×2×2×2×2×1×2×2 = 1,728 combinaciones
        # Con 5 folds: 8,640 fits (~30-40 min)

    elif modelo_nombre.lower() == 'xgboost':
        # ✅ Calcular scale_pos_weight una vez con todo y_train
        if y_train is not None:
            neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
            spw = (neg / pos) if pos > 0 else 1.0
            logger.info(f"   Calculado scale_pos_weight para XGBoost: {spw:.3f}")
        else:
            spw = 1.0
        
        modelo = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,  # ✅ Fijado aquí
            random_state=SEED,
            n_jobs=-1
        )
        # ✅ Parrilla PEQUEÑA centrada en tu baseline
        param_grid = {
            "n_estimators": [300, 400, 500],           # 3 valores
            "learning_rate": [0.03, 0.05, 0.07],       # 3 valores
            "max_depth": [3, 4, 5],                    # 3 valores
            "min_child_weight": [1, 3],                # 2 valores
            "subsample": [0.7, 0.8],                   # 2 valores
            "colsample_bytree": [0.7, 0.8],            # 2 valores
            "gamma": [0, 0.1],                         # 2 valores
            "reg_alpha": [0.0, 0.1],                   # 2 valores
            "reg_lambda": [1.0, 2.0],                  # 2 valores
        }
        # Total: 3×3×3×2×2×2×2×2×2 = 1,728 combinaciones
        # Con 5 folds: 8,640 fits (~30-45 min)

    elif modelo_nombre.lower() == 'randomforest':
        modelo = RandomForestClassifier(
            random_state=SEED,
            n_jobs=-1,
            class_weight="balanced"
        )
        # ✅ Parrilla PEQUEÑA centrada en tu baseline
        param_grid = {
            "n_estimators": [300, 400, 500, 600],      # 4 valores
            "max_depth": [10, 12, 15],                 # 3 valores
            "min_samples_split": [2, 4],               # 2 valores
            "min_samples_leaf": [1, 2],                # 2 valores
            "max_features": ["sqrt", "log2"],          # 2 valores
        }
        # Total: 4×3×2×2×2 = 96 combinaciones
        # Con 5 folds: 480 fits (~15-25 min)

    else:
        raise ValueError(f"Modelo '{modelo_nombre}' no soportado")
        
    return modelo, param_grid

def contar_combinaciones(param_grid: dict) -> int:
    """Cuenta total de combinaciones en la parrilla."""
    total = 1
    for v in param_grid.values():
        total *= len(v)
    return total

# =========================
# EVALUACIÓN EN TEST HOLD-OUT
# =========================
def evaluar_en_test_holdout(modelo_entrenado, df_test, feats, target):
    """Evalúa el mejor modelo en test hold-out."""
    X_test = df_test[feats]
    y_test = df_test[target].astype(int)
    
    y_pred = modelo_entrenado.predict(X_test)
    f1_test = f1_score(y_test, y_pred, average='macro')
    
    return f1_test

# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Optimización con GridSearchCV + Walk-Forward CV."
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="RandomForest",
        help="Modelo a optimizar (LightGBM, XGBoost, RandomForest)"
    )
    parser.add_argument(
        "--csv", 
        type=str, 
        default="datos_salida/finales/dataset_entrenamiento_final.csv",
        help="Ruta al CSV"
    )
    parser.add_argument(
        "--target",
        type=str,
        default="riesgo_fichaje",
        help="Columna objetivo"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Cores para paralelización (-1 = todos)"
    )
    args = parser.parse_args()

    # 1. Cargar datos
    df = pd.read_csv(args.csv, sep=";", encoding="utf-8-sig")
    logger.info(f"📂 Dataset: {len(df)} filas, {df.shape[1]} columnas")

    # 2. Definir temporadas y hold-out
    temporadas = sorted(
        df["season"].unique().tolist(),
        key=lambda s: tuple(map(int, s.split('_')))
    )
    
    last_season = temporadas[-1]
    df_train_cv = df[df["season"] != last_season].copy()
    df_test = df[df["season"] == last_season].copy()
    
    # ✅ CRÍTICO: Resetear índices ANTES de crear X_train
    df_train_cv = df_train_cv.reset_index(drop=True)
    
    logger.info(f"🎯 Test Hold-out: {last_season} ({len(df_test)} registros)")
    logger.info(f"🛠️ Optimización: {len(df_train_cv)} registros (CV)")

    # 3. Features y target
    feats = seleccionar_features(df_train_cv, args.target)
    X_train = df_train_cv[feats]
    y_train = df_train_cv[args.target].astype(int)

    # 4. Modelo base + parrilla
    modelo_base, param_grid = get_modelo_y_params(args.modelo, y_train)
    
    # 5. Construir CV Walk-Forward
    temporadas_cv = sorted(
        df_train_cv["season"].unique().tolist(),
        key=lambda s: tuple(map(int, s.split('_')))
    )
    folds_temporales = construir_folds_temporales(temporadas_cv)
    cv_splitter = crear_splits_indices_manual(df_train_cv, folds_temporales)

    if not cv_splitter:
        logger.error("No se pudieron crear folds. Abortando.")
        return

    total_combinaciones = contar_combinaciones(param_grid)
    logger.info(f"🔢 Combinaciones en grid: {total_combinaciones}")
    logger.info(f"🔁 Folds walk-forward: {len(cv_splitter)}")
    logger.info(f"⚙️ Total de fits: {total_combinaciones * len(cv_splitter)}")

    # 6. GridSearchCV
    f1_scorer = make_scorer(f1_score, average="macro")
    grid_search = GridSearchCV(
        estimator=modelo_base,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=cv_splitter,
        n_jobs=args.n_jobs,
        verbose=2,
        return_train_score=True
    )

    logger.info(f"\n▶️ Iniciando Grid Search para {args.modelo}...")
    start_time = time()
    
    grid_search.fit(X_train, y_train)
    
    end_time = time()

    # 7. Resultados en CV
    logger.info("\n" + "="*80)
    logger.info(f"🏁 GRID SEARCH COMPLETADO: {args.modelo.upper()}")
    logger.info("="*80)
    logger.info(f"⏱️ Tiempo total: {(end_time - start_time)/60:.2f} minutos")
    logger.info(f"🏆 Mejor F1-Macro (CV): {grid_search.best_score_:.4f}")
    logger.info("\n🛠️ Mejores Hiperparámetros:")
    
    best_params_str = json.dumps(grid_search.best_params_, indent=2)
    logger.info(best_params_str)
    
    # 8. Evaluación en test hold-out
    logger.info("\n" + "="*80)
    logger.info(f"🎯 EVALUACIÓN EN TEST HOLD-OUT ({last_season})")
    logger.info("="*80)
    
    f1_test = evaluar_en_test_holdout(
        grid_search.best_estimator_, 
        df_test, 
        feats, 
        args.target
    )
    
    logger.info(f"📊 F1-Score en Test Hold-out: {f1_test:.4f}")
    logger.info(f"📈 Diferencia CV→Test: {f1_test - grid_search.best_score_:+.4f}")
    
    # 9. Instrucciones finales
    logger.info("\n" + "="*80)
    logger.info("📝 PRÓXIMOS PASOS:")
    logger.info("="*80)
    logger.info("1. Copia los hiperparámetros de arriba")
    logger.info("2. Pégalos en crear_modelos() de entrenar_modelos_comparativo.py")
    logger.info("3. Re-ejecuta entrenar_modelos_comparativo.py")
    logger.info("4. Compara con otros modelos optimizados")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    main()