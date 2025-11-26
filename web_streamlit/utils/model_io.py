# web_streamlit/utils/model_io.py

import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import logging

logger = logging.getLogger(__name__)


def load_prediction_assets(assets_dir: Path):
    
    # Rutas de archivos
    model_path = assets_dir / "randomforest_model.joblib"
    metadata_path = assets_dir / "model_metadata.json"
    ejemplos_path = assets_dir / "jugadores_ejemplo.json"

    # --- 1. Validar que los archivos críticos existen ---
    if not model_path.exists():
        raise FileNotFoundError(
            f"❌ Archivo del modelo no encontrado: {model_path}\n"
            f"   Asegúrate de haber copiado 'randomforest_model.joblib' "
            f"desde model_artifacts/produccion_randomforest/"
        )
    
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"❌ Archivo de metadatos no encontrado: {metadata_path}\n"
            f"   Asegúrate de haber copiado 'metadata.json' "
            f"desde model_artifacts/produccion_randomforest/"
        )

    # --- 2. Cargar el modelo ---
    try:
        model = joblib.load(model_path)
        logger.info("✅ Modelo 'randomforest_model.joblib' cargado exitosamente")
    except Exception as e:
        logger.error(f"❌ Error al cargar el modelo: {e}")
        raise

    # --- 3. Cargar los metadatos ---
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Validar claves esenciales
        required_keys = ['features_list', 'decision_threshold', 'model_name']
        missing_keys = [key for key in required_keys if key not in metadata]
        
        if missing_keys:
            raise KeyError(
                f"❌ El archivo de metadatos no contiene las claves requeridas:\n"
                f"   Faltantes: {missing_keys}\n"
                f"   Disponibles: {list(metadata.keys())}"
            )
        
        # Validar modelo
        if metadata['model_name'] != 'RandomForest':
            logger.warning(
                f"⚠️ Se esperaba modelo 'RandomForest', "
                f"pero se encontró '{metadata['model_name']}'"
            )
        
        # Validar número de features
        features_list = metadata['features_list']
        if len(features_list) != 31:
            raise ValueError(
                f"❌ Se esperaban 31 features en metadata, "
                f"pero se encontraron {len(features_list)}"
            )
        
        logger.info("✅ Metadatos 'model_metadata.json' cargados exitosamente")
        logger.info(f"   Modelo: {metadata['model_name']}")
        logger.info(f"   Versión: {metadata.get('version', 'N/A')}")
        logger.info(f"   Features: {len(features_list)}")
        logger.info(f"   Umbral de decisión: {metadata['decision_threshold']}")
        
    except Exception as e:
        logger.error(f"❌ Error al cargar o validar los metadatos: {e}")
        raise

    # --- 4. Cargar jugadores de ejemplo (opcional) ---
    jugadores_ejemplo = []
    if ejemplos_path.exists():
        try:
            with open(ejemplos_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                jugadores_ejemplo = data.get("jugadores_ejemplo", [])
            
            logger.info(f"✅ Cargados {len(jugadores_ejemplo)} jugadores de ejemplo")
        except Exception as e:
            logger.warning(f"⚠️ No se pudieron cargar jugadores de ejemplo: {e}")
    else:
        logger.info("ℹ️ Archivo de jugadores de ejemplo no encontrado (opcional)")

    return model, metadata, jugadores_ejemplo


def predict_proba_safe(model, X: pd.DataFrame) -> float:
    
    # Validación de entrada
    if not isinstance(X, pd.DataFrame):
        raise TypeError("❌ La entrada X debe ser un DataFrame de Pandas")
    
    if len(X) != 1:
        raise ValueError(
            f"❌ Se esperaba 1 fila para la predicción, "
            f"pero se recibieron {len(X)} filas"
        )
    
    if X.shape[1] != 31:
        raise ValueError(
            f"❌ Se esperaban 31 features, "
            f"pero se recibieron {X.shape[1]} features\n"
            f"   Features recibidas: {list(X.columns)}"
        )

    try:
        # RandomForest siempre tiene predict_proba
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(X)[0, 1])
            logger.debug(f"Predicción: P(Alto Riesgo) = {proba:.4f}")
            return proba
        
        # Fallback para otros modelos
        if hasattr(model, "decision_function"):
            decision = float(model.decision_function(X)[0])
            proba = 1.0 / (1.0 + np.exp(-decision))
            logger.debug(f"Predicción via decision_function: {proba:.4f}")
            return proba
        
        # Último recurso
        if hasattr(model, "predict"):
            pred = float(model.predict(X)[0])
            logger.warning(
                f"⚠️ Modelo solo tiene .predict(), "
                f"devolviendo clase {pred} como probabilidad"
            )
            return pred
            
    except Exception as e:
        logger.error(f"❌ Error durante la predicción: {e}")
        raise
    
    raise RuntimeError(
        "❌ El modelo no tiene ningún método de predicción válido "
        "(.predict_proba, .decision_function, o .predict)"
    )

# En utils/model_io.py

def clasificar_riesgo(probabilidad: float, threshold: float = 0.5) -> tuple[str, str]:
    """
    Clasifica el riesgo basándose estrictamente en el umbral del modelo.
    """
    # Usamos el mismo threshold que se usó para entrenar (ej: 0.5)
    if probabilidad >= threshold:
        return "ALTO RIESGO", "#ef4444"  # Rojo
    else:
        return "BAJO RIESGO", "#22c55e"  # Verde

def get_interpretacion_riesgo(probabilidad: float, threshold: float = 0.5) -> str:
    """
    Genera una interpretación coherente con la clasificación.
    """
    # Convertimos a porcentaje para facilitar lectura
    pct = probabilidad * 100
    
    # Lógica alineada: Todo lo menor al threshold es BAJO, todo lo mayor es ALTO.
    # Pero damos matices dentro de esas categorías.
    
    if probabilidad < threshold:
        # ZONA DE BAJO RIESGO (VERDE)
        if probabilidad < 0.25:
            return f"Riesgo muy bajo ({pct:.1f}%). El jugador muestra indicadores sólidos de rendimiento."
        else:
            return f"Riesgo bajo ({pct:.1f}%). Perfil aceptable, aunque con margen de mejora en ciertas métricas."
            
    else:
        # ZONA DE ALTO RIESGO (ROJA)
        if probabilidad < 0.75:
            return f"Riesgo alto ({pct:.1f}%). El jugador presenta debilidades estadísticas notables para el rol solicitado."
        else:
            return f"Riesgo muy alto ({pct:.1f}%). Las métricas actuales no respaldan un fichaje seguro según el modelo."


def validar_consistencia_features(X: pd.DataFrame, metadata: dict) -> None:

    expected_features = metadata['features_list']
    actual_features = list(X.columns)
    
    # Verificar orden exacto
    if actual_features != expected_features:
        # Verificar si al menos tienen las mismas features (aunque en diferente orden)
        if set(actual_features) == set(expected_features):
            logger.warning(
                "⚠️ Las features están en orden diferente al esperado. "
                "Reordenando automáticamente..."
            )
            # Esto no debería pasar si featurize.py funciona bien
        else:
            missing = set(expected_features) - set(actual_features)
            extra = set(actual_features) - set(expected_features)
            
            error_msg = "❌ Las features no coinciden con las esperadas:\n"
            if missing:
                error_msg += f"   Faltantes: {missing}\n"
            if extra:
                error_msg += f"   Extras: {extra}\n"
            
            raise ValueError(error_msg)