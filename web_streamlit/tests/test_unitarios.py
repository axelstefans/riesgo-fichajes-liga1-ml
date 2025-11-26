import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Añadir la raíz del proyecto al path para poder importar utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.model_io import load_prediction_assets
from utils.featurize import crear_features_numericas

# Definir ruta a assets para las pruebas
ASSETS_DIR = Path(__file__).parent.parent / "assets"

def test_existencia_y_carga_assets():
    """Prueba 1: Verifica que el modelo y los metadatos existan y carguen."""
    try:
        model, metadata, _ = load_prediction_assets(ASSETS_DIR)
        assert model is not None, "El modelo no debería ser None"
        assert isinstance(metadata, dict), "Los metadatos deben ser un diccionario"
    except FileNotFoundError:
        pytest.fail("No se encontraron los archivos .joblib o .json en la carpeta assets")

def test_integridad_metadatos():
    """Prueba 2: Verifica que los metadatos tengan las claves necesarias."""
    _, metadata, _ = load_prediction_assets(ASSETS_DIR)
    claves_necesarias = ['features_list', 'optimal_threshold', 'model_name']
    for clave in claves_necesarias:
        assert clave in metadata, f"Falta la clave crítica {clave} en metadata"

def test_feature_engineering_logica():
    """Prueba 3: Verifica que la creación de features no rompa con datos dummy."""
    # Crear un dataframe de prueba con 1 jugador
    datos_dummy = pd.DataFrame({
        'minutesPlayed': [900],
        'goals': [2],
        'totalShots': [10],
        'totalPasses': [100],
        'accuratePasses': [80]
    })
    
    # Ejecutar la función de tu proyecto
    df_res = crear_features_numericas(datos_dummy)
    
    # Validaciones
    assert 'goals_p90' in df_res.columns, "Debe crear la columna goals_p90"
    assert df_res.iloc[0]['goals_p90'] == 0.2, "El cálculo de p90 es incorrecto (2/10 part = 0.2)"
    assert df_res.iloc[0]['accuratePassesPercentage'] == 80.0, "El cálculo de porcentaje es incorrecto"