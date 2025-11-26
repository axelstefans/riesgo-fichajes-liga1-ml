# pages/lotes.py
# -*- coding: utf-8 -*-

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

from utils.feature_utils import align_and_cast


# =========================
# Rutas a assets del modelo
# =========================
APP_ROOT = Path(__file__).resolve().parents[1]        # .../Web Streamlit
ASSETS_DIR = APP_ROOT / "assets"

MODEL_JOBLIB_PATH = ASSETS_DIR / "model_xgb.joblib"
FEATURE_ORDER_PATH = ASSETS_DIR / "model_feature_order.json"
METADATA_PATH = ASSETS_DIR / "metadata.json"          # opcional


# =========================
# Parámetros de negocio
# =========================
MIN_MINUTES = 450   # Filtro para robustez (ruido por-90)


# =========================
# Utilidades numéricas
# =========================
def _safe_div(num, den, default=0.0):
    try:
        num = float(num)
        den = float(den)
        if den == 0:
            return default
        return num / den
    except Exception:
        return default


def _per90(count, minutes):
    return 90.0 * _safe_div(count, minutes, default=0.0)


def _pct(part, total):
    return 100.0 * _safe_div(part, total, default=0.0)


# =========================
# Carga de modelo y columnas
# =========================
@st.cache_resource(show_spinner=False)
def _load_model_and_columns():
    if not FEATURE_ORDER_PATH.exists():
        st.error(f"No encuentro el orden de features: {FEATURE_ORDER_PATH}")
        st.stop()

    try:
        feat_order = json.loads(FEATURE_ORDER_PATH.read_text(encoding="utf-8"))
        if not isinstance(feat_order, list):
            raise ValueError("model_feature_order.json debe ser una lista de columnas.")
    except Exception as e:
        st.error(f"Error leyendo model_feature_order.json: {e}")
        st.stop()

    threshold = 0.5
    if METADATA_PATH.exists():
        try:
            meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
            threshold = float(meta.get("decision_threshold", threshold))
        except Exception:
            pass

    if not MODEL_JOBLIB_PATH.exists():
        st.error(f"No encuentro el modelo entrenado: {MODEL_JOBLIB_PATH}")
        st.stop()

    try:
        model = load(MODEL_JOBLIB_PATH)
    except Exception as e:
        st.error(f"No pude cargar el modelo .joblib: {e}")
        st.stop()

    return model, feat_order, threshold


# =====================================
# Derivación de features desde datos crudos
# =====================================
RAW_REQUIRED = [
    # Identificación / contexto
    "nombre_jugador", "edad", "posicion", "nacionalidad",

    # Minutos + presencias
    "minutos_jugados_total", "partidos_totales", "partidos_titular",

    # Ofensivo
    "goles", "asistencias", "tiros_totales", "tiros_puerta", "pases_clave",

    # Técnico
    "pases_completados", "pases_totales", "duelos_ganados", "duelos_totales",

    # Defensivo
    "tackles", "intercepciones", "duelos_aereos_ganados", "duelos_aereos_totales",

    # Posesión / pérdidas
    "balones_perdidos", "posesion_perdida", "toques",  # 'toques' es útil p/ toques_balon_90 si lo tienes

    # Disciplina
    "tarjetas_amarillas", "tarjetas_rojas", "faltas_cometidas",

    # Opcionales (si existen, mejor):
    "big_chances_created", "big_chances_missed", "regates_exitosos",
    "accurate_crosses", "total_cross", "goals_inside_box", "goals_outside_box",
    "was_fouled"
]

def _ensure_columns(df: pd.DataFrame, cols: list):
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def derive_features_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Normalizar nombres si el usuario cambió mayúsculas/minúsculas
    df.columns = [c.strip() for c in df.columns]

    # Asegurar columnas mínimas
    df = _ensure_columns(df, RAW_REQUIRED)

    # Tipos básicos
    for c in [
        "edad", "minutos_jugados_total", "partidos_totales", "partidos_titular",
        "goles", "asistencias", "tiros_totales", "tiros_puerta", "pases_clave",
        "pases_completados", "pases_totales", "duelos_ganados", "duelos_totales",
        "tackles", "intercepciones", "duelos_aereos_ganados", "duelos_aereos_totales",
        "balones_perdidos", "posesion_perdida", "toques",
        "tarjetas_amarillas", "tarjetas_rojas", "faltas_cometidas",
        "big_chances_created", "big_chances_missed", "regates_exitosos",
        "accurate_crosses", "total_cross", "goals_inside_box", "goals_outside_box",
        "was_fouled"
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derivados por-90
    m = df["minutos_jugados_total"].fillna(0)

    df["contribuciones_gol_90"] = _per90(df["goles"].fillna(0) + df["asistencias"].fillna(0), m)
    df["pases_clave_90"] = _per90(df["pases_clave"].fillna(0), m)
    df["tiros_puerta_90"] = _per90(df["tiros_puerta"].fillna(0), m)

    df["tackles_90"] = _per90(df["tackles"].fillna(0), m)
    df["intercepciones_90"] = _per90(df["intercepciones"].fillna(0), m)
    df["duelos_aereos_90"] = _per90(df["duelos_aereos_ganados"].fillna(0), m)

    # Pérdidas / posesión / toques
    df["perdidas_posesion_90"] = _per90(df["posesion_perdida"].fillna(0), m)
    # Si tienes 'toques' brutos:
    df["toques_balon_90"] = _per90(df["toques"].fillna(0), m)

    # Disciplina / faltas
    df["faltas_cometidas_90"] = _per90(df["faltas_cometidas"].fillna(0), m)
    df["tarjetas_90"] = _per90(df["tarjetas_amarillas"].fillna(0) + df["tarjetas_rojas"].fillna(0), m)

    # Porcentajes
    df["titularidad_pct"] = _pct(df["partidos_titular"].fillna(0), df["partidos_totales"].fillna(0))
    df["duelos_ganados_pct"] = _pct(df["duelos_ganados"].fillna(0), df["duelos_totales"].fillna(0))
    df["pases_acertados_pct"] = _pct(df["pases_completados"].fillna(0), df["pases_totales"].fillna(0))

    # Big chances / regates (si existen)
    df["big_chances_created_90"] = _per90(df.get("big_chances_created", 0), m)
    df["big_chances_missed_90"] = _per90(df.get("big_chances_missed", 0), m)
    df["regates_exitosos_90"] = _per90(df.get("regates_exitosos", 0), m)

    # Centros: ratio
    total_cross = df.get("total_cross", np.nan)
    acc_cross = df.get("accurate_crosses", np.nan)
    df["ratio_centros_acertados"] = _safe_div(acc_cross, total_cross, default=0.0)

    # Goles en área: ratio
    inside = df.get("goals_inside_box", np.nan)
    outside = df.get("goals_outside_box", np.nan)
    df["ratio_goles_area"] = _safe_div(inside, (inside + outside), default=0.0)

    # Fue-fauleado (si existe)
    df["veces_faulado_90"] = _per90(df.get("was_fouled", 0), m)

    # Nacionalidad → dummies
    def _map_nacionalidad(n):
        n = str(n).strip().lower()
        if "peru" in n or "perú" in n:
            return "Perú"
        if "uruguay" in n:
            return "Uruguay"
        if "colombia" in n:
            return "Colombia"
        return "Otras"

    nat = df["nacionalidad"].apply(_map_nacionalidad)
    df["nac_Perú"] = (nat == "Perú").astype(int)
    df["nac_Uruguay"] = (nat == "Uruguay").astype(int)
    df["nac_Colombia"] = (nat == "Colombia").astype(int)
    df["nac_Otras"] = (nat == "Otras").astype(int)

    # Posición → dummies (solo las usadas en tu dataset final)
    pos = df["posicion"].astype(str).str.lower()
    df["pos_Delantero"] = pos.str.contains("delan").astype(int)
    df["pos_Mediocampista"] = pos.str.contains("medio").astype(int)

    # Flags de contexto (opcionales); si el usuario no los provee, asumir 0
    for c in ["contexto_equipo_top", "proviene_liga_extranjera", "proviene_club_grande"]:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # edad y minutos ya vienen; asegurar tipo
    df["edad"] = pd.to_numeric(df["edad"], errors="coerce")
    df["minutos_jugados_total"] = pd.to_numeric(df["minutos_jugados_total"], errors="coerce")

    # Columnas finales candidatas (sin targets ni fugas)
    derived_cols = [
        "edad",
        "contribuciones_gol_90", "pases_clave_90", "tiros_puerta_90",
        "tackles_90", "faltas_cometidas_90", "tarjetas_90",
        "minutos_jugados_total", "titularidad_pct", "intercepciones_90",
        "duelos_ganados_pct", "big_chances_created_90", "big_chances_missed_90",
        "conversion_tiros_pct",  # OJO: si no existe origen directo, lo dejamos en 0
        "duelos_aereos_90", "regates_exitosos_90", "perdidas_posesion_90",
        "pases_acertados_pct", "ratio_goles_area", "ratio_centros_acertados",
        "veces_faulado_90", "toques_balon_90",
        "pos_Delantero", "pos_Mediocampista",
        "nac_Colombia", "nac_Otras", "nac_Perú", "nac_Uruguay",
        "contexto_equipo_top", "proviene_liga_extranjera", "proviene_club_grande"
    ]

    # Si 'conversion_tiros_pct' no puede calcularse con lo que trae el usuario, dejar 0
    if "conversion_tiros_pct" not in df.columns:
        df["conversion_tiros_pct"] = 0.0

    # Subconjunto derivado (sin alterar columnas originales)
    out = df.copy()

    # Mantener nombre para la salida final
    if "nombre_jugador" not in out.columns:
        out["nombre_jugador"] = ""

    # Devolver merged (para poder luego alinear contra feature_order)
    return out, derived_cols


# =====================================
# Alineación de features al orden de entrenamiento
# =====================================
def build_X_for_model(df_derived: pd.DataFrame, feature_order: list[str]) -> pd.DataFrame:
    X = pd.DataFrame(index=df_derived.index)
    missing = []
    for col in feature_order:
        if col in df_derived.columns:
            X[col] = pd.to_numeric(df_derived[col], errors="coerce")
        else:
            # Si falta la feature, imputar 0 y registrar
            X[col] = 0.0
            missing.append(col)

    # Warnings útiles
    if missing:
        st.warning(f"Columnas ausentes imputadas con 0: {missing}")

    # Seguridad ante NaNs
    X = X.fillna(0.0)
    return X


# =====================================
# Render principal (página lotes)
# =====================================
def render():
    st.title("Evaluación por Lotes")
    st.caption("Sube un Excel con datos crudos de jugadores; la app derivará las métricas por-90 y predecirá el **riesgo de fichaje**.")

    with st.expander("Instrucciones", expanded=True):
        st.markdown("""
        1) **Descarga la plantilla** y complétala con tus jugadores (métricas crudas).  
        2) **Sube el Excel**.  
        3) La app derivará automáticamente métricas por-90 y porcentajes requeridos.  
        4) Se filtrarán jugadores con **< 450 minutos** por robustez estadística.  
        5) Obtendrás **predicción, probabilidad** y podrás **descargar** resultados.
        """)

    # Botón descarga plantilla
    render_boton_plantilla()

    st.markdown("---")

    archivo = st.file_uploader(
        "Sube tu archivo Excel", type=["xlsx", "xls"],
        help="Usa la plantilla para asegurar nombres y formatos"
    )

    if not archivo:
        return

    try:
        df_raw = pd.read_excel(archivo)
    except Exception as e:
        st.error(f"No pude leer el Excel: {e}")
        return

    st.success(f"Archivo cargado: **{archivo.name}**")
    st.markdown("### Vista previa")
    st.dataframe(df_raw.head(10), use_container_width=True)

    # Derivar features
    df_derived, derived_cols = derive_features_from_raw(df_raw)

    # Filtro por minutos
    n_before = len(df_derived)
    df_derived = df_derived[df_derived["minutos_jugados_total"] >= MIN_MINUTES].copy()
    n_after = len(df_derived)
    dropped = n_before - n_after
    if dropped > 0:
        st.warning(f"{dropped} jugadores fueron omitidos por tener < {MIN_MINUTES} minutos.")

    if df_derived.empty:
        st.error("No quedaron jugadores con minutos suficientes.")
        return

    # Cargar modelo y columnas
    model, feat_order, threshold = _load_model_and_columns()

    # Construir X alineado
    X = build_X_for_model(df_derived, feat_order)
    X = X.astype(np.float32) 

    # Predicción
    with st.spinner("Calculando predicciones..."):
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[:, 1]
            else:
                # fallback (poco frecuente con XGBClassifier)
                scores = model.decision_function(X)
                # min-max a [0,1] para simular probas
                proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        except Exception as e:
            st.error(f"Error al predecir: {e}")
            return

    y_pred = (proba >= float(threshold)).astype(int)
    etiqueta = np.where(y_pred == 1, "Alto Riesgo", "Bajo Riesgo")

    # Armar resultados para mostrar/descargar
    out_cols = ["nombre_jugador"] if "nombre_jugador" in df_derived.columns else []
    resultados = pd.DataFrame(index=df_derived.index)
    if out_cols:
        resultados["nombre_jugador"] = df_derived["nombre_jugador"].astype(str)

    resultados["Predicción"] = etiqueta
    resultados["Probabilidad"] = np.round(proba, 4)
    resultados["Minutos"] = df_derived["minutos_jugados_total"].astype(int)
    resultados["Edad"] = df_derived["edad"].fillna(0).astype(int)
    resultados["Posición"] = df_derived.get("posicion", "").astype(str)
    resultados["Nacionalidad"] = df_derived.get("nacionalidad", "").astype(str)

    st.markdown("### Resultados")
    st.dataframe(resultados, use_container_width=True)

    # Descarga CSV
    csv = resultados.to_csv(index=False)
    st.download_button(
        label="Descargar Resultados (CSV)",
        data=csv,
        file_name="resultados_evaluacion_lotes.csv",
        mime="text/csv"
    )


def render_boton_plantilla():
    # Plantilla mínima con columnas crudas recomendadas
    plantilla_df = pd.DataFrame({
        "nombre_jugador": ["Ejemplo Jugador 1", "Ejemplo Jugador 2"],
        "edad": [25, 28],
        "posicion": ["Mediocampista", "Delantero"],
        "nacionalidad": ["Peruana", "Uruguaya"],
        "minutos_jugados_total": [1800, 2100],
        "partidos_totales": [22, 28],
        "partidos_titular": [16, 21],
        "goles": [5, 10],
        "asistencias": [6, 4],
        "tiros_totales": [30, 55],
        "tiros_puerta": [14, 28],
        "pases_clave": [18, 9],
        "pases_completados": [520, 380],
        "pases_totales": [680, 490],
        "duelos_ganados": [90, 110],
        "duelos_totales": [165, 190],
        "tackles": [28, 18],
        "intercepciones": [17, 12],
        "duelos_aereos_ganados": [25, 42],
        "duelos_aereos_totales": [45, 70],
        "balones_perdidos": [20, 25],
        "posesion_perdida": [27, 33],
        "toques": [1300, 1100],
        "tarjetas_amarillas": [3, 5],
        "tarjetas_rojas": [0, 1],
        "faltas_cometidas": [22, 27],
        # Opcionales (si los completas, mejor ajuste con tu set de entrenamiento):
        "big_chances_created": [5, 4],
        "big_chances_missed": [3, 2],
        "regates_exitosos": [20, 18],
        "accurate_crosses": [15, 10],
        "total_cross": [65, 45],
        "goals_inside_box": [7, 9],
        "goals_outside_box": [3, 1],
        "was_fouled": [36, 28],
        # Flags de contexto (0/1):
        "contexto_equipo_top": [0, 1],
        "proviene_liga_extranjera": [1, 0],
        "proviene_club_grande": [0, 0],
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        plantilla_df.to_excel(writer, sheet_name="Jugadores", index=False)
    buffer.seek(0)

    st.download_button(
        label="Descargar Plantilla Excel",
        data=buffer,
        file_name="plantilla_fichajes.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
