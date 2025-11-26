# pages/analisis.py
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st

# Rutas por defecto
COMPARISON_DIR = Path("reports/final_model_comparison")
RESUMEN_COMPARATIVO_CSV = COMPARISON_DIR / "resumen_comparativo.csv"
RESUMEN_ULT_TEMP_CSV = COMPARISON_DIR / "resumen_ult_temporada.csv"

MODEL_PATH = Path("assets/model_xgb.joblib")  # puedes cambiarlo si usas otro
FEATURE_ORDER_PATH = Path("assets/model_feature_order.json")
METADATA_PATH = Path("assets/metadata.json")


@st.cache_data(show_spinner=False)
def _load_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass
    return None


@st.cache_resource(show_spinner=False)
def _load_model(path: Path):
    try:
        import joblib
        if path.exists():
            return joblib.load(path)
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _best_model_by_f1(df_resumen: pd.DataFrame) -> str | None:
    if df_resumen is None or df_resumen.empty:
        return None
    if "F1_macro" not in df_resumen.columns or "Modelo" not in df_resumen.columns:
        return None
    return df_resumen.sort_values("F1_macro", ascending=False).iloc[0]["Modelo"]


def _feature_importance_df(model, feature_order: list[str] | None, top_k=15) -> pd.DataFrame | None:
    try:
        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            return None
        importances = np.asarray(importances).ravel()
        if feature_order is None or len(feature_order) != len(importances):
            # Intento de fallback si el modelo es un Pipeline simple
            try:
                # Extrae estimador final si es pipeline
                final_est = getattr(model, "steps", [[None, model]])[-1][1]
                importances = np.asarray(getattr(final_est, "feature_importances_", None)).ravel()
                model = final_est
            except Exception:
                return None
        names = feature_order if feature_order is not None else [f"f_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"Variable": names, "Importancia": importances})
        df = df.sort_values("Importancia", ascending=False).head(top_k).reset_index(drop=True)
        df["Ranking"] = df.index + 1
        # Reordenar columnas
        return df[["Ranking", "Variable", "Importancia"]]
    except Exception:
        return None


def render():
    st.title("Análisis del modelo")
    st.caption("Panel dinámico: se alimenta de los artefactos más recientes del entrenamiento.")
    st.markdown("---")

    # Cargar artefactos
    df_resumen = _load_csv(RESUMEN_COMPARATIVO_CSV)
    df_ult = _load_csv(RESUMEN_ULT_TEMP_CSV)
    model = _load_model(MODEL_PATH)
    feature_order = _load_json(FEATURE_ORDER_PATH)
    meta = _load_json(METADATA_PATH) or {}

    # Bloque de estado de artefactos
    with st.expander("Estado de artefactos", expanded=False):
        cols = st.columns(4)
        cols[0].write("**resumen_comparativo.csv**")
        cols[0].success("✓ Encontrado") if df_resumen is not None else cols[0].warning("No encontrado")

        cols[1].write("**resumen_ult_temporada.csv**")
        cols[1].success("✓ Encontrado") if df_ult is not None else cols[1].warning("No encontrado")

        cols[2].write("**Modelo**")
        cols[2].success("✓ Cargado") if model is not None else cols[2].warning("No cargado")

        cols[3].write("**feature_order.json**")
        cols[3].success("✓ Encontrado") if feature_order is not None else cols[3].warning("No encontrado")

    # Info general del modelo
    st.subheader("Información")
    info_cols = st.columns(3)
    algo_name = type(model).__name__ if model is not None else "—"
    info_cols[0].metric("Algoritmo", algo_name)
    info_cols[1].metric("Umbral (metadata)", f"{meta.get('threshold', 0.50):.2f}" if isinstance(meta.get('threshold', None), (int, float)) else "—")
    info_cols[2].metric("Artefactos", "Actualizados" if all([df_resumen is not None, df_ult is not None, model is not None]) else "Incompletos")

    st.markdown("---")

    # Métricas comparativas (promedio walk-forward)
    st.subheader("Comparativo por modelo (promedio Walk-Forward)")
    if df_resumen is None or df_resumen.empty:
        st.info("No se encontró `resumen_comparativo.csv`. Ejecuta el script de entrenamiento comparativo para generarlo.")
    else:
        # Ordenar por F1 desc y mostrar
        show_cols = [
            "Modelo", "Accuracy", "Precision_macro", "Recall_macro",
            "F1_macro", "ROC_AUC_macro", "Balanced_Acc"
        ]
        disp = df_resumen.copy()
        disponibles = [c for c in show_cols if c in disp.columns]
        disp = disp[disponibles].sort_values("F1_macro", ascending=False)
        st.dataframe(disp, use_container_width=True)

        # Mejor modelo por F1
        best = _best_model_by_f1(df_resumen)
        if best is not None:
            st.success(f"Mejor modelo (promedio F1): **{best}**")

    st.markdown("---")

    # Desempeño en la última temporada (del mejor modelo)
    st.subheader("Rendimiento en la última temporada")
    if df_ult is None or df_ult.empty:
        st.info("No se encontró `resumen_ult_temporada.csv`.")
    else:
        if df_resumen is not None and not df_resumen.empty:
            best = _best_model_by_f1(df_resumen)
            df_show = df_ult.copy()
            if best is not None and "Modelo" in df_show.columns:
                df_show = df_show[df_show["Modelo"] == best]
                if df_show.empty:
                    df_show = df_ult.copy()  # fallback: mostrar todo
            st.dataframe(
                df_show.sort_values("F1_macro", ascending=False),
                use_container_width=True
            )
        else:
            st.dataframe(df_ult.sort_values("F1_macro", ascending=False), use_container_width=True)

    st.markdown("---")

    # Importancia de variables (si el modelo lo soporta)
    st.subheader("Importancia de variables (si está disponible)")
    if model is None:
        st.info("Modelo no cargado; no se puede calcular importancia.")
    else:
        feat_imp = _feature_importance_df(model, feature_order)
        if feat_imp is None or feat_imp.empty:
            st.info("El modelo no expone `feature_importances_` o el orden de features no coincide.")
        else:
            st.dataframe(feat_imp, use_container_width=True)

    st.markdown("---")

    # Descargas rápidas
    st.subheader("Descargas")
    c1, c2 = st.columns(2)
    if df_resumen is not None:
        c1.download_button(
            "Descargar resumen_comparativo.csv",
            data=df_resumen.to_csv(index=False).encode("utf-8-sig"),
            file_name="resumen_comparativo.csv",
            mime="text/csv",
        )
    else:
        c1.button("Descargar resumen_comparativo.csv", disabled=True)

    if df_ult is not None:
        c2.download_button(
            "Descargar resumen_ult_temporada.csv",
            data=df_ult.to_csv(index=False).encode("utf-8-sig"),
            file_name="resumen_ult_temporada.csv",
            mime="text/csv",
        )
    else:
        c2.button("Descargar resumen_ult_temporada.csv", disabled=True)

    # Nota de límites (sin confusión ni SHAP global porque dependen del set de test/datos)
    with st.expander("Notas y límites", expanded=False):
        st.write(
            "- Este panel no fija métricas: **refleja los CSV generados por tu último entrenamiento**.\n"
            "- La matriz de confusión y SHAP global requieren datos de test; no se generan aquí para evitar desalineaciones."
        )
