# web_streamlit/pages/individual.py
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import json
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import unicodedata

# --- IMPORTS DEL SISTEMA ---
from utils.model_io import (
    load_prediction_assets,
    predict_proba_safe,
    clasificar_riesgo,
    get_interpretacion_riesgo
)
from utils.featurize import featurize_single_player
from utils.sofascore import (
    buscar_jugador_sofascore, 
    obtener_stats_sofascore, 
    mapear_sofascore_a_app
)
from utils.llm_analysis import generar_analisis_ia 
from rapidfuzz import process, fuzz

# --- DICCIONARIO PARA TRADUCIR GRÁFICOS ---
DICCIONARIO_FEATURES = {
    "minutesPlayed": "Minutos Jugados",
    "edad": "Edad",
    "goals_p90": "Goles (p/90)",
    "assists_p90": "Asistencias (p/90)",
    "keyPasses_p90": "Pases Clave (p/90)",
    "successfulDribbles_p90": "Regates (p/90)",
    "aerialDuelsWon_p90": "Duelos Aéreos (p/90)",
    "accurateFinalThirdPasses_p90": "Pases Ofensivos (p/90)",
    "interceptions_p90": "Intercepciones (p/90)",
    "tackles_p90": "Entradas (p/90)",
    "clearances_p90": "Despejes (p/90)",
    "pos_Delantero": "Posición: Delantero",
    "pos_Mediocampista": "Posición: Medio",
    "pos_Defensa": "Posición: Defensa",
    "nac_Argentina": "Nac: Argentina",
    "nac_Perú": "Nac: Perú",
    "contexto_equipo_top": "Destino Equipo Top",
    "yellowCards_p90": "Tarjetas Amarillas (p/90)",
    "fouls_p90": "Faltas Cometidas (p/90)",
    "wasFouled_p90": "Faltas Recibidas (p/90)",
    "totalShots_p90": "Tiros Totales (p/90)",
    "shotsOnTarget_p90": "Tiros al Arco (p/90)",
    "accurateCrosses_p90": "Centros Precisos (p/90)",
    "accurateLongBalls_p90": "Balones Largos (p/90)",
    "accuratePassesPercentage": "% Pase Preciso",
    "goalConversionPercentage": "% Efectividad Gol"
}

# --- UTILIDADES UI ---
def _norm(text):
    if not text: return ""
    return unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode('utf-8').lower().strip()

def _reset_session():
    st.session_state.search_results = []
    st.session_state.datos_actuales = {}
    st.session_state.info_origen = ""

# --- CONFIGURACIÓN ---
st.set_page_config(layout="wide", page_title="Evaluación de Fichaje | IA Scout")

@st.cache_data(show_spinner=False)
def load_catalogs():
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        ASSETS_DIR = BASE_DIR / "assets"
        CATALOGO_PATH = ASSETS_DIR / "clubes_conmebol_autocomplete.json"
        LIGA1_PATH = ASSETS_DIR / "clubes_liga1.json"
        
        clubs_auto = []
        if CATALOGO_PATH.exists():
            with CATALOGO_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw_list = [str(c).strip() for lst in data.values() for c in lst]
            seen = set()
            for c in raw_list:
                if c not in seen: seen.add(c); clubs_auto.append(c)
        
        liga1_list = []
        if LIGA1_PATH.exists():
            with LIGA1_PATH.open("r", encoding="utf-8") as f:
                liga1_list = json.load(f)
                liga1_list.sort()
        else:
            liga1_list = ["Universitario", "Alianza Lima", "Sporting Cristal", "Melgar"]
            
        return clubs_auto, liga1_list
    except:
        return [], []

_, CLUBES_LIGA1 = load_catalogs()

@st.cache_resource(show_spinner="Cargando cerebro digital...")
def _load_artifacts():
    try:
        BASE_DIR = Path(__file__).resolve().parent.parent
        ASSETS_DIR = BASE_DIR / "assets"
        
        # === CORRECCIÓN CRÍTICA ===
        # load_prediction_assets devuelve 3 cosas: model, metadata, jugadores_ejemplo (lista)
        # Antes asignabas la lista a 'explainer', por eso fallaba.
        # Ahora usamos '_' para ignorar la lista.
        model, metadata, _ = load_prediction_assets(ASSETS_DIR)
        
        explainer = None
        try:
            explainer = shap.TreeExplainer(model)
            st.session_state["explainer_loaded"] = True
        except:
            st.session_state["explainer_loaded"] = False
                
        return model, metadata, explainer
    except Exception as e:
        raise e

def _mostrar_resultados(nombre_jugador, proba, explainer, X, metadata, datos_extra):
    st.markdown("---")
    
    umbral = metadata.get("decision_threshold", 0.5)
    etiqueta, _ = clasificar_riesgo(proba, umbral)
    proba_pct = proba * 100
    
    if etiqueta == "ALTO RIESGO":
        main_color, bg_color, icon = "#ef4444", "#fef2f2", "⚠️"
    else:
        main_color, bg_color, icon = "#22c55e", "#f0fdf4", "✅"

    # DASHBOARD
    with st.container(border=True):
        st.subheader(f"📊 Resultado del Análisis: {nombre_jugador}")
        
        c_text, c_gauge = st.columns([3, 2], gap="large")
        
        with c_text:
            st.markdown(f"""
                <div style="background-color: {bg_color}; border-left: 8px solid {main_color}; padding: 25px; border-radius: 8px; height: 100%; display: flex; flex-direction: column; justify-content: center;">
                    <div style="font-size: 2.5rem; font-weight: 800; color: {main_color}; display: flex; align-items: center; gap: 15px;">
                        {icon} {etiqueta}
                    </div>
                    <div style="margin-top: 10px; font-size: 1.2rem; color: #6b7280;">
                        Probabilidad de Riesgo Calculada: <strong>{proba_pct:.1f}%</strong>
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
        with c_gauge:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number", value = proba_pct,
                number = {'suffix': "%", 'font': {'size': 40, 'color': main_color}},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': main_color},
                    'bgcolor': "white",
                    'steps': [{'range': [0, umbral*100], 'color': '#dcfce7'}, {'range': [umbral*100, 100], 'color': '#fee2e2'}],
                    'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.75, 'value': proba_pct}
                }
            ))
            fig.update_layout(margin=dict(l=30,r=30,t=30,b=30), height=220, paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # ANÁLISIS IA
    if st.session_state.get("explainer_loaded", False) and explainer:
        st.markdown("---")
        st.subheader("🧠 Análisis del Director Deportivo (IA)")
        
        try:
            shap_values = explainer(X)
            if len(shap_values.values.shape) == 3:
                vals = shap_values.values[:, :, 1][0]
                base = shap_values.base_values[:, 1][0] if len(shap_values.base_values.shape) == 2 else shap_values.base_values[0]
            else:
                vals = shap_values.values[0]
                base = shap_values.base_values[0]

            indices = np.argsort(np.abs(vals))[::-1][:5]
            factores_clave = []
            
            for i in indices:
                feat_raw = X.columns[i]
                feat_name = DICCIONARIO_FEATURES.get(feat_raw, feat_raw)
                feat_val = X.iloc[0, i]
                impacto = "Aumenta Riesgo" if vals[i] > 0 else "Disminuye Riesgo"
                
                val_str = f"{feat_val:.2f}"
                if "_p90" in feat_raw: val_str += " p/90"
                elif "Percentage" in feat_raw: val_str += "%"
                elif "edad" in feat_raw: val_str += " años"
                elif "contexto" in feat_raw or "pos_" in feat_raw: val_str = "Sí" if feat_val > 0.5 else "No"

                factores_clave.append({"nombre": feat_name, "valor": val_str, "impacto": impacto})

            with st.spinner("🤖 La IA está redactando el informe técnico..."):
                analisis_texto = generar_analisis_ia(
                    nombre=nombre_jugador,
                    posicion=datos_extra.get("posicion", "Jugador"),
                    edad=datos_extra.get("edad", "25"),
                    riesgo_etiqueta=etiqueta,
                    probabilidad=proba_pct,
                    factores_clave=factores_clave
                )

            with st.container(border=True):
                st.info(f"📝 **Informe Técnico:**\n\n{analisis_texto}")

            st.write("")
            with st.expander("📉 Ver gráfico técnico detallado (SHAP Waterfall)", expanded=False):
                st.caption("Desglose matemático de las variables que más impactaron la decisión:")
                X_grafico = X.copy()
                X_grafico.columns = [DICCIONARIO_FEATURES.get(c, c) for c in X.columns]
                explanation = shap.Explanation(values=vals, base_values=base, data=X_grafico.iloc[0], feature_names=X_grafico.columns)
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(explanation, max_display=10, show=False)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
                
        except Exception as e:
            st.warning(f"No se pudo generar el detalle explicativo: {e}")

# ==============================================================================
# RENDERIZADO PRINCIPAL
# ==============================================================================
def render():
    st.title("🕵️ Asistente de Fichajes con IA")
    st.markdown("Evalúa el riesgo de contratar un jugador basándote en su **rendimiento real reciente**.")
    
    if "search_results" not in st.session_state: st.session_state.search_results = []
    if "datos_actuales" not in st.session_state: st.session_state.datos_actuales = {}
    if "info_origen" not in st.session_state: st.session_state.info_origen = ""

    try:
        model, metadata, explainer = _load_artifacts()
    except Exception as e:
        st.error(f"⚠️ ERROR TÉCNICO: {e}"); return

    # 1. BUSCADOR
    st.header("1. Buscar Jugador", divider="gray")
    with st.container(border=True):
        c1, c2 = st.columns([4, 1])
        with c1: search_query = st.text_input("Nombre del jugador:", placeholder="Ej: Paolo Guerrero...", label_visibility="collapsed")
        with c2:
            if st.button("🔍 Buscar", use_container_width=True):
                _reset_session()
                if len(search_query) > 2:
                    with st.spinner("Conectando..."):
                        st.session_state.search_results = buscar_jugador_sofascore(search_query)
        
        if st.session_state.search_results:
            opts = {f"{r['name']} ({r.get('team', {}).get('name', 'Libre')})" : r for r in st.session_state.search_results}
            selected = st.selectbox("Resultados encontrados:", list(opts.keys()))
            if st.button("⬇️ Importar Estadísticas", type="primary"):
                jugador = opts[selected]
                with st.spinner("Analizando..."):
                    res = obtener_stats_sofascore(jugador['id'])
                    if res:
                        stats = res.get('stats', {})
                        st.session_state.datos_actuales = mapear_sofascore_a_app(res, jugador)
                        st.session_state.info_origen = f"📅 Datos: {stats.get('src_tournament', 'Liga Local')} ({stats.get('minutesPlayed')} mins jugados)"
                        st.rerun()
                    else: st.error("Sin datos recientes.")

    # 2. VERIFICACIÓN
    datos = st.session_state.datos_actuales
    if datos:
        st.header("2. Verificar y Editar Datos", divider="gray")
        st.info(f"✅ **{st.session_state.info_origen}**")

        with st.container(border=True):
            col_img, col_info = st.columns([1, 3])
            with col_img:
                if datos.get("imagen_url"): st.image(datos["imagen_url"], width=200)
                else: st.write("👤")
            with col_info:
                st.text_input("Nombre del jugador", value=datos.get("nombre_jugador", "Jugador"), disabled=True)
                r1, r2 = st.columns(2)
                r1.text_input("Edad", value=datos.get("edad"), disabled=True)
                r1.text_input("Nacionalidad", value=datos.get("nacionalidad_str"), disabled=True)
                r2.text_input("Posición Original", value=datos.get("posicion"), disabled=True)
                r2.text_input("Club Actual", value=datos.get("club_origen"), disabled=True)

        with st.form("form_analisis"):
            st.markdown("### ⚙️ Configuración del Fichaje")
            
            club_destino = st.selectbox(
                "¿A qué club lo quieres fichar? *", 
                options=CLUBES_LIGA1, 
                index=None, 
                placeholder="Selecciona el equipo destino..."
            )
            
            st.markdown("---")
            st.markdown("#### 📝 Variables de Rendimiento (Edición Manual)")
            t_part, t_of, t_def = st.tabs(["⏱️ Participación", "⚽ Ofensiva", "🛡️ Defensa"])

            with t_part:
                c1, c2, c3 = st.columns(3)
                with c1:
                    minutesPlayed = st.number_input("Minutos jugados *", 0, 5000, int(datos.get("minutesPlayed", 0)))
                    appearances = st.number_input("Partidos jugados *", 0, 60, int(datos.get("appearances", 0)))
                with c2: yellowCards = st.number_input("Tarjetas amarillas", 0, 50, int(datos.get("yellowCards", 0)))
                with c3:
                    fouls = st.number_input("Faltas cometidas", 0, 300, int(datos.get("fouls", 0)))
                    wasFouled = st.number_input("Faltas recibidas", 0, 300, int(datos.get("wasFouled", 0)))

            with t_of:
                c1, c2, c3 = st.columns(3)
                with c1:
                    goals = st.number_input("Goles", 0, 100, int(datos.get("goals", 0)))
                    assists = st.number_input("Asistencias", 0, 100, int(datos.get("assists", 0)))
                    penaltyGoals = st.number_input("Goles penal", 0, 50, int(datos.get("penaltyGoals", 0)))
                    penaltiesTaken = st.number_input("Penales lanzados", 0, 50, int(datos.get("penaltiesTaken", 0)))
                with c2:
                    totalShots = st.number_input("Tiros totales", 0, 300, int(datos.get("totalShots", 0)))
                    shotsOnTarget = st.number_input("Tiros a puerta", 0, 200, int(datos.get("shotsOnTarget", 0)))
                    shotsOffTarget = st.number_input("Tiros fuera", 0, 200, int(datos.get("shotsOffTarget", 0)))
                    blockedShots = st.number_input("Tiros bloqueados", 0, 100, int(datos.get("blockedShots", 0)))
                with c3:
                    keyPasses = st.number_input("Pases clave", 0, 150, int(datos.get("keyPasses", 0)))
                    successfulDribbles = st.number_input("Regates exitosos", 0, 300, int(datos.get("successfulDribbles", 0)))
                    offsides = st.number_input("Offsides", 0, 150, int(datos.get("offsides", 0)))

            with t_def:
                c1, c2, c3 = st.columns(3)
                with c1:
                    totalPasses = st.number_input("Pases totales", 0, 3000, int(datos.get("totalPasses", 0)))
                    accuratePasses = st.number_input("Pases acertados", 0, 3000, int(datos.get("accuratePasses", 0)))
                    accurateFinalThirdPasses = st.number_input("Pases ult. tercio", 0, 1000, int(datos.get("accurateFinalThirdPasses", 0)))
                with c2:
                    accurateCrosses = st.number_input("Centros acertados", 0, 500, int(datos.get("accurateCrosses", 0)))
                    accurateLongBalls = st.number_input("Balones largos ok", 0, 500, int(datos.get("accurateLongBalls", 0)))
                    clearances = st.number_input("Despejes", 0, 500, int(datos.get("clearances", 0)))
                with c3:
                    aerialDuelsWon = st.number_input("Duelos aéreos", 0, 300, int(datos.get("aerialDuelsWon", 0)))
                    dribbledPast = st.number_input("Veces regateado", 0, 200, int(datos.get("dribbledPast", 0)))

            st.write("")
            submitted = st.form_submit_button("🚀 EJECUTAR ANÁLISIS DE RIESGO", use_container_width=True)

        if submitted:
            if not club_destino: st.error("⚠️ Falta Club Destino"); return
            if minutesPlayed == 0: st.error("⚠️ Minutos = 0"); return

            suma_tiros = shotsOnTarget + shotsOffTarget + blockedShots
            if suma_tiros > totalShots: totalShots = suma_tiros

            raw_data = {
                "edad": datos["edad"], "posicion": datos["posicion"], 
                "nacionalidad_str": datos["nacionalidad_str"], "club_origen": datos["club_origen"],
                "club_destino": club_destino,
                "minutesPlayed": minutesPlayed, "appearances": appearances,
                "yellowCards": yellowCards, "fouls": fouls, "wasFouled": wasFouled,
                "goals": goals, "assists": assists, "penaltyGoals": penaltyGoals,
                "penaltiesTaken": penaltiesTaken, "totalShots": totalShots,
                "shotsOnTarget": shotsOnTarget, "shotsOffTarget": shotsOffTarget,
                "blockedShots": blockedShots, "keyPasses": keyPasses,
                "successfulDribbles": successfulDribbles, "offsides": offsides,
                "totalPasses": totalPasses, "accuratePasses": accuratePasses,
                "accurateFinalThirdPasses": accurateFinalThirdPasses,
                "accurateCrosses": accurateCrosses, "accurateLongBalls": accurateLongBalls,
                "aerialDuelsWon": aerialDuelsWon, "dribbledPast": dribbledPast, "clearances": clearances,
            }

            try:
                with st.spinner("Calculando..."):
                    X = featurize_single_player(raw_data)
                    proba = predict_proba_safe(model, X)
                _mostrar_resultados(datos.get("nombre_jugador"), proba, explainer, X, metadata, raw_data)
            except Exception as e: st.error(f"Error: {e}")

    else:
        with st.container(border=True):
            st.info("👈 **Comienza aquí:** Escribe el nombre de un jugador en el buscador (Paso 1).")

if __name__ == "__main__":
    render()