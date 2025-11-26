# app.py
import streamlit as st

# ----------------------------
# Configuración base
# ----------------------------
st.set_page_config(
    page_title="Riesgo Fichajes Liga 1",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Tema claro y profesional
# ----------------------------
st.markdown("""
<style>
/* Ocultar navegación nativa de Streamlit */
[data-testid="stSidebarNav"] { display: none; }

/* Títulos principales */
.main h1 { 
    font-size: 2.2rem !important; 
    color: #1e3a8a; 
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.main h2 { 
    font-size: 1.6rem !important; 
    color: #1e40af; 
    font-weight: 500;
    margin-top: 1.5rem;
}

/* Subtítulos */
.main h3 { 
    font-size: 1.3rem !important; 
    color: #3730a3; 
    font-weight: 500;
}

/* Métricas */
[data-testid="stMetricValue"] { 
    font-size: 1.8rem;
    font-weight: 600;
}

/* Botones */
.stButton>button {
    width: 100%; 
    background-color: #2563eb; 
    color: white; 
    font-weight: 600;
    border-radius: 8px; 
    padding: 0.6rem 1.2rem; 
    border: none;
    font-size: 1rem;
    transition: all 0.3s ease;
}

.stButton>button:hover { 
    background-color: #1d4ed8;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

/* Expanders */
.streamlit-expanderHeader { 
    background-color: #f1f5f9; 
    border-radius: 8px; 
    font-weight: 600;
    font-size: 1.1rem;
    padding: 0.75rem 1rem;
}

/* Captions y texto subtle */
.stCaption, .subtle { 
    color: #64748b; 
    font-size: 0.9rem; 
}

/* Dividers */
hr {
    margin: 1.5rem 0;
    border-color: #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar
# ----------------------------
try:
    from components.sidebar import render_sidebar
    modo = render_sidebar()
except Exception as e:
    st.error(f"Error al cargar sidebar: {e}")
    st.sidebar.title("Estimador de Riesgo")
    st.sidebar.caption("Liga 1 Perú")
    modo = st.sidebar.radio(
        "Navegación Principal:",
        ["Inicio", "Evaluación Individual", "Evaluación por Lotes", "Análisis del Modelo"],
        index=0
    )

def render_home():
    st.title("Estimador de Riesgo de Fichajes")
    st.caption("Prototipo de Machine Learning para el análisis predictivo del riesgo en fichajes en la Liga 1 Peruana")
    st.divider()
    
    st.markdown(
        """
        Herramienta de apoyo que analiza patrones de rendimiento para complementar la toma de decisiones.
        """
    )
    
    with st.expander("ℹ️ Detalles técnicos del modelo"):
        st.markdown(
            """
            - **Modelo:** RandomForest (Ensemble Learning)
            - **Rendimiento:** F1-Score de 70.7% en test hold-out (temporada 2024-2025)
            - **Features:** 31 métricas técnico-tácticas normalizadas por 90 minutos
            - **Explicabilidad:** Análisis SHAP (XAI) para interpretación de predicciones
            
            El objetivo no es reemplazar la observación humana, sino complementarla 
            con evidencia cuantitativa objetiva basada en patrones de rendimiento históricos.
            """
        )
    
    st.info("👈 Utilice el **menú de navegación** para acceder a la evaluación individual de jugadores.")
    st.divider()
    
    # ✅ CORREGIDO: Solo 1 card para Evaluación Individual
    st.markdown("### Evaluación Individual de Jugadores")
    st.markdown(
        """
        Ingrese las métricas de rendimiento de un jugador para recibir un análisis detallado 
        de riesgo, incluyendo:
        
        - **Probabilidad de Alto Riesgo** basada en patrones históricos
        - **Clasificación** (Bajo Riesgo / Alto Riesgo)
        - **Explicabilidad con SHAP** mostrando qué factores influyen en la predicción
        - **Interpretación** del nivel de riesgo detectado
        
        Puede seleccionar jugadores de ejemplo precargados o ingresar datos manualmente.
        """
    )

# ----------------------------
# Router
# ----------------------------
try:
    if modo == "Inicio":
        render_home()
    
    elif modo == "Evaluación Individual":
        from pages.individual import render as render_individual
        render_individual()
    
    # ✅ ELIMINADAS: Referencias a lotes y analisis
    
    else:
        st.error(f"⚠️ Modo no reconocido: '{modo}'")
        render_home()

except ImportError as e:
    st.error(f"❌ Error al importar módulo de página: {e}")
    st.info("Verifica que los archivos en pages/ existan y sean importables")

except Exception as e:
    st.error(f"❌ Error al renderizar '{modo}'")
    with st.expander("🔍 Ver detalles del error"):
        st.exception(e)