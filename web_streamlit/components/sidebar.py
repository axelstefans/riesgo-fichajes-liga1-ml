# components/sidebar.py
import datetime
import streamlit as st

def render_sidebar() -> str:

    with st.sidebar:
        st.title("Navegación")
        st.caption("Seleccione el modo de análisis")
        st.divider()
        
        # ✅ CORREGIDO: Solo Inicio y Evaluación Individual
        opciones = [
            "Inicio",
            "Evaluación Individual",
        ]
        
        seleccion = st.radio(
            "Opciones Principales:",
            options=opciones,
            index=0,
            key="nav_mode",
        )
        
        st.divider()
        st.caption(f"© {datetime.date.today().year} - UPN")
        
        return seleccion