# utils/llm_analysis.py
import os
from groq import Groq
import logging

logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN GROQ ---
# 1. Pega aquí tu API Key de Groq (Empieza con 'gsk_...')
GROQ_API_KEY = "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" 

# 2. Selección del Modelo
# Recomendado: "llama3-70b-8192" (Muy inteligente y rápido)
# Opción B: "llama3-8b-8192" (Extremadamente rápido, menos razonamiento)
# Opción C: "mixtral-8x7b-32768" (Excelente balance)
MODEL_NAME = "llama-3.3-70b-versatile"

def generar_analisis_ia(nombre, posicion, edad, riesgo_etiqueta, probabilidad, factores_clave):
    """
    Genera un análisis narrativo utilizando la API ultra-rápida de Groq.
    """
    try:
        # --- 1. INICIALIZAR CLIENTE ---
        client = Groq(api_key=GROQ_API_KEY)

        # --- 2. PREPARACIÓN DE DATOS ---
        lista_factores_texto = ""
        for f in factores_clave:
            lista_factores_texto += f"- {f['nombre']}: {f['valor']} ({f['impacto']})\n"

        # --- 3. CONSTRUCCIÓN DEL PROMPT ---
        # Prompt del Sistema (La Personalidad)
        system_prompt = (
            "Eres un Director Deportivo de élite especializado en Scouting de fútbol. "
            "Tu trabajo es redactar informes técnicos breves, persuasivos y profesionales."
        )

        # Prompt del Usuario (Los Datos)
        user_prompt = f"""
        Redacta un análisis sobre este posible fichaje:

        DATOS DEL JUGADOR:
        - Nombre: {nombre} ({edad} años)
        - Posición Evaluada: {posicion}
        
        RESULTADO DEL ALGORITMO:
        - Riesgo: {riesgo_etiqueta} ({probabilidad:.1f}%)
        
        FACTORES CLAVE (Matemáticos):
        {lista_factores_texto}

        INSTRUCCIONES:
        1. Escribe en dos párrafos fluidos (6 líneas por párrafo).
        2. NO uses listas ni viñetas.
        3. NO menciones términos técnicos crudos (ej: 'p/90'). Tradúcelos a cualidades (ej: 'gran visión', 'falta de gol').
        4. Justifica el riesgo ({riesgo_etiqueta}) conectándolo con los factores.
        5. Sé directo y profesional.
        """

        # --- 4. LLAMADA A LA API ---
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=MODEL_NAME,
            temperature=0.6, # Creatividad moderada para ser preciso
            max_tokens=500,
        )

        # Retornar el contenido
        return chat_completion.choices[0].message.content

    except Exception as e:
        logger.error(f"Error Groq API: {e}")
        return f"El análisis narrativo no está disponible. (Error técnico: {str(e)})"