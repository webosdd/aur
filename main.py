import os
import json
from openai import OpenAI
from pybit.unified_trading import HTTP
import requests

# --- Configuración Inicial (Usa tus API keys) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")

# Cliente para OpenRouter con el modelo que puede navegar
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://your-site.com",  # Reemplaza con tu URL
        "X-Title": "BTC Dual Analysis Bot"
    }
)

MODEL_NAME = "google/gemini-3-flash-preview:online" # <--- MODELO CON INTERNET

# --- Funciones Auxiliares (Telegram, Bybit, etc.) ---
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}
    try:
        requests.post(url, data=payload, timeout=10)
    except Exception as e:
        print(f"Error sending to Telegram: {e}")

# --- 1. Función de Análisis Técnico con el Gráfico ---
def get_technical_analysis(image_base64):
    prompt = """
    Actúa como un analista técnico cuantitativo. Analiza el gráfico diario de BTC/USDT proporcionado e identifica los siguientes elementos. Responde ÚNICAMENTE con un objeto JSON válido y sin texto adicional. Usa 'null' si un valor no es aplicable.

    {
      "precio_actual_usd": "número",
      "soporte_inmediato_usd": "número o null",
      "resistencia_inmediata_usd": "número o null",
      "tendencia": "Alcista | Bajista | Neutral | Consolidación",
      "confianza_tecnica_porcentaje": 0-100,
      "momentum": "Momentum Alcista | Momentum Bajista | Sin Momentum",
      "puntos_liquidez": ["descripción de zona de liquidity grab", "descripción de otra", "..."],
      "analisis_ema": "describe la posición del precio respecto a las EMAs 20, 50, 100 y 200",
      "analisis_rsi": "describe el nivel del RSI y su implicación",
      "analisis_macd": "describe el estado del MACD y su histograma",
      "resumen_tecnico": "resumen ejecutivo del análisis técnico"
    }
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=1000
        )
        content = response.choices[0].message.content
        # Limpiar posibles caracteres no JSON (como ```json)
        content = content.strip('`').replace('json', '', 1).strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error in technical analysis: {e}")
        return None

# --- 2. Función de Análisis Fundamental con Búsqueda Integrada ---
def get_fundamental_and_unified_analysis(technical_json):
    prompt = f"""
    Debes actuar como un analista fundamental y de sentimiento de mercados. Tu tarea es realizar un análisis del mercado de Bitcoin (BTC) para el día de hoy, 2026-05-28.

    Realiza una búsqueda en internet sobre los siguientes puntos clave, usando tu capacidad de navegación:
    1. **Sentimiento del Mercado**: Busca el "Crypto Fear and Greed Index" actual.
    2. **Flujo de ETFs de Bitcoin (Spot)**: Encuentra los datos más recientes de entradas/salidas de capital en los ETFs de Bitcoin al contado en EE. UU. (como IBIT de BlackRock).
    3. **Macroeconomía**: Identifica las noticias más relevantes sobre las decisiones de tasas de interés de la Reserva Federal (Fed) y su impacto en el dólar y los bonos del tesoro.
    4. **Regulaciones**: Busca las noticias más importantes sobre proyectos de ley o regulaciones que afecten a Bitcoin en EE. UU. (ej. Ley GENIUS, CLARITY Act).
    5. **Comportamiento de Ballenas**: Investiga si hay reportes recientes de movimientos significativos de ballenas (grandes tenedores) que estén acumulando o distribuyendo Bitcoin.
    6. **Noticias Destacadas**: Identifica otras 2-3 noticias importantes que puedan estar moviendo el precio de Bitcoin hoy.

    **Análisis Unificado**: Después de recopilar esta información, intégrala con el siguiente análisis técnico proporcionado:

    {json.dumps(technical_json, indent=2)}

    Finalmente, proporciona un veredicto unificado sobre la dirección esperada del precio de Bitcoin en las próximas horas/días (alcista, bajista o neutral). Sustenta tu respuesta en los datos clave de ambos análisis.

    Responde ÚNICAMENTE con un objeto JSON válido y sin texto adicional, siguiendo esta estructura exacta:

    {{
      "fear_greed_index": "número y texto (ej. 22 - Miedo Extremo)",
      "etf_flow_usd": "número en millones con signo (ej. -733)",
      "tendencias_macro": "texto breve sobre Fed, tasas, dólar",
      "regulaciones_clave": "texto breve sobre noticias regulatorias",
      "comportamiento_ballenas": "texto breve",
      "otras_noticias": ["noticia 1", "noticia 2"],
      "resumen_fundamental": "resumen ejecutivo del análisis fundamental",
      "veredicto_unificado": "Alcista | Bajista | Neutral",
      "confianza_unificada_porcentaje": 0-100,
      "explicacion_unificada": "por qué se espera esa dirección, basado en los datos de ambos análisis"
    }}
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1500
        )
        content = response.choices[0].message.content
        content = content.strip('`').replace('json', '', 1).strip()
        return json.loads(content)
    except Exception as e:
        print(f"Error in fundamental analysis: {e}")
        return None

# --- 3. Función Principal (Ejecutar análisis completo) ---
def run_analysis_cycle():
    print("🔄 Iniciando ciclo de análisis dual...")
    send_telegram_message("🔄 Iniciando ciclo de análisis dual de BTC...")

    # --- Paso 1: Generar imagen del gráfico (Ya debes tener esta función) ---
    # img_base64 = generate_price_chart_with_indicators() # <--- TU FUNCIÓN DE GRÁFICO
    # Simulamos que tenemos la imagen en base64 por ahora
    img_base64 = "..."

    # --- Paso 2: Análisis Técnico ---
    tech_analysis = get_technical_analysis(img_base64)
    if tech_analysis:
        send_telegram_message(f"📈 [ANÁLISIS TÉCNICO]\n"
                              f"Precio: ${tech_analysis.get('precio_actual_usd')}\n"
                              f"Soporte: ${tech_analysis.get('soporte_inmediato_usd')}\n"
                              f"Resistencia: ${tech_analysis.get('resistencia_inmediata_usd')}\n"
                              f"Tendencia: {tech_analysis.get('tendencia')}\n"
                              f"Confianza: {tech_analysis.get('confianza_tecnica_porcentaje')}%\n"
                              f"Resumen: {tech_analysis.get('resumen_tecnico')}")
    else:
        send_telegram_message("⚠️ Error en el análisis técnico.")
        return

    # --- Paso 3: Análisis Fundamental y Unificado (con búsqueda) ---
    fundamental_analysis = get_fundamental_and_unified_analysis(tech_analysis)
    if fundamental_analysis:
        unified_message = (
            f"🌍 [ANÁLISIS FUNDAMENTAL UNIFICADO]\n"
            f"• Fear & Greed: {fundamental_analysis.get('fear_greed_index')}\n"
            f"• ETFs Spot: ${fundamental_analysis.get('etf_flow_usd')}M\n"
            f"• Macro: {fundamental_analysis.get('tendencias_macro')}\n"
            f"• Regulaciones: {fundamental_analysis.get('regulaciones_clave')}\n"
            f"• Ballenas: {fundamental_analysis.get('comportamiento_ballenas')}\n"
            f"• Otras Noticias: {fundamental_analysis.get('otras_noticias')}\n\n"
            f"🎯 VEREDICTO UNIFICADO: {fundamental_analysis.get('veredicto_unificado')}\n"
            f"🔒 Confianza: {fundamental_analysis.get('confianza_unificada_porcentaje')}%\n"
            f"💡 Explicación: {fundamental_analysis.get('explicacion_unificada')}"
        )
        send_telegram_message(unified_message)
    else:
        send_telegram_message("⚠️ Error en el análisis fundamental.")

# --- Configurar el scheduler (ejecutar cada 5 horas) ---
import schedule
import time

# Ejecutar una vez al iniciar
run_analysis_cycle()

# Programar cada 5 horas
schedule.every(5).hours.do(run_analysis_cycle)

while True:
    schedule.run_pending()
    time.sleep(60)
