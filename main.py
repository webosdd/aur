import os
import time
import io
import json
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timezone
from openai import OpenAI
from pybit.unified_trading import HTTP
from PIL import Image
import base64
import schedule

# ================= CONFIGURACIÓN =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("analisis_btc.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Variables de entorno
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

if not all([BYBIT_API_KEY, BYBIT_API_SECRET, OPENROUTER_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID]):
    raise ValueError("❌ Faltan variables de entorno")

SYMBOL = "BTCUSDT"
INTERVALO_VELAS = "D"      # Velas diarias
LIMITE_VELAS = 120         # Últimas 120 velas para gráfico

# Cliente OpenRouter (Gemini con visión + internet)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://railway.app",
        "X-Title": "BTC Dual Analyst"
    }
)
MODELO = "google/gemini-3.1-flash-image-preview:online"   # con internet

# Cliente Bybit (solo para datos)
bybit = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# ================= TELEGRAM =================
def telegram_mensaje(texto):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def telegram_enviar_imagen(fig, caption=""):
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': buf}
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption}
        requests.post(url, files=files, data=data, timeout=15)
        buf.close()
        plt.close(fig)
    except Exception as e:
        logger.error(f"Error enviando imagen: {e}")

# ================= OBTENER DATOS BYBIT =================
def obtener_velas_diarias(limit=100):
    try:
        resp = bybit.get_kline(category="spot", symbol=SYMBOL, interval=INTERVALO_VELAS, limit=limit)
        if resp.get("retCode") != 0:
            return pd.DataFrame()
        lista = resp["result"]["list"][::-1]
        df = pd.DataFrame(lista, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error velas: {e}")
        return pd.DataFrame()

# ================= INDICADORES =================
def calcular_indicadores(df):
    if df.empty:
        return df
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    return df.dropna()

def soporte_resistencia_historicos(df):
    if len(df) < 20:
        return 0, 0
    soporte = df['low'].rolling(20).min().iloc[-1]
    resistencia = df['high'].rolling(20).max().iloc[-1]
    return soporte, resistencia

def tendencia_regresion(df, ventana=80):
    y = df['close'].values[-ventana:]
    x = np.arange(len(y))
    slope, intercept, r, _, _ = linregress(x, y)
    return slope, intercept

# ================= GRÁFICO COMPLETO =================
def generar_grafico(df, soporte, resistencia, slope, intercept):
    df_plot = df.tail(LIMITE_VELAS).copy()
    if df_plot.empty:
        return None

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.arange(len(df_plot))

    # Velas japonesas
    for i in range(len(df_plot)):
        o = df_plot['open'].iloc[i]
        h = df_plot['high'].iloc[i]
        l = df_plot['low'].iloc[i]
        c = df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, abs(c-o)+0.1, color=color, alpha=0.9))

    # Soporte / Resistencia
    ax.axhline(soporte, color='cyan', ls='--', lw=2, label=f'Soporte ({soporte:.0f})')
    ax.axhline(resistencia, color='magenta', ls='--', lw=2, label=f'Resistencia ({resistencia:.0f})')

    # EMAs
    ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    ax.plot(x, df_plot['ema50'], 'orange', lw=2, label='EMA50')

    # Línea de tendencia (regresión lineal)
    y_trend = intercept + slope * x
    ax.plot(x, y_trend, 'white', linestyle='-.', lw=2, label=f'Tendencia (pendiente {slope:.2f})')

    # RSI en eje Y derecho
    ax2 = ax.twinx()
    ax2.plot(x, df_plot['rsi'], color='orange', lw=1.5, alpha=0.7, label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI', color='orange')

    # MACD en otro eje Y
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x, df_plot['macd'], 'blue', lw=1, label='MACD')
    ax3.plot(x, df_plot['signal'], 'red', lw=1, label='Señal')
    ax3.bar(x, df_plot['histogram'], color='gray', alpha=0.3, width=0.8)
    ax3.set_ylabel('MACD', color='blue')

    # Puntos de liquidez (máximos/mínimos recientes)
    # Detectamos picos locales en highs y lows
    highs = df_plot['high'].values
    lows = df_plot['low'].values
    for i in range(2, len(df_plot)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            ax.scatter(x[i], highs[i], s=100, c='white', marker='v', edgecolors='red', linewidth=2, zorder=5)
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            ax.scatter(x[i], lows[i], s=100, c='white', marker='^', edgecolors='green', linewidth=2, zorder=5)

    ax.set_title(f"{SYMBOL} - Análisis Técnico Diario", color='white')
    ax.set_xlabel('Días atrás', color='white')
    ax.set_ylabel('Precio (USDT)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.legend(loc='upper left')
    fig.tight_layout()
    return fig

def fig_a_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')
    if img.width > 1200:
        ratio = 1200 / img.width
        new_size = (1200, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    out_buf = io.BytesIO()
    img.save(out_buf, format='JPEG', quality=85)
    b64 = base64.b64encode(out_buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return f"data:image/jpeg;base64,{b64}"

# ================= LLAMADAS A LA IA =================
def analisis_tecnico_por_imagen(img_base64):
    prompt = """
    Eres un analista técnico experto. Analiza el gráfico diario de BTC/USDT y extrae la siguiente información. Responde SOLO con un JSON válido, sin texto adicional.

    {
      "precio_actual_usd": float,
      "soporte_inmediato_usd": float,
      "resistencia_inmediata_usd": float,
      "tendencia": "Alcista | Bajista | Lateral",
      "confianza_tecnica": 0-100,
      "momentum": "Alcista | Bajista | Neutro",
      "puntos_liquidez": ["zona de liquidez 1", "zona 2"],
      "analisis_ema": "posición del precio vs EMAs",
      "analisis_rsi": "nivel RSI e interpretación",
      "analisis_macd": "estado del MACD",
      "resumen_tecnico": "breve resumen"
    }
    """
    try:
        response = client.chat.completions.create(
            model=MODELO,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_base64}}
                ]
            }],
            temperature=0.2,
            max_tokens=800
        )
        content = response.choices[0].message.content
        content = content.strip().strip('```json').strip('```').strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error IA técnica: {e}")
        return None

def analisis_fundamental_y_unificado(tecnico_json):
    prompt = f"""
    Actúa como analista fundamental. Usa tu capacidad de navegación por internet para buscar información actualizada (fecha: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}) sobre Bitcoin:

    1. Crypto Fear & Greed Index (valor numérico y texto)
    2. Flujo neto de ETFs de Bitcoin al contado (en millones USD, con signo + o -)
    3. Noticias macroeconómicas relevantes (Fed, tasas, dólar)
    4. Novedades regulatorias (ej. Ley GENIUS, noticias recientes)
    5. Comportamiento de ballenas (acumulación o distribución)
    6. Otras 2-3 noticias importantes

    Luego integra esta información con el siguiente análisis técnico:
    {json.dumps(tecnico_json, indent=2)}

    Emite un veredicto unificado para las próximas horas/días (alcista, bajista o neutral), con confianza (0-100) y explicación.
    Responde SOLO con JSON, sin texto extra:

    {{
      "fear_greed_index": "ej. 22 - Miedo extremo",
      "etf_flow_usd": -733,
      "tendencias_macro": "texto breve",
      "regulaciones_clave": "texto breve",
      "comportamiento_ballenas": "texto breve",
      "otras_noticias": ["noticia1", "noticia2"],
      "resumen_fundamental": "resumen ejecutivo",
      "veredicto_unificado": "Alcista | Bajista | Neutral",
      "confianza_unificada": 0-100,
      "explicacion_unificada": "explicación detallada"
    }}
    """
    try:
        response = client.chat.completions.create(
            model=MODELO,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        content = response.choices[0].message.content
        content = content.strip().strip('```json').strip('```').strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error IA fundamental: {e}")
        return None

# ================= CICLO PRINCIPAL (CADA 5 HORAS) =================
def ciclo_analisis():
    logger.info("🔄 Iniciando ciclo de análisis dual (técnico + fundamental)")
    telegram_mensaje("🔍 Iniciando análisis dual de BTC (gráfico + noticias) - cada 5h")

    # 1. Obtener datos y generar gráfico
    df = obtener_velas_diarias(limit=LIMITE_VELAS)
    if df.empty:
        logger.error("No se obtuvieron velas")
        telegram_mensaje("❌ Error: No se pudieron obtener velas de Bybit")
        return

    df = calcular_indicadores(df)
    sop_hist, res_hist = soporte_resistencia_historicos(df)
    slope, intercept = tendencia_regresion(df)

    fig = generar_grafico(df, sop_hist, res_hist, slope, intercept)
    if fig is None:
        logger.error("Error generando gráfico")
        return

    # Enviar gráfico a Telegram
    telegram_enviar_imagen(fig, caption=f"📊 Análisis Técnico {SYMBOL} - {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")

    # Convertir gráfico a base64 para IA
    img_b64 = fig_a_base64(fig)

    # 2. Análisis técnico por IA
    tech = analisis_tecnico_por_imagen(img_b64)
    if not tech:
        telegram_mensaje("⚠️ Falló el análisis técnico con IA")
        return

    # Enviar resumen técnico a Telegram
    msg_tech = (
        f"📈 ANÁLISIS TÉCNICO\n"
        f"Precio: ${tech.get('precio_actual_usd')}\n"
        f"Soporte: ${tech.get('soporte_inmediato_usd')}\n"
        f"Resistencia: ${tech.get('resistencia_inmediata_usd')}\n"
        f"Tendencia: {tech.get('tendencia')}\n"
        f"Confianza: {tech.get('confianza_tecnica')}%\n"
        f"Resumen: {tech.get('resumen_tecnico')}"
    )
    telegram_mensaje(msg_tech)

    # 3. Análisis fundamental + unificado (con internet)
    fund = analisis_fundamental_y_unificado(tech)
    if not fund:
        telegram_mensaje("⚠️ Falló el análisis fundamental con internet")
        return

    msg_fund = (
        f"🌍 ANÁLISIS FUNDAMENTAL + UNIFICADO\n"
        f"Fear & Greed: {fund.get('fear_greed_index')}\n"
        f"ETFs: {fund.get('etf_flow_usd')}M USD\n"
        f"Macro: {fund.get('tendencias_macro')}\n"
        f"Regulaciones: {fund.get('regulaciones_clave')}\n"
        f"Ballenas: {fund.get('comportamiento_ballenas')}\n"
        f"Noticias: {', '.join(fund.get('otras_noticias', []))}\n\n"
        f"🎯 VEREDICTO UNIFICADO: {fund.get('veredicto_unificado')}\n"
        f"🔒 Confianza: {fund.get('confianza_unificada')}%\n"
        f"💡 Explicación: {fund.get('explicacion_unificada')}"
    )
    telegram_mensaje(msg_fund)

    logger.info("✅ Ciclo de análisis completado")

# ================= MAIN =================
if __name__ == "__main__":
    logger.info("🚀 Bot de Análisis Dual BTC iniciado")
    telegram_mensaje("🚀 Bot de Análisis Dual BTC activo - Análisis cada 5 horas")

    # Ejecutar una vez al inicio
    ciclo_analisis()

    # Programar cada 5 horas
    schedule.every(5).hours.do(ciclo_analisis)

    while True:
        schedule.run_pending()
        time.sleep(60)
