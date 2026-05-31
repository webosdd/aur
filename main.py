import os
import time
import io
import base64
import json
import logging
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
from datetime import datetime, timezone
from openai import OpenAI
import matplotlib.patches as patches

# ================= CONFIGURACIÓN =================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Parámetros del bot
SYMBOL = "BTCUSDT"
INTERVAL = "1"               # velas de 1 minuto
CICLO_SEGUNDOS = 180         # 3 minutos
GRAFICO_VELAS_LIMIT = 120    # últimas 120 velas en el gráfico

# Riesgo y simulación (paper trading)
RISK_PER_TRADE = 0.01        # 1% del balance por trade
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PNL_GLOBAL = 0.0
PAPER_POSICION_ACTIVA = None   # 'Buy' o 'Sell'
PAPER_PRECIO_ENTRADA = 0.0
PAPER_SL = 0.0
PAPER_TP1 = 0.0
PAPER_TP2 = 0.0
PAPER_TP3 = 0.0
PAPER_SIZE_BTC = 0.0
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0

# Variables de entorno (Railway)
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ Faltan BYBIT_API_KEY o BYBIT_API_SECRET")
if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
    raise Exception("❌ Faltan TELEGRAM_TOKEN o TELEGRAM_CHAT_ID")
if not SILICONFLOW_API_KEY:
    raise Exception("❌ Faltan SILICONFLOW_API_KEY")

# Cliente SiliconFlow
client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1")
MODELO_VISION = "Tongyi-MAI/Z-Image-Turbo"   # Modelo de visión

# ================= FUNCIONES BYBIT =================
def obtener_velas(limit=300):
    url = "https://api.bybit.com/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code != 200:
        raise Exception(f"Error Bybit: {r.text}")
    data = r.json()
    if data['retCode'] != 0:
        raise Exception(f"Bybit error: {data}")
    lista = data['result']['list'][::-1]
    df = pd.DataFrame(lista, columns=['time','open','high','low','close','volume','turnover'])
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df

# ================= INDICADORES TÉCNICOS =================
def calcular_indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    # Soporte/resistencia dinámica (ventana 50)
    df['soporte'] = df['low'].rolling(50).min()
    df['resistencia'] = df['high'].rolling(50).max()
    return df

# ================= GRÁFICO ENRIQUECIDO =================
def generar_grafico(df, decision_ia=None, entry_price=None, sl=None, tp1=None, tp2=None, tp3=None):
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    if df_plot.empty:
        return None

    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(df_plot))

    # Velas japonesas
    for i, (idx, row) in enumerate(df_plot.iterrows()):
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        color = '#00ff00' if c >= o else '#ff0000'
        ax.plot([i, i], [l, h], color=color, linewidth=1.5)
        rect = patches.Rectangle((i-0.35, min(o,c)), 0.7, abs(c-o), facecolor=color, alpha=0.9)
        ax.add_patch(rect)

    # EMAs
    ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA 20')
    ax.plot(x, df_plot['ema50'].values, color='orange', linewidth=2, label='EMA 50')

    # Soporte y resistencia (últimos valores)
    soporte = df_plot['soporte'].iloc[-1]
    resistencia = df_plot['resistencia'].iloc[-1]
    ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f'Soporte ({soporte:.0f})')
    ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f'Resistencia ({resistencia:.0f})')

    # Línea de tendencia
    y_vals = df_plot['close'].values
    slope, intercept = linregress(np.arange(len(y_vals)), y_vals)[:2]
    trend_line = intercept + slope * x
    ax.plot(x, trend_line, color='white', linestyle='-.', linewidth=2, label=f'Tendencia (pendiente {slope:.2f})')

    # Niveles de entrada, SL, TP si existen
    if entry_price and decision_ia:
        ax.scatter(x[-1], entry_price, s=200, marker='*' if decision_ia=='Buy' else 's',
                   color='lime' if decision_ia=='Buy' else 'red', edgecolors='black', zorder=5,
                   label=f'Entrada {decision_ia} @ {entry_price:.0f}')
        if sl:
            ax.axhline(sl, color='red', linestyle=':', linewidth=1.5, label=f'SL @ {sl:.0f}')
        if tp1:
            ax.axhline(tp1, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label=f'TP1 @ {tp1:.0f}')
        if tp2:
            ax.axhline(tp2, color='green', linestyle=':', linewidth=1.5, alpha=0.5, label=f'TP2 @ {tp2:.0f}')
        if tp3:
            ax.axhline(tp3, color='green', linestyle=':', linewidth=1.5, alpha=0.3, label=f'TP3 @ {tp3:.0f}')

    # Texto informativo
    info = f"BTCUSDT - 1m\nÚltimo precio: {df_plot['close'].iloc[-1]:.0f}\nSoporte: {soporte:.0f}\nResistencia: {resistencia:.0f}\nEMA20: {df_plot['ema20'].iloc[-1]:.0f}\nEMA50: {df_plot['ema50'].iloc[-1]:.0f}"
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(facecolor='black', alpha=0.7))

    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.set_xlabel('Velas', color='white')
    ax.set_ylabel('Precio USDT', color='white')
    ax.set_title(f"{SYMBOL} - Análisis Técnico (IA Visión)", color='white')
    ax.legend(loc='upper left', facecolor='black', edgecolor='white')
    plt.tight_layout()
    return fig

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"

# ================= ANÁLISIS CON IA DE VISIÓN =================
def analizar_con_ia_vision(img_base64, precio_actual):
    prompt = f"""
Eres un trader profesional experto en análisis técnico. Analiza el gráfico de BTCUSDT en velas de 1 minuto.
El gráfico muestra velas japonesas (verde alcista, rojo bajista), EMA20 (amarilla), EMA50 (naranja),
soporte (cyan), resistencia (magenta), y línea de tendencia blanca.

Precio actual: {precio_actual:.2f} USDT.

Basándote en los patrones de velas, la tendencia, la posición del precio respecto a EMAs,
soportes y resistencias, decide si es momento de COMPRAR (Buy), VENDER (Sell) o MANTENER (Hold).

IMPORTANTE: Los soportes rotos se convierten en resistencias y viceversa. Las EMAs también actúan como soporte/resistencia dinámica.

Si decides Buy o Sell, proporciona:
- Precio de entrada (normalmente el precio actual o ligeramente ajustado).
- Stop Loss (SL) en USDT, basado en ATR o estructura.
- Take Profit 1, 2 y 3 (TP1, TP2, TP3) en USDT, basados en niveles de resistencia/soporte o múltiplos de ATR.
- Una breve explicación de por qué tomas esa decisión (máximo 100 palabras).

Si decides Hold, simplemente explica por qué no hay oportunidad.

Responde ÚNICAMENTE con un JSON válido, sin texto adicional. Usa esta estructura:
{{
  "decision": "Buy | Sell | Hold",
  "entrada": número (solo si no es Hold),
  "sl": número (solo si no es Hold),
  "tp1": número,
  "tp2": número,
  "tp3": número,
  "razon": "texto explicativo"
}}
"""
    try:
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_base64}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            temperature=0.2,
            max_tokens=800
        )
        content = response.choices[0].message.content
        content = content.strip().strip('```json').strip('```').strip()
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error IA visión: {e}")
        return None

# ================= PAPER TRADING =================
def abrir_posicion(decision, entrada, sl, tp1, tp2, tp3, balance_actual):
    global PAPER_POSICION_ACTIVA, PAPER_PRECIO_ENTRADA, PAPER_SL, PAPER_TP1, PAPER_TP2, PAPER_TP3, PAPER_SIZE_BTC
    if PAPER_POSICION_ACTIVA is not None:
        return False
    riesgo_usd = balance_actual * RISK_PER_TRADE
    distancia_sl = abs(entrada - sl)
    if distancia_sl == 0:
        return False
    size_btc = riesgo_usd / distancia_sl
    PAPER_POSICION_ACTIVA = decision
    PAPER_PRECIO_ENTRADA = entrada
    PAPER_SL = sl
    PAPER_TP1 = tp1
    PAPER_TP2 = tp2
    PAPER_TP3 = tp3
    PAPER_SIZE_BTC = size_btc
    return True

def verificar_cierre(precio_actual):
    global PAPER_POSICION_ACTIVA, PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_WIN, PAPER_LOSS, PAPER_TRADES_TOTALES
    if PAPER_POSICION_ACTIVA is None:
        return None
    cerrar = False
    motivo = None
    pnl = 0.0
    if PAPER_POSICION_ACTIVA == 'Buy':
        if precio_actual <= PAPER_SL:
            cerrar = True
            motivo = "SL"
            pnl = (precio_actual - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC
        elif precio_actual >= PAPER_TP1:
            cerrar = True
            motivo = "TP1"
            pnl = (PAPER_TP1 - PAPER_PRECIO_ENTRADA) * PAPER_SIZE_BTC
    elif PAPER_POSICION_ACTIVA == 'Sell':
        if precio_actual >= PAPER_SL:
            cerrar = True
            motivo = "SL"
            pnl = (PAPER_PRECIO_ENTRADA - precio_actual) * PAPER_SIZE_BTC
        elif precio_actual <= PAPER_TP1:
            cerrar = True
            motivo = "TP1"
            pnl = (PAPER_PRECIO_ENTRADA - PAPER_TP1) * PAPER_SIZE_BTC
    if cerrar:
        PAPER_BALANCE += pnl
        PAPER_PNL_GLOBAL += pnl
        PAPER_TRADES_TOTALES += 1
        if pnl > 0:
            PAPER_WIN += 1
        else:
            PAPER_LOSS += 1
        resultado = {
            "decision": PAPER_POSICION_ACTIVA,
            "entrada": PAPER_PRECIO_ENTRADA,
            "salida": precio_actual if motivo=="SL" else (PAPER_TP1 if PAPER_POSICION_ACTIVA=='Buy' else PAPER_TP1),
            "pnl": pnl,
            "motivo": motivo,
            "balance": PAPER_BALANCE
        }
        PAPER_POSICION_ACTIVA = None
        return resultado
    return None

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

# ================= CICLO PRINCIPAL =================
def ciclo_trading():
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_POSICION_ACTIVA
    logger.info("🔄 Iniciando ciclo de análisis con IA visión...")
    try:
        df = obtener_velas(limit=300)
        df = calcular_indicadores(df)
        if df.empty:
            raise Exception("No hay datos")
        precio_actual = df['close'].iloc[-1]

        # Generar gráfico base
        fig = generar_grafico(df)
        if fig is None:
            raise Exception("Error generando gráfico")
        img_b64 = fig_to_base64(fig)

        # Consultar IA
        decision_ia = analizar_con_ia_vision(img_b64, precio_actual)
        if not decision_ia:
            telegram_mensaje("⚠️ No se pudo obtener decisión de la IA.")
            return

        decision = decision_ia.get('decision')
        razon = decision_ia.get('razon', 'Sin explicación')
        entrada = decision_ia.get('entrada')
        sl = decision_ia.get('sl')
        tp1 = decision_ia.get('tp1')
        tp2 = decision_ia.get('tp2')
        tp3 = decision_ia.get('tp3')

        # Verificar cierre de posición existente
        cierre = verificar_cierre(precio_actual)
        if cierre:
            msg_cierre = (
                f"🔒 *CIERRE DE POSICIÓN*\n"
                f"Operación: {cierre['decision']}\n"
                f"Entrada: {cierre['entrada']:.0f}\n"
                f"Salida: {cierre['salida']:.0f} ({cierre['motivo']})\n"
                f"PnL: {cierre['pnl']:.2f} USD\n"
                f"Balance: {cierre['balance']:.2f} USD\n"
                f"PnL Global: {PAPER_PNL_GLOBAL:.2f} USD"
            )
            telegram_mensaje(msg_cierre)

        # Nueva señal si no hay posición activa
        if PAPER_POSICION_ACTIVA is None and decision in ['Buy', 'Sell']:
            if abrir_posicion(decision, entrada, sl, tp1, tp2, tp3, PAPER_BALANCE):
                fig2 = generar_grafico(df, decision, entrada, sl, tp1, tp2, tp3)
                if fig2:
                    caption = (
                        f"🚀 *SEÑAL DE {decision.upper()}*\n"
                        f"📊 *Entrada:* {entrada:.0f}\n"
                        f"🛑 *SL:* {sl:.0f}\n"
                        f"🎯 *TP1:* {tp1:.0f} | *TP2:* {tp2:.0f} | *TP3:* {tp3:.0f}\n"
                        f"💰 *Balance:* {PAPER_BALANCE:.2f} USD\n"
                        f"📈 *PnL Global:* {PAPER_PNL_GLOBAL:.2f} USD\n\n"
                        f"🧠 *Razón:* {razon}"
                    )
                    telegram_enviar_imagen(fig2, caption)
                else:
                    telegram_mensaje(f"Señal {decision} pero error en gráfico.")
            else:
                telegram_mensaje(f"❌ No se pudo abrir posición {decision}.")
        else:
            # Solo enviar gráfico de monitoreo
            if PAPER_POSICION_ACTIVA is None:
                caption = f"📊 *Análisis Técnico*\nDecisión IA: *HOLD*\nRazón: {razon}\nBalance: {PAPER_BALANCE:.2f} USD\nPnL Global: {PAPER_PNL_GLOBAL:.2f} USD"
            else:
                caption = f"📊 *Posición activa: {PAPER_POSICION_ACTIVA}*\nBalance: {PAPER_BALANCE:.2f} USD\nPnL Global: {PAPER_PNL_GLOBAL:.2f} USD"
            telegram_enviar_imagen(fig, caption)

    except Exception as e:
        error_msg = f"🚨 Error en ciclo: {e}"
        logger.error(error_msg)
        telegram_mensaje(error_msg)

# ================= MAIN =================
if __name__ == "__main__":
    telegram_mensaje("🤖 Bot de Trading con IA Visión (Tongyi-MAI/Z-Image-Turbo) iniciado. Ciclo cada 3 minutos.")
    while True:
        ciclo_trading()
        time.sleep(CICLO_SEGUNDOS)
