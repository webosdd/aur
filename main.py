# ======================================================
# BOT TRADING V90.2 BYBIT – IA GEMINI 3.1 FLASH (MULTIMODAL)
# - La IA recibe el gráfico de velas (imagen) + datos numéricos
# - Decisiones autónomas basadas en análisis visual y técnico
# ======================================================

import os
import time
import io
import base64
import hmac
import hashlib
import json
import re
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from datetime import datetime, timezone

plt.rcParams['figure.figsize'] = (12, 6)

# ======================================================
# CONFIGURACIÓN
# ======================================================

SYMBOL = "BTCUSDT"
INTERVAL = "5"                # 5 minutos
RISK_PER_TRADE = 0.0025       # 0.25% del balance por operación
MAX_SIMULTANEOUS_POSITIONS = 3
MAX_DRAWDOWN_PERCENT = 20.0
PAUSE_ON_DRAWDOWN_SECONDS = 3600
SLEEP_SECONDS = 60

# Trailing stop para la segunda mitad
TRAILING_STEP = 0.5

# Papel (simulación)
PAPER_BALANCE_INICIAL = 100.0
PAPER_BALANCE = PAPER_BALANCE_INICIAL
PAPER_PEAK_BALANCE = PAPER_BALANCE_INICIAL
PAPER_DRAWDOWN_PAUSED_UNTIL = None
PAPER_PNL_GLOBAL = 0.0
PAPER_POSICIONES_ACTIVAS = []
PAPER_TRADES_CERRADOS = []
PAPER_WIN = 0
PAPER_LOSS = 0
PAPER_TRADES_TOTALES = 0
PAPER_NEXT_TRADE_ID = 1

# ======================================================
# CREDENCIALES
# ======================================================

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise Exception("❌ OPENROUTER_API_KEY no configurada")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ BYBIT_API_KEY o BYBIT_API_SECRET no configuradas")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "BTC Trading Bot"}
)

MODELO_IA = "google/gemini-3.1-flash-image-preview"

# ======================================================
# FUNCIONES BYBIT
# ======================================================

BASE_URL = "https://api.bybit.com"

def obtener_velas(limit=300):
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=20)
    if not r.text:
        raise Exception("Respuesta vacía de Bybit")
    try:
        data_json = r.json()
    except Exception:
        raise Exception(f"Bybit devolvió respuesta no-JSON: {r.text}")
    if not isinstance(data_json, dict):
        raise Exception(f"Bybit devolvió JSON no dict: {type(data_json)}")
    if "retCode" in data_json and data_json["retCode"] != 0:
        raise Exception(f"Bybit Error retCode={data_json.get('retCode')} retMsg={data_json.get('retMsg')}")
    if "result" not in data_json or not isinstance(data_json["result"], dict):
        raise Exception(f"Respuesta inválida Bybit: {data_json}")
    if "list" not in data_json["result"] or not isinstance(data_json["result"]["list"], list):
        raise Exception(f"Bybit result sin 'list' o no es lista: {data_json['result']}")
    data = data_json["result"]["list"][::-1]
    if len(data) == 0:
        raise Exception("Bybit devolvió lista vacía de velas")
    df = pd.DataFrame(data, columns=['time','open','high','low','close','volume','turnover'])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df

# ======================================================
# INDICADORES (incluye MACD)
# ======================================================

def calcular_indicadores(df):
    df['ema20'] = df['close'].ewm(span=20).mean()
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    return df.dropna()

def detectar_soportes_resistencias(df, ventana=50):
    soporte = df['low'].rolling(ventana).min().iloc[-1]
    resistencia = df['high'].rolling(ventana).max().iloc[-1]
    return soporte, resistencia

def detectar_tendencia(df, ventana=80):
    y = df['close'].values[-ventana:]
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    if slope > 0.02:
        direccion = 'ALCISTA'
    elif slope < -0.02:
        direccion = 'BAJISTA'
    else:
        direccion = 'LATERAL'
    return slope, intercept, direccion

# ======================================================
# GENERAR GRÁFICO DE VELAS (para IA y Telegram)
# ======================================================

def generar_grafico_base64(df, decision="HOLD", soporte=None, resistencia=None, 
                           slope=None, intercept=None, razones=None, 
                           precio_entrada=None, trade_id=None):
    """Genera gráfico y devuelve imagen en base64 (para la IA) y figura (para Telegram)"""
    try:
        df_plot = df.copy().tail(120)
        if df_plot.empty:
            return None, None
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(df_plot))
        # Velas
        for i, (idx, row) in enumerate(df_plot.iterrows()):
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            color = 'green' if c >= o else 'red'
            ax.vlines(i, l, h, color=color, linewidth=1)
            cuerpo_y = min(o, c)
            cuerpo_h = abs(c - o)
            if cuerpo_h == 0:
                cuerpo_h = 0.0001
            rect = plt.Rectangle((i - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color, alpha=0.9)
            ax.add_patch(rect)
        # Niveles
        if soporte:
            ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        if resistencia:
            ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")
        # EMA20
        if 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')
        # Tendencia
        if slope is not None and intercept is not None:
            y_plot = df_plot['close'].values
            x_plot = np.arange(len(y_plot))
            slope_plot, intercept_plot = np.polyfit(x_plot, y_plot, 1)
            tendencia_linea = intercept_plot + slope_plot * x_plot
            ax.plot(x_plot, tendencia_linea, color='#FFA500', linewidth=2, linestyle='-', label=f"Tendencia slope {slope_plot:.4f}")
        # Marcador de entrada
        if precio_entrada and trade_id:
            entrada_index = len(df_plot) - 1
            if decision == 'BUY':
                ax.scatter(entrada_index, precio_entrada, s=200, marker='^', color='lime', edgecolors='black', label='Entrada BUY')
                ax.axvline(entrada_index, color='lime', linestyle=':', linewidth=2)
            elif decision == 'SELL':
                ax.scatter(entrada_index, precio_entrada, s=200, marker='v', color='red', edgecolors='black', label='Entrada SELL')
                ax.axvline(entrada_index, color='red', linestyle=':', linewidth=2)
            texto = f"Trade #{trade_id}\n{decision}\nPrecio: {precio_entrada:.2f}\nBalance: {PAPER_BALANCE:.2f} USD\nPnL Global: {PAPER_PNL_GLOBAL:.4f}\nRazones:\n" + "\n".join(razones[:4]) if razones else ""
            ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'), color='white')
        ax.set_title(f"{SYMBOL} - {INTERVAL}m")
        ax.set_xlabel("Velas")
        ax.set_ylabel("Precio")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='lower left')
        step = max(1, int(len(df_plot)/10))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([t.strftime('%H:%M') for t in df_plot.index[::step]], rotation=45)
        plt.tight_layout()
        # Convertir a base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        # Devolver también la figura para Telegram
        return img_base64, fig
    except Exception as e:
        print(f"Error generando gráfico: {e}")
        return None, None

# ======================================================
# IA MULTIMODAL: recibe imagen + datos numéricos
# ======================================================

def obtener_decision_ia_multimodal(df, soporte, resistencia, slope, tendencia):
    """
    Genera el gráfico, lo convierte a base64 y lo envía a Gemini junto con el prompt.
    Retorna (decision, razones, imagen_base64, figura) para luego poder enviar a Telegram.
    """
    # Primero generamos la imagen (sin marcador de entrada, solo el contexto actual)
    img_base64, fig = generar_grafico_base64(df, soporte=soporte, resistencia=resistencia, slope=slope, intercept=None)
    if not img_base64:
        return "HOLD", ["No se pudo generar gráfico"], None, None
    
    # Datos numéricos adicionales
    ultimo = df.iloc[-1]
    precio = ultimo['close']
    ema20 = ultimo['ema20']
    atr = ultimo['atr']
    rsi = ultimo['rsi'] if 'rsi' in ultimo else 50
    macd = ultimo['macd'] if 'macd' in ultimo else 0
    macd_signal = ultimo['macd_signal'] if 'macd_signal' in ultimo else 0
    volumen = ultimo['volume'] if 'volume' in ultimo else 0
    vol_media = df['volume'].rolling(20).mean().iloc[-1] if 'volume' in df else 0
    
    prompt = f"""
Eres un trader profesional y experimentado en BTC/USDT, operando en timeframe de {INTERVAL} minutos.
Te voy a proporcionar una imagen del gráfico de velas japonesas con EMA20, niveles de soporte/resistencia, y una línea de tendencia (regresión lineal). Además, te doy los siguientes datos numéricos actuales:

- Precio actual: {precio:.2f}
- EMA20: {ema20:.2f}
- ATR (volatilidad): {atr:.2f}
- RSI: {rsi:.1f}
- MACD: {macd:.2f} | Señal: {macd_signal:.2f}
- Tendencia lineal (pendiente): {slope:.5f} ({tendencia})
- Soporte dinámico (mín 50 velas): {soporte:.2f}
- Resistencia dinámica (máx 50 velas): {resistencia:.2f}
- Volumen actual: {volumen:.0f} | Media 20 velas: {vol_media:.0f}

Analiza la imagen y los datos. Eres completamente autónomo: no tienes reglas fijas. Puedes basarte en patrones de velas (martillo, engulfing, doji, etc.), mechas, estructura de mercado, caza de liquidez, divergencias, comportamiento del volumen, etc. Decide si COMPRAR (BUY), VENDER (SELL) o NO HACER NADA (HOLD).

Tu decisión debe ser fundamentada con 2-4 razones claras y específicas, refiriéndote tanto a lo que ves en la imagen como a los números.

Devuelve ÚNICAMENTE un JSON con este formato exacto, sin texto adicional:
{{"decision": "BUY/SELL/HOLD", "razones": ["razón específica 1", "razón específica 2", ...]}}
"""
    try:
        response = client.chat.completions.create(
            model=MODELO_IA,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                    ]
                }
            ],
            temperature=0.4,
            max_tokens=500
        )
        contenido = response.choices[0].message.content
        print(f"Respuesta IA cruda: {contenido[:300]}...")
        if not contenido:
            raise Exception("Respuesta vacía")
        json_match = re.search(r'\{.*\}', contenido, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
        else:
            data = json.loads(contenido)
        decision = data.get("decision", "HOLD").upper()
        razones = data.get("razones", ["Sin razones"])
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
        return decision, razones, img_base64, fig
    except Exception as e:
        print(f"Error IA multimodal: {e}")
        return "HOLD", [f"Error: {e}"], img_base64, fig

# ======================================================
# PAPER TRADING (idéntico al anterior)
# ======================================================

def paper_abrir_posicion(decision, precio, atr, razones, tiempo):
    global PAPER_BALANCE, PAPER_POSICIONES_ACTIVAS, PAPER_TRADES_TOTALES, PAPER_NEXT_TRADE_ID
    if len(PAPER_POSICIONES_ACTIVAS) >= MAX_SIMULTANEOUS_POSITIONS:
        return None
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    if decision == "BUY":
        sl_inicial = precio - atr
        tp1 = precio + (atr * 2)
    else:
        sl_inicial = precio + atr
        tp1 = precio - (atr * 2)
    distancia_sl = abs(precio - sl_inicial)
    if distancia_sl == 0:
        return None
    size_btc_total = riesgo_usd / distancia_sl
    size_usd_total = size_btc_total * precio
    size_btc_tp1 = size_btc_total / 2
    size_btc_trail = size_btc_total - size_btc_tp1
    trade_id = PAPER_NEXT_TRADE_ID
    PAPER_NEXT_TRADE_ID += 1
    PAPER_TRADES_TOTALES += 1
    pos = {
        "id": trade_id,
        "decision": decision,
        "entry_price": precio,
        "entry_time": tiempo,
        "entry_atr": atr,
        "sl_initial": sl_inicial,
        "tp1_price": tp1,
        "size_btc_tp1": size_btc_tp1,
        "size_btc_trail": size_btc_trail,
        "size_usd_total": size_usd_total,
        "razones": razones,
        "tp1_hit": False,
        "trailing_sl": sl_inicial,
        "best_price": precio,
    }
    PAPER_POSICIONES_ACTIVAS.append(pos)
    return pos

def paper_calcular_pnl_posicion(pos, precio_actual):
    if pos["decision"] == "BUY":
        return (precio_actual - pos["entry_price"]) * pos["size_btc_trail"]  # solo para trailing, pero se usa por separado
    else:
        return (pos["entry_price"] - precio_actual) * pos["size_btc_trail"]

def paper_actualizar_trailing(pos, precio_actual):
    if not pos["tp1_hit"]:
        return pos["trailing_sl"]
    if pos["decision"] == "BUY":
        if precio_actual > pos["best_price"]:
            pos["best_price"] = precio_actual
            new_sl = precio_actual - (pos["entry_atr"] * TRAILING_STEP)
            if new_sl > pos["trailing_sl"]:
                pos["trailing_sl"] = new_sl
    else:
        if precio_actual < pos["best_price"]:
            pos["best_price"] = precio_actual
            new_sl = precio_actual + (pos["entry_atr"] * TRAILING_STEP)
            if new_sl < pos["trailing_sl"]:
                pos["trailing_sl"] = new_sl
    return pos["trailing_sl"]

def paper_revisar_sl_tp(precio_actual, tiempo_actual):
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_WIN, PAPER_LOSS
    global PAPER_POSICIONES_ACTIVAS, PAPER_TRADES_CERRADOS
    global PAPER_PEAK_BALANCE, PAPER_DRAWDOWN_PAUSED_UNTIL

    cerradas = []
    for pos in PAPER_POSICIONES_ACTIVAS[:]:
        if not pos["tp1_hit"]:
            if (pos["decision"] == "BUY" and precio_actual >= pos["tp1_price"]) or \
               (pos["decision"] == "SELL" and precio_actual <= pos["tp1_price"]):
                pnl_tp1 = 0
                if pos["decision"] == "BUY":
                    pnl_tp1 = (pos["tp1_price"] - pos["entry_price"]) * pos["size_btc_tp1"]
                else:
                    pnl_tp1 = (pos["entry_price"] - pos["tp1_price"]) * pos["size_btc_tp1"]
                pnl_tp1 -= abs(pnl_tp1) * 0.001
                PAPER_BALANCE += pnl_tp1
                PAPER_PNL_GLOBAL += pnl_tp1
                if pnl_tp1 > 0:
                    PAPER_WIN += 0.5
                else:
                    PAPER_LOSS += 0.5
                cerradas.append({
                    "trade_id": pos["id"],
                    "type": "TP1 parcial (50%)",
                    "pnl": pnl_tp1,
                    "price": pos["tp1_price"],
                    "balance_after": PAPER_BALANCE
                })
                pos["tp1_hit"] = True
                pos["best_price"] = pos["tp1_price"] if pos["decision"] == "BUY" else pos["tp1_price"]
                pos["trailing_sl"] = pos["sl_initial"]
        if pos["tp1_hit"]:
            nuevo_sl = paper_actualizar_trailing(pos, precio_actual)
            if (pos["decision"] == "BUY" and precio_actual <= nuevo_sl) or \
               (pos["decision"] == "SELL" and precio_actual >= nuevo_sl):
                pnl_trail = 0
                if pos["decision"] == "BUY":
                    pnl_trail = (nuevo_sl - pos["entry_price"]) * pos["size_btc_trail"]
                else:
                    pnl_trail = (pos["entry_price"] - nuevo_sl) * pos["size_btc_trail"]
                pnl_trail -= abs(pnl_trail) * 0.001
                PAPER_BALANCE += pnl_trail
                PAPER_PNL_GLOBAL += pnl_trail
                if pnl_trail > 0:
                    PAPER_WIN += 0.5
                else:
                    PAPER_LOSS += 0.5
                cerradas.append({
                    "trade_id": pos["id"],
                    "type": f"Trailing SL ({nuevo_sl:.2f})",
                    "pnl": pnl_trail,
                    "price": nuevo_sl,
                    "balance_after": PAPER_BALANCE
                })
                PAPER_POSICIONES_ACTIVAS.remove(pos)
                continue
        else:
            if (pos["decision"] == "BUY" and precio_actual <= pos["sl_initial"]) or \
               (pos["decision"] == "SELL" and precio_actual >= pos["sl_initial"]):
                size_total = pos["size_btc_tp1"] + pos["size_btc_trail"]
                if pos["decision"] == "BUY":
                    pnl_total = (pos["sl_initial"] - pos["entry_price"]) * size_total
                else:
                    pnl_total = (pos["entry_price"] - pos["sl_initial"]) * size_total
                pnl_total -= abs(pnl_total) * 0.001
                PAPER_BALANCE += pnl_total
                PAPER_PNL_GLOBAL += pnl_total
                if pnl_total > 0:
                    PAPER_WIN += 1
                else:
                    PAPER_LOSS += 1
                cerradas.append({
                    "trade_id": pos["id"],
                    "type": "SL completo",
                    "pnl": pnl_total,
                    "price": pos["sl_initial"],
                    "balance_after": PAPER_BALANCE
                })
                PAPER_POSICIONES_ACTIVAS.remove(pos)
                continue

    if PAPER_BALANCE > PAPER_PEAK_BALANCE:
        PAPER_PEAK_BALANCE = PAPER_BALANCE
    drawdown_pct = (PAPER_PEAK_BALANCE - PAPER_BALANCE) / PAPER_PEAK_BALANCE * 100 if PAPER_PEAK_BALANCE > 0 else 0
    pausa = False
    if drawdown_pct >= MAX_DRAWDOWN_PERCENT and PAPER_DRAWDOWN_PAUSED_UNTIL is None:
        PAPER_DRAWDOWN_PAUSED_UNTIL = tiempo_actual + pd.Timedelta(seconds=PAUSE_ON_DRAWDOWN_SECONDS)
        pausa = True
    return cerradas, pausa

# ======================================================
# TELEGRAM
# ======================================================

def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception:
        pass

def telegram_grafico(fig):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        requests.post(url, files={'photo': buf}, data={'chat_id': TELEGRAM_CHAT_ID}, timeout=15)
        buf.close()
    except Exception:
        pass

def log_estado(df, tendencia, slope, soporte, resistencia, decision, razones):
    ahora = datetime.now(timezone.utc)
    precio = df['close'].iloc[-1]
    print("="*100)
    print(f"🕒 {ahora} | BTC: {precio:.2f}")
    print(f"📐 Tendencia: {tendencia} | Slope: {slope:.5f}")
    print(f"🧱 Soporte: {soporte:.2f} | Resistencia: {resistencia:.2f}")
    print(f"🎯 Decisión IA: {decision if decision else 'NO TRADE'}")
    print(f"🧠 Razones: {', '.join(razones)}")
    print(f"💵 Balance Paper: {PAPER_BALANCE:.2f} USD | PnL Global: {PAPER_PNL_GLOBAL:.4f}")
    print(f"📊 Posiciones activas: {len(PAPER_POSICIONES_ACTIVAS)}/{MAX_SIMULTANEOUS_POSITIONS}")
    peak = max(PAPER_PEAK_BALANCE, PAPER_BALANCE)
    dd = (peak - PAPER_BALANCE) / peak * 100 if peak > 0 else 0
    print(f"📉 Drawdown actual: {dd:.2f}% (máx permitido {MAX_DRAWDOWN_PERCENT}%)")
    if PAPER_DRAWDOWN_PAUSED_UNTIL:
        print(f"⏸️ PAUSADO hasta {PAPER_DRAWDOWN_PAUSED_UNTIL}")
    print("="*100)

# ======================================================
# LOOP PRINCIPAL
# ======================================================

def run_bot():
    global PAPER_PEAK_BALANCE, PAPER_DRAWDOWN_PAUSED_UNTIL
    telegram_mensaje("🤖 BOT MULTIMODAL INICIADO (Gemini 3.1 Flash, ve el gráfico)")
    while True:
        try:
            ahora = datetime.now(timezone.utc)
            if PAPER_DRAWDOWN_PAUSED_UNTIL and ahora < PAPER_DRAWDOWN_PAUSED_UNTIL:
                restante = (PAPER_DRAWDOWN_PAUSED_UNTIL - ahora).total_seconds()
                if restante > 0:
                    print(f"⏸️ Pausa por drawdown. Reanudando en {restante:.0f} segundos...")
                    time.sleep(min(60, restante))
                    continue
                else:
                    PAPER_DRAWDOWN_PAUSED_UNTIL = None
                    telegram_mensaje("✅ Pausa por drawdown finalizada. Bot reanudado.")
                    PAPER_PEAK_BALANCE = PAPER_BALANCE

            df = obtener_velas(limit=200)
            df = calcular_indicadores(df)

            soporte, resistencia = detectar_soportes_resistencias(df)
            slope, intercept, tendencia = detectar_tendencia(df)

            # Obtener decisión de IA multimodal (ve el gráfico)
            decision, razones, _, fig = obtener_decision_ia_multimodal(df, soporte, resistencia, slope, tendencia)
            log_estado(df, tendencia, slope, soporte, resistencia, decision, razones)

            precio_actual = df['close'].iloc[-1]
            tiempo_actual = df.index[-1]

            # Revisar cierres
            cerradas, pausa_activada = paper_revisar_sl_tp(precio_actual, tiempo_actual)
            if pausa_activada:
                telegram_mensaje(f"⚠️ DRAWDOWN SUPERADO ({MAX_DRAWDOWN_PERCENT}%) - BOT PAUSADO 1 HORA")
                continue

            for c in cerradas:
                resultado = "✅ GANADOR" if c["pnl"] > 0 else "❌ PERDEDOR"
                msg_cierre = (
                    f"📌 *CIERRE Trade #{c['trade_id']}* - {c['type']}\n"
                    f"{resultado}\n"
                    f"💰 PnL: {c['pnl']:+.4f} USD\n"
                    f"💵 Balance tras cierre: {c['balance_after']:.2f} USD\n"
                    f"📊 Win/Loss acumulado: {PAPER_WIN:.0f}/{PAPER_LOSS:.0f}"
                )
                telegram_mensaje(msg_cierre)

            if PAPER_BALANCE > PAPER_PEAK_BALANCE:
                PAPER_PEAK_BALANCE = PAPER_BALANCE

            # Abrir nueva posición si IA lo indica
            if decision in ("BUY", "SELL") and len(PAPER_POSICIONES_ACTIVAS) < MAX_SIMULTANEOUS_POSITIONS:
                atr_actual = df['atr'].iloc[-1]
                nueva_pos = paper_abrir_posicion(decision, precio_actual, atr_actual, razones, tiempo_actual)
                if nueva_pos:
                    # Generar gráfico con marcador de entrada para Telegram
                    _, fig_con_marca = generar_grafico_base64(
                        df, decision=decision, soporte=soporte, resistencia=resistencia,
                        slope=slope, intercept=intercept, razones=razones,
                        precio_entrada=precio_actual, trade_id=nueva_pos['id']
                    )
                    msg_entrada = (
                        f"🚀 *NUEVA ENTRADA PAPER - Trade #{nueva_pos['id']}*\n"
                        f"📌 Dirección: {decision}\n"
                        f"💲 Precio Entry: {precio_actual:.2f}\n"
                        f"🛑 Stop Loss inicial: {nueva_pos['sl_initial']:.2f}\n"
                        f"🎯 TP1 (50%): {nueva_pos['tp1_price']:.2f}\n"
                        f"🔄 Trailing para 50% restante: step {TRAILING_STEP}×ATR\n"
                        f"💰 Riesgo: {RISK_PER_TRADE*100:.2f}% del balance\n"
                        f"📦 Tamaño total: {nueva_pos['size_usd_total']:.2f} USD\n"
                        f"🧠 Setup IA:\n" + "\n".join(razones)
                    )
                    telegram_mensaje(msg_entrada)
                    if fig_con_marca:
                        telegram_grafico(fig_con_marca)
                        plt.close(fig_con_marca)
            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"🚨 ERROR: {e}")
            telegram_mensaje(f"🚨 ERROR BOT: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
