# ======================================================
# BOT TRADING BTC/USDT – IA COMPLETAMENTE AUTÓNOMA
# - La IA decide dirección, SL, TP1, TP2, TP3
# - Cierres parciales: 50% en TP1, 25% en TP2, 25% trailing o SL en TP3
# - Comisiones simuladas (0.1%)
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
RISK_PER_TRADE = 0.0025       # 0.25% del balance por operación (solo como referencial, la IA decide SL real)
MAX_SIMULTANEOUS_POSITIONS = 3
MAX_DRAWDOWN_PERCENT = 20.0
PAUSE_ON_DRAWDOWN_SECONDS = 3600
SLEEP_SECONDS = 60

# Comisión simulada (0.1% por operación)
COMMISSION_RATE = 0.001

# Trailing step por defecto si la IA no lo especifica (para el último tramo)
DEFAULT_TRAILING_STEP_PERCENT = 0.0005  # 0.05%

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
# GENERAR GRÁFICO (multimodal)
# ======================================================

def generar_grafico_base64(df, decision="HOLD", soporte=None, resistencia=None, 
                           slope=None, intercept=None, razones=None, 
                           precio_entrada=None, trade_id=None, niveles=None):
    try:
        df_plot = df.copy().tail(120)
        if df_plot.empty:
            return None, None
        fig, ax = plt.subplots(figsize=(14, 7))
        x = np.arange(len(df_plot))
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
        if soporte:
            ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        if resistencia:
            ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")
        if 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')
        if slope is not None:
            y_plot = df_plot['close'].values
            x_plot = np.arange(len(y_plot))
            slope_plot, intercept_plot = np.polyfit(x_plot, y_plot, 1)
            tendencia_linea = intercept_plot + slope_plot * x_plot
            ax.plot(x_plot, tendencia_linea, color='#FFA500', linewidth=2, linestyle='-', label=f"Tendencia slope {slope_plot:.4f}")
        if precio_entrada and trade_id:
            entrada_index = len(df_plot) - 1
            if decision == 'BUY':
                ax.scatter(entrada_index, precio_entrada, s=200, marker='^', color='lime', edgecolors='black', label='Entrada BUY')
                ax.axvline(entrada_index, color='lime', linestyle=':', linewidth=2)
            elif decision == 'SELL':
                ax.scatter(entrada_index, precio_entrada, s=200, marker='v', color='red', edgecolors='black', label='Entrada SELL')
                ax.axvline(entrada_index, color='red', linestyle=':', linewidth=2)
            # Dibujar niveles de TP y SL si se proporcionan
            if niveles:
                if 'sl' in niveles:
                    ax.axhline(niveles['sl'], color='red', linestyle='--', linewidth=1.5, alpha=0.7, label=f"SL {niveles['sl']:.2f}")
                if 'tp1' in niveles:
                    ax.axhline(niveles['tp1'], color='lime', linestyle='--', linewidth=1.5, alpha=0.7, label=f"TP1 {niveles['tp1']:.2f}")
                if 'tp2' in niveles:
                    ax.axhline(niveles['tp2'], color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f"TP2 {niveles['tp2']:.2f}")
                if 'tp3' in niveles:
                    ax.axhline(niveles['tp3'], color='yellow', linestyle='--', linewidth=1.5, alpha=0.7, label=f"TP3 {niveles['tp3']:.2f}")
            texto = f"Trade #{trade_id}\n{decision}\nPrecio: {precio_entrada:.2f}\nBalance: {PAPER_BALANCE:.2f} USD\nPnL Global: {PAPER_PNL_GLOBAL:.4f}\nRazones:\n" + "\n".join(razones[:4]) if razones else ""
            ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'), color='white')
        ax.set_title(f"{SYMBOL} - {INTERVAL}m")
        ax.set_xlabel("Velas")
        ax.set_ylabel("Precio (USDT)")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='lower left')
        step = max(1, int(len(df_plot)/10))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([t.strftime('%H:%M') for t in df_plot.index[::step]], rotation=45)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        return img_base64, fig
    except Exception as e:
        print(f"Error gráfico: {e}")
        return None, None

# ======================================================
# IA MULTIMODAL: decide entrada, SL, TP1, TP2, TP3
# ======================================================

def obtener_decision_ia_multimodal(df, soporte, resistencia, slope, tendencia):
    img_base64, fig = generar_grafico_base64(df, soporte=soporte, resistencia=resistencia, slope=slope)
    if not img_base64:
        return "HOLD", ["No se pudo generar gráfico"], None, None, None, None, None
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
Eres un trader profesional de BTC/USDT en timeframe de {INTERVAL} minutos.
Te proporciono la imagen del gráfico de velas japonesas con EMA20, niveles de soporte/resistencia y línea de tendencia.
Datos numéricos actuales:
- Precio: {precio:.2f}
- EMA20: {ema20:.2f}
- ATR: {atr:.2f}
- RSI: {rsi:.1f}
- MACD: {macd:.2f} | Señal: {macd_signal:.2f}
- Tendencia pendiente: {slope:.5f} ({tendencia})
- Soporte: {soporte:.2f}
- Resistencia: {resistencia:.2f}
- Volumen: {volumen:.0f} | Media 20: {vol_media:.0f}

Tu tarea es decidir si COMPRAR (BUY), VENDER (SELL) o NO HACER NADA (HOLD).
Además, si decides entrar, debes definir los niveles óptimos de:
- Stop Loss (sl_price)
- Take Profit 1 (tp1_price) -> se cerrará el 50% de la posición aquí
- Take Profit 2 (tp2_price) -> se cerrará el 25% de la posición aquí
- Take Profit 3 (tp3_price) -> el 25% restante se gestionará con trailing stop (usando este nivel como referencia inicial para trailing, o como SL fijo).

Devuelve ÚNICAMENTE un JSON con este formato exacto (sin texto adicional):
{{
    "decision": "BUY/SELL/HOLD",
    "razones": ["razón1", "razón2", ...],
    "sl_price": 12345.67,
    "tp1_price": 12400.00,
    "tp2_price": 12450.00,
    "tp3_price": 12500.00
}}
Si es HOLD, los campos de precio pueden ser nulos.
Los precios deben ser realistas y estar en la dirección del trade (para BUY: tp3 > tp2 > tp1 > entry > sl; para SELL: sl > entry > tp1 > tp2 > tp3).
Eres autónomo: usa tu análisis visual y técnico para elegir los mejores niveles.
"""
    try:
        response = client.chat.completions.create(
            model=MODELO_IA,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]}
            ],
            temperature=0.4,
            max_tokens=600
        )
        contenido = response.choices[0].message.content
        print(f"IA respuesta: {contenido[:300]}...")
        if not contenido:
            raise Exception("Vacío")
        json_match = re.search(r'\{.*\}', contenido, re.DOTALL)
        data = json.loads(json_match.group(0)) if json_match else json.loads(contenido)
        decision = data.get("decision", "HOLD").upper()
        razones = data.get("razones", ["Sin razones"])
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
        sl = data.get("sl_price")
        tp1 = data.get("tp1_price")
        tp2 = data.get("tp2_price")
        tp3 = data.get("tp3_price")
        # Validación básica: si es BUY/SELL, deben existir niveles coherentes
        if decision != "HOLD" and (sl is None or tp1 is None or tp2 is None or tp3 is None):
            print("IA no devolvió todos los niveles, se ignora entrada")
            return "HOLD", ["Faltan niveles de salida"], img_base64, fig, None, None, None, None
        return decision, razones, img_base64, fig, sl, tp1, tp2, tp3
    except Exception as e:
        print(f"Error IA: {e}")
        return "HOLD", [f"Error: {e}"], img_base64, fig, None, None, None, None

# ======================================================
# PAPER TRADING CON CIERRES PARCIALES (50%, 25%, 25%)
# ======================================================

def paper_abrir_posicion(decision, precio, razones, tiempo, sl_price, tp1_price, tp2_price, tp3_price):
    global PAPER_BALANCE, PAPER_POSICIONES_ACTIVAS, PAPER_TRADES_TOTALES, PAPER_NEXT_TRADE_ID
    if len(PAPER_POSICIONES_ACTIVAS) >= MAX_SIMULTANEOUS_POSITIONS:
        return None

    # Validar que los niveles tengan sentido
    if decision == "BUY":
        if not (sl_price < precio < tp1_price < tp2_price < tp3_price):
            print(f"Niveles inválidos para BUY: sl={sl_price}, entry={precio}, tp1={tp1_price}, tp2={tp2_price}, tp3={tp3_price}")
            return None
    else:  # SELL
        if not (sl_price > precio > tp1_price > tp2_price > tp3_price):
            print(f"Niveles inválidos para SELL: sl={sl_price}, entry={precio}, tp1={tp1_price}, tp2={tp2_price}, tp3={tp3_price}")
            return None

    # Calcular tamaño de posición basado en riesgo fijo (0.25% del balance) y distancia al SL
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    distancia_sl = abs(precio - sl_price)
    if distancia_sl == 0:
        return None
    size_btc_total = riesgo_usd / distancia_sl
    size_usd_total = size_btc_total * precio

    # Distribución: 50% en TP1, 25% en TP2, 25% en TP3
    size_btc_tp1 = size_btc_total * 0.5
    size_btc_tp2 = size_btc_total * 0.25
    size_btc_tp3 = size_btc_total * 0.25

    trade_id = PAPER_NEXT_TRADE_ID
    PAPER_NEXT_TRADE_ID += 1
    PAPER_TRADES_TOTALES += 1

    pos = {
        "id": trade_id,
        "decision": decision,
        "entry_price": precio,
        "entry_time": tiempo,
        "sl_price": sl_price,
        "tp1_price": tp1_price,
        "tp2_price": tp2_price,
        "tp3_price": tp3_price,
        "size_btc_tp1": size_btc_tp1,
        "size_btc_tp2": size_btc_tp2,
        "size_btc_tp3": size_btc_tp3,
        "size_usd_total": size_usd_total,
        "razones": razones,
        "tp1_hit": False,
        "tp2_hit": False,
        "tp3_hit": False,       # el último tramo se manejará con trailing
        "trailing_sl": sl_price,
        "best_price": precio,
    }
    PAPER_POSICIONES_ACTIVAS.append(pos)
    return pos

def paper_actualizar_trailing(pos, precio_actual):
    """Actualiza el trailing stop para el último tramo (25%) usando un step fijo pequeño."""
    if not pos["tp2_hit"]:   # solo después de TP2 se activa trailing para TP3
        return pos["trailing_sl"]
    step_abs = pos["entry_price"] * DEFAULT_TRAILING_STEP_PERCENT
    if pos["decision"] == "BUY":
        if precio_actual > pos["best_price"]:
            pos["best_price"] = precio_actual
            new_sl = precio_actual - step_abs
            if new_sl > pos["trailing_sl"]:
                pos["trailing_sl"] = new_sl
    else:
        if precio_actual < pos["best_price"]:
            pos["best_price"] = precio_actual
            new_sl = precio_actual + step_abs
            if new_sl < pos["trailing_sl"]:
                pos["trailing_sl"] = new_sl
    return pos["trailing_sl"]

def paper_revisar_sl_tp(precio_actual, tiempo_actual):
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_WIN, PAPER_LOSS
    global PAPER_POSICIONES_ACTIVAS, PAPER_TRADES_CERRADOS
    global PAPER_PEAK_BALANCE, PAPER_DRAWDOWN_PAUSED_UNTIL

    cerradas = []
    for pos in PAPER_POSICIONES_ACTIVAS[:]:
        # 1) Verificar TP1 (50%)
        if not pos["tp1_hit"]:
            if (pos["decision"] == "BUY" and precio_actual >= pos["tp1_price"]) or \
               (pos["decision"] == "SELL" and precio_actual <= pos["tp1_price"]):
                pnl_tp1 = 0
                if pos["decision"] == "BUY":
                    pnl_tp1 = (pos["tp1_price"] - pos["entry_price"]) * pos["size_btc_tp1"]
                else:
                    pnl_tp1 = (pos["entry_price"] - pos["tp1_price"]) * pos["size_btc_tp1"]
                pnl_tp1 -= abs(pnl_tp1) * COMMISSION_RATE
                PAPER_BALANCE += pnl_tp1
                PAPER_PNL_GLOBAL += pnl_tp1
                if pnl_tp1 > 0:
                    PAPER_WIN += 0.5
                else:
                    PAPER_LOSS += 0.5
                cerradas.append({
                    "trade_id": pos["id"],
                    "type": "TP1 (50%)",
                    "pnl": pnl_tp1,
                    "price": pos["tp1_price"],
                    "balance_after": PAPER_BALANCE
                })
                pos["tp1_hit"] = True
                # Actualizar mejor precio para trailing después de TP1
                pos["best_price"] = pos["tp1_price"] if pos["decision"] == "BUY" else pos["tp1_price"]
                pos["trailing_sl"] = pos["sl_price"]  # reiniciar trailing

        # 2) Verificar TP2 (25%)
        if pos["tp1_hit"] and not pos["tp2_hit"]:
            if (pos["decision"] == "BUY" and precio_actual >= pos["tp2_price"]) or \
               (pos["decision"] == "SELL" and precio_actual <= pos["tp2_price"]):
                pnl_tp2 = 0
                if pos["decision"] == "BUY":
                    pnl_tp2 = (pos["tp2_price"] - pos["entry_price"]) * pos["size_btc_tp2"]
                else:
                    pnl_tp2 = (pos["entry_price"] - pos["tp2_price"]) * pos["size_btc_tp2"]
                pnl_tp2 -= abs(pnl_tp2) * COMMISSION_RATE
                PAPER_BALANCE += pnl_tp2
                PAPER_PNL_GLOBAL += pnl_tp2
                if pnl_tp2 > 0:
                    PAPER_WIN += 0.25
                else:
                    PAPER_LOSS += 0.25
                cerradas.append({
                    "trade_id": pos["id"],
                    "type": "TP2 (25%)",
                    "pnl": pnl_tp2,
                    "price": pos["tp2_price"],
                    "balance_after": PAPER_BALANCE
                })
                pos["tp2_hit"] = True
                # Actualizar mejor precio para trailing
                pos["best_price"] = pos["tp2_price"] if pos["decision"] == "BUY" else pos["tp2_price"]
                # Trailing se activa después de TP2 para el último tramo

        # 3) Gestión del último tramo (25%): trailing stop o TP3 fijo
        if pos["tp2_hit"] and not pos["tp3_hit"]:
            # Actualizar trailing
            nuevo_sl = paper_actualizar_trailing(pos, precio_actual)
            # Verificar si se alcanza el trailing SL o el TP3
            # Primero, si alcanza TP3, cerramos todo el tramo restante con ganancia
            if (pos["decision"] == "BUY" and precio_actual >= pos["tp3_price"]) or \
               (pos["decision"] == "SELL" and precio_actual <= pos["tp3_price"]):
                pnl_tp3 = 0
                if pos["decision"] == "BUY":
                    pnl_tp3 = (pos["tp3_price"] - pos["entry_price"]) * pos["size_btc_tp3"]
                else:
                    pnl_tp3 = (pos["entry_price"] - pos["tp3_price"]) * pos["size_btc_tp3"]
                pnl_tp3 -= abs(pnl_tp3) * COMMISSION_RATE
                PAPER_BALANCE += pnl_tp3
                PAPER_PNL_GLOBAL += pnl_tp3
                if pnl_tp3 > 0:
                    PAPER_WIN += 0.25
                else:
                    PAPER_LOSS += 0.25
                cerradas.append({
                    "trade_id": pos["id"],
                    "type": "TP3 (25%)",
                    "pnl": pnl_tp3,
                    "price": pos["tp3_price"],
                    "balance_after": PAPER_BALANCE
                })
                pos["tp3_hit"] = True
                PAPER_POSICIONES_ACTIVAS.remove(pos)
                continue
            # Si no se alcanzó TP3, verificar si se toca el trailing SL
            elif (pos["decision"] == "BUY" and precio_actual <= nuevo_sl) or \
                 (pos["decision"] == "SELL" and precio_actual >= nuevo_sl):
                pnl_trail = 0
                if pos["decision"] == "BUY":
                    pnl_trail = (nuevo_sl - pos["entry_price"]) * pos["size_btc_tp3"]
                else:
                    pnl_trail = (pos["entry_price"] - nuevo_sl) * pos["size_btc_tp3"]
                pnl_trail -= abs(pnl_trail) * COMMISSION_RATE
                PAPER_BALANCE += pnl_trail
                PAPER_PNL_GLOBAL += pnl_trail
                if pnl_trail > 0:
                    PAPER_WIN += 0.25
                else:
                    PAPER_LOSS += 0.25
                cerradas.append({
                    "trade_id": pos["id"],
                    "type": "Trailing SL (25%)",
                    "pnl": pnl_trail,
                    "price": nuevo_sl,
                    "balance_after": PAPER_BALANCE
                })
                pos["tp3_hit"] = True
                PAPER_POSICIONES_ACTIVAS.remove(pos)
                continue

        # 4) Si no se ha alcanzado ningún TP, verificar SL inicial (solo si no se ha cerrado nada aún)
        if not pos["tp1_hit"]:
            if (pos["decision"] == "BUY" and precio_actual <= pos["sl_price"]) or \
               (pos["decision"] == "SELL" and precio_actual >= pos["sl_price"]):
                # Stop loss golpeado antes de cualquier TP: cerrar toda la posición
                size_total = pos["size_btc_tp1"] + pos["size_btc_tp2"] + pos["size_btc_tp3"]
                if pos["decision"] == "BUY":
                    pnl_total = (pos["sl_price"] - pos["entry_price"]) * size_total
                else:
                    pnl_total = (pos["entry_price"] - pos["sl_price"]) * size_total
                pnl_total -= abs(pnl_total) * COMMISSION_RATE
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
                    "price": pos["sl_price"],
                    "balance_after": PAPER_BALANCE
                })
                PAPER_POSICIONES_ACTIVAS.remove(pos)
                continue

    # Actualizar drawdown
    if PAPER_BALANCE > PAPER_PEAK_BALANCE:
        PAPER_PEAK_BALANCE = PAPER_BALANCE
    drawdown_pct = (PAPER_PEAK_BALANCE - PAPER_BALANCE) / PAPER_PEAK_BALANCE * 100 if PAPER_PEAK_BALANCE > 0 else 0
    pausa = False
    if drawdown_pct >= MAX_DRAWDOWN_PERCENT and PAPER_DRAWDOWN_PAUSED_UNTIL is None:
        PAPER_DRAWDOWN_PAUSED_UNTIL = tiempo_actual + pd.Timedelta(seconds=PAUSE_ON_DRAWDOWN_SECONDS)
        pausa = True
    return cerradas, pausa

# ======================================================
# TELEGRAM Y LOG
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
# BUCLE PRINCIPAL
# ======================================================

def run_bot():
    global PAPER_PEAK_BALANCE, PAPER_DRAWDOWN_PAUSED_UNTIL
    telegram_mensaje("🤖 BOT BTC/USDT AUTÓNOMO INICIADO (IA decide SL, TP1, TP2, TP3)")
    while True:
        try:
            ahora = datetime.now(timezone.utc)
            if PAPER_DRAWDOWN_PAUSED_UNTIL and ahora < PAPER_DRAWDOWN_PAUSED_UNTIL:
                restante = (PAPER_DRAWDOWN_PAUSED_UNTIL - ahora).total_seconds()
                if restante > 0:
                    print(f"⏸️ Pausa drawdown. Reanuda en {restante:.0f}s")
                    time.sleep(min(60, restante))
                    continue
                else:
                    PAPER_DRAWDOWN_PAUSED_UNTIL = None
                    telegram_mensaje("✅ Pausa drawdown finalizada.")
                    PAPER_PEAK_BALANCE = PAPER_BALANCE

            df = obtener_velas(limit=200)
            df = calcular_indicadores(df)

            soporte, resistencia = detectar_soportes_resistencias(df)
            slope, intercept, tendencia = detectar_tendencia(df)

            decision, razones, _, fig, sl, tp1, tp2, tp3 = obtener_decision_ia_multimodal(df, soporte, resistencia, slope, tendencia)
            log_estado(df, tendencia, slope, soporte, resistencia, decision, razones)

            precio_actual = df['close'].iloc[-1]
            tiempo_actual = df.index[-1]

            cerradas, pausa = paper_revisar_sl_tp(precio_actual, tiempo_actual)
            if pausa:
                telegram_mensaje(f"⚠️ DRAWDOWN {MAX_DRAWDOWN_PERCENT}% - BOT PAUSADO 1H")
                continue

            for c in cerradas:
                resultado = "✅ GANADOR" if c["pnl"] > 0 else "❌ PERDEDOR"
                msg = f"📌 *CIERRE Trade #{c['trade_id']}* - {c['type']}\n{resultado}\n💰 PnL: {c['pnl']:+.4f} USD\n💵 Balance: {c['balance_after']:.2f} USD\n📊 W/L: {PAPER_WIN:.1f}/{PAPER_LOSS:.1f}"
                telegram_mensaje(msg)

            if PAPER_BALANCE > PAPER_PEAK_BALANCE:
                PAPER_PEAK_BALANCE = PAPER_BALANCE

            if decision in ("BUY", "SELL") and len(PAPER_POSICIONES_ACTIVAS) < MAX_SIMULTANEOUS_POSITIONS:
                nueva_pos = paper_abrir_posicion(decision, precio_actual, razones, tiempo_actual, sl, tp1, tp2, tp3)
                if nueva_pos:
                    # Mostrar niveles al usuario
                    msg_entrada = (
                        f"🚀 *NUEVA ENTRADA PAPER - Trade #{nueva_pos['id']}*\n"
                        f"📌 Dirección: {decision}\n"
                        f"💲 Entry: {precio_actual:.2f}\n"
                        f"🛑 SL: {nueva_pos['sl_price']:.2f}\n"
                        f"🎯 TP1 (50%): {nueva_pos['tp1_price']:.2f}\n"
                        f"🎯 TP2 (25%): {nueva_pos['tp2_price']:.2f}\n"
                        f"🎯 TP3 (25%): {nueva_pos['tp3_price']:.2f} (con trailing)\n"
                        f"💰 Riesgo asumido: {RISK_PER_TRADE*100:.2f}% del balance\n"
                        f"📦 Tamaño: {nueva_pos['size_usd_total']:.2f} USD\n"
                        f"🧠 Setup IA:\n" + "\n".join(razones)
                    )
                    telegram_mensaje(msg_entrada)
                    # Gráfico con niveles marcados
                    niveles = {
                        "sl": nueva_pos['sl_price'],
                        "tp1": nueva_pos['tp1_price'],
                        "tp2": nueva_pos['tp2_price'],
                        "tp3": nueva_pos['tp3_price']
                    }
                    _, fig_con_marca = generar_grafico_base64(
                        df, decision=decision, soporte=soporte, resistencia=resistencia,
                        slope=slope, intercept=intercept, razones=razones,
                        precio_entrada=precio_actual, trade_id=nueva_pos['id'], niveles=niveles
                    )
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
