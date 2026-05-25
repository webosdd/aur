# ======================================================
# BOT TRADING V90.2 BYBIT – IA GEMINI 3.1 FLASH (SCALPING 5m)
# - TP/SL basados en porcentaje fijo (ajustable)
# - Trailing step pequeño desde el inicio
# - TP1 cerca de soporte/resistencia si aplica
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
# CONFIGURACIÓN (AJUSTABLE PARA SCALPING)
# ======================================================

SYMBOL = "BTCUSDT"
INTERVAL = "5"                # 5 minutos
RISK_PER_TRADE = 0.0025       # 0.25% del balance por operación
MAX_SIMULTANEOUS_POSITIONS = 3
MAX_DRAWDOWN_PERCENT = 20.0
PAUSE_ON_DRAWDOWN_SECONDS = 3600
SLEEP_SECONDS = 60

# Parámetros de scalping (porcentajes)
STOP_LOSS_PERCENT = 0.0015      # 0.15% de pérdida máxima
TAKE_PROFIT_PERCENT = 0.0025    # 0.25% de ganancia objetivo (para la mitad)
TRAILING_STEP_PERCENT = 0.0005  # 0.05% de trailing step

# Si se activa, el TP se ajustará al nivel de resistencia/soporte más cercano
# (si la distancia al nivel es menor que TAKE_PROFIT_PERCENT * 1.5)
USE_LEVELS_FOR_TP = True

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
# FUNCIONES BYBIT (sin cambios)
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
# GENERAR GRÁFICO (multimodal) – igual que antes
# ======================================================

def generar_grafico_base64(df, decision="HOLD", soporte=None, resistencia=None, 
                           slope=None, intercept=None, razones=None, 
                           precio_entrada=None, trade_id=None):
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
# IA MULTIMODAL (sin cambios)
# ======================================================

def obtener_decision_ia_multimodal(df, soporte, resistencia, slope, tendencia):
    img_base64, fig = generar_grafico_base64(df, soporte=soporte, resistencia=resistencia, slope=slope)
    if not img_base64:
        return "HOLD", ["No se pudo generar gráfico"], None, None
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
Eres un trader profesional experimentado en BTC/USDT en timeframe de {INTERVAL} minutos (scalping).
Te proporciono la imagen del gráfico de velas con EMA20, niveles de soporte/resistencia y línea de tendencia.
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

Analiza visualmente el gráfico y los números. Decide BUY, SELL o HOLD.
Eres autónomo, sin reglas fijas. Puedes usar patrones de velas, mechas, estructura, divergencias, etc.
Devuelve JSON: {{"decision": "BUY/SELL/HOLD", "razones": ["razón1", "razón2", ...]}}
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
            max_tokens=500
        )
        contenido = response.choices[0].message.content
        print(f"IA respuesta: {contenido[:200]}...")
        if not contenido:
            raise Exception("Vacío")
        json_match = re.search(r'\{.*\}', contenido, re.DOTALL)
        data = json.loads(json_match.group(0)) if json_match else json.loads(contenido)
        decision = data.get("decision", "HOLD").upper()
        razones = data.get("razones", ["Sin razones"])
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
        return decision, razones, img_base64, fig
    except Exception as e:
        print(f"Error IA: {e}")
        return "HOLD", [f"Error: {e}"], img_base64, fig

# ======================================================
# PAPER TRADING CON SCALPING (TP/SL por porcentaje, trailing desde inicio)
# ======================================================

def calcular_tp_sl_porcentual(precio, decision, soporte=None, resistencia=None):
    """
    Calcula SL y TP basados en porcentajes fijos.
    Si USE_LEVELS_FOR_TP está activo y el nivel de soporte/resistencia está cerca,
    se usa ese nivel como TP (si es más rentable que el porcentaje).
    """
    sl_price = precio * (1 - STOP_LOSS_PERCENT) if decision == "BUY" else precio * (1 + STOP_LOSS_PERCENT)
    tp_price = precio * (1 + TAKE_PROFIT_PERCENT) if decision == "BUY" else precio * (1 - TAKE_PROFIT_PERCENT)
    
    if USE_LEVELS_FOR_TP:
        if decision == "BUY" and resistencia:
            # Para compra, el TP podría ser la resistencia
            distancia_res = (resistencia - precio) / precio
            if 0 < distancia_res < TAKE_PROFIT_PERCENT * 1.5:  # si la resistencia está más cerca que el TP fijo
                tp_price = resistencia
        elif decision == "SELL" and soporte:
            distancia_sop = (precio - soporte) / precio
            if 0 < distancia_sop < TAKE_PROFIT_PERCENT * 1.5:
                tp_price = soporte
    return sl_price, tp_price

def paper_abrir_posicion(decision, precio, razones, tiempo, soporte=None, resistencia=None):
    global PAPER_BALANCE, PAPER_POSICIONES_ACTIVAS, PAPER_TRADES_TOTALES, PAPER_NEXT_TRADE_ID
    if len(PAPER_POSICIONES_ACTIVAS) >= MAX_SIMULTANEOUS_POSITIONS:
        return None

    sl_price, tp_price = calcular_tp_sl_porcentual(precio, decision, soporte, resistencia)
    # El riesgo real en USD es la distancia al SL
    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    distancia_sl = abs(precio - sl_price)
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
        "sl_initial": sl_price,
        "tp1_price": tp_price,
        "size_btc_tp1": size_btc_tp1,
        "size_btc_trail": size_btc_trail,
        "size_usd_total": size_usd_total,
        "razones": razones,
        "tp1_hit": False,
        "trailing_sl": sl_price,          # trailing stop inicial
        "best_price": precio,             # mejor precio alcanzado
        "trailing_active": True,          # trailing activo desde el inicio
    }
    PAPER_POSICIONES_ACTIVAS.append(pos)
    return pos

def paper_actualizar_trailing(pos, precio_actual):
    """Actualiza el trailing stop para toda la posición (desde el inicio)"""
    if not pos["trailing_active"]:
        return pos["trailing_sl"]
    step_abs = pos["entry_price"] * TRAILING_STEP_PERCENT
    if pos["decision"] == "BUY":
        if precio_actual > pos["best_price"]:
            pos["best_price"] = precio_actual
            new_sl = precio_actual - step_abs
            if new_sl > pos["trailing_sl"]:
                pos["trailing_sl"] = new_sl
    else:  # SELL
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
        # Actualizar trailing stop (si está activo)
        nuevo_sl = paper_actualizar_trailing(pos, precio_actual)
        # Verificar si se alcanza TP1 (solo si no se ha tomado aún)
        if not pos["tp1_hit"]:
            if (pos["decision"] == "BUY" and precio_actual >= pos["tp1_price"]) or \
               (pos["decision"] == "SELL" and precio_actual <= pos["tp1_price"]):
                # Cerrar la primera mitad
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
                # Una vez tomado TP1, seguimos con trailing para la segunda mitad
        # Verificar trailing stop (para la segunda mitad si TP1 ya se tomó, o para toda la posición si no)
        # Pero usamos el mismo nuevo_sl para ambas mitades. Si no se ha tomado TP1 y se toca trailing, se cierra todo.
        if (pos["decision"] == "BUY" and precio_actual <= nuevo_sl) or \
           (pos["decision"] == "SELL" and precio_actual >= nuevo_sl):
            # Calcular qué parte está aún abierta
            if pos["tp1_hit"]:
                # Solo queda la segunda mitad
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
                    "type": f"Trailing SL (50%)",
                    "pnl": pnl_trail,
                    "price": nuevo_sl,
                    "balance_after": PAPER_BALANCE
                })
            else:
                # No se había tomado TP1, se cierra todo el tamaño
                size_total = pos["size_btc_tp1"] + pos["size_btc_trail"]
                if pos["decision"] == "BUY":
                    pnl_total = (nuevo_sl - pos["entry_price"]) * size_total
                else:
                    pnl_total = (pos["entry_price"] - nuevo_sl) * size_total
                pnl_total -= abs(pnl_total) * 0.001
                PAPER_BALANCE += pnl_total
                PAPER_PNL_GLOBAL += pnl_total
                if pnl_total > 0:
                    PAPER_WIN += 1
                else:
                    PAPER_LOSS += 1
                cerradas.append({
                    "trade_id": pos["id"],
                    "type": "Trailing SL (completo)",
                    "pnl": pnl_total,
                    "price": nuevo_sl,
                    "balance_after": PAPER_BALANCE
                })
            PAPER_POSICIONES_ACTIVAS.remove(pos)
            continue
        # Si no se alcanzó TP1 ni trailing, pero el precio toca el SL inicial (aunque trailing lo haya subido, no es necesario porque trailing ya es más estricto)
        # No es necesario verificar SL inicial aparte, pues trailing ya lo cubre.
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
# TELEGRAM Y LOG (sin cambios)
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
# LOOP PRINCIPAL (con ajuste de TP/SL basado en niveles)
# ======================================================

def run_bot():
    global PAPER_PEAK_BALANCE, PAPER_DRAWDOWN_PAUSED_UNTIL
    telegram_mensaje("🤖 BOT SCALPING 5m INICIADO (Gemini multimodal, TP/SL por % y niveles)")
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

            decision, razones, _, fig = obtener_decision_ia_multimodal(df, soporte, resistencia, slope, tendencia)
            log_estado(df, tendencia, slope, soporte, resistencia, decision, razones)

            precio_actual = df['close'].iloc[-1]
            tiempo_actual = df.index[-1]

            cerradas, pausa = paper_revisar_sl_tp(precio_actual, tiempo_actual)
            if pausa:
                telegram_mensaje(f"⚠️ DRAWDOWN {MAX_DRAWDOWN_PERCENT}% - BOT PAUSADO 1H")
                continue

            for c in cerradas:
                resultado = "✅ GANADOR" if c["pnl"] > 0 else "❌ PERDEDOR"
                msg = f"📌 *CIERRE Trade #{c['trade_id']}* - {c['type']}\n{resultado}\n💰 PnL: {c['pnl']:+.4f} USD\n💵 Balance: {c['balance_after']:.2f} USD\n📊 W/L: {PAPER_WIN:.0f}/{PAPER_LOSS:.0f}"
                telegram_mensaje(msg)

            if PAPER_BALANCE > PAPER_PEAK_BALANCE:
                PAPER_PEAK_BALANCE = PAPER_BALANCE

            if decision in ("BUY", "SELL") and len(PAPER_POSICIONES_ACTIVAS) < MAX_SIMULTANEOUS_POSITIONS:
                nueva_pos = paper_abrir_posicion(decision, precio_actual, razones, tiempo_actual, soporte, resistencia)
                if nueva_pos:
                    msg_entrada = (
                        f"🚀 *NUEVA ENTRADA PAPER - Trade #{nueva_pos['id']}*\n"
                        f"📌 Dirección: {decision}\n"
                        f"💲 Entry: {precio_actual:.2f}\n"
                        f"🛑 SL: {nueva_pos['sl_initial']:.2f} ({(abs(nueva_pos['sl_initial']-precio_actual)/precio_actual*100):.2f}%)\n"
                        f"🎯 TP1 (50%): {nueva_pos['tp1_price']:.2f} ({(abs(nueva_pos['tp1_price']-precio_actual)/precio_actual*100):.2f}%)\n"
                        f"🔄 Trailing step: {TRAILING_STEP_PERCENT*100:.2f}%\n"
                        f"💰 Riesgo: {RISK_PER_TRADE*100:.2f}% balance\n"
                        f"📦 Tamaño: {nueva_pos['size_usd_total']:.2f} USD\n"
                        f"🧠 Setup IA:\n" + "\n".join(razones)
                    )
                    telegram_mensaje(msg_entrada)
                    _, fig_con_marca = generar_grafico_base64(
                        df, decision=decision, soporte=soporte, resistencia=resistencia,
                        slope=slope, intercept=intercept, razones=razones,
                        precio_entrada=precio_actual, trade_id=nueva_pos['id']
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
