# ======================================================
# BOT TRADING V90.2 BYBIT – IA GEMINI 3.1 FLASH IMAGE PREVIEW
# ======================================================

import os
import time
import io
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

# ======================================================
# CREDENCIALES (variables de entorno en Railway)
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

# Cliente OpenRouter con el modelo correcto
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "BTC Trading Bot"}
)

# Ruta CORRECTA del modelo, verificada en la documentación de OpenRouter
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
# INDICADORES
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
# IA GEMINI CORREGIDA (con la ruta correcta y manejo de errores)
# ======================================================

def obtener_decision_ia(df, soporte, resistencia, slope, tendencia):
    ultimo = df.iloc[-1]
    precio = ultimo['close']
    ema20 = ultimo['ema20']
    atr = ultimo['atr']
    rsi = ultimo['rsi'] if 'rsi' in ultimo else 50
    ultimas_velas = df.tail(5)[['open','high','low','close']].to_dict(orient='records')
    velas_texto = "\n".join([f"  {v}" for v in ultimas_velas])

    prompt = f"""
Eres un trader algorítmico experto en BTC/USDT en timeframe de {INTERVAL} minutos.
Analiza la siguiente situación y decide si comprar (BUY), vender (SELL) o no hacer nada (HOLD).
Devuelve ÚNICAMENTE un JSON válido con este formato exacto (sin texto adicional, solo el JSON):
{{"decision": "BUY/SELL/HOLD", "razones": ["razón1", "razón2", ...]}}

Datos actuales:
- Precio actual: {precio:.2f}
- EMA20: {ema20:.2f}
- ATR: {atr:.2f}
- RSI: {rsi:.1f}
- Tendencia (80 velas): {tendencia} (slope {slope:.5f})
- Soporte: {soporte:.2f}
- Resistencia: {resistencia:.2f}
- Últimas 5 velas:
{velas_texto}

Reglas:
- BUY solo si hay confluencia alcista (precio cerca de soporte, tendencia alcista, RSI > 30).
- SELL solo si hay confluencia bajista (precio cerca de resistencia, tendencia bajista, RSI < 70).
- HOLD si no está claro.
"""
    try:
        response = client.chat.completions.create(
            model=MODELO_IA,  # Ruta correcta
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        contenido = response.choices[0].message.content
        print(f"Respuesta IA cruda: {contenido[:200]}...")  # Depuración
        
        if not contenido:
            raise Exception("Respuesta vacía de la IA")

        # Intentar extraer JSON de la respuesta (por si el modelo añade texto adicional)
        json_match = re.search(r'\{.*\}', contenido, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            data = json.loads(json_str)
        else:
            # Si no hay JSON, intentar parsear directamente
            data = json.loads(contenido)
        
        decision = data.get("decision", "HOLD").upper()
        razones = data.get("razones", ["Sin razones específicas"])
        if decision not in ["BUY", "SELL", "HOLD"]:
            decision = "HOLD"
        return decision, razones

    except json.JSONDecodeError as e:
        print(f"Error JSON: {e} - Contenido: {contenido}")
        # En caso de error, asumimos HOLD para no operar con datos corruptos
        return "HOLD", [f"Error parseando JSON: {e}, se asume HOLD"]
    except Exception as e:
        print(f"Error IA: {e}")
        return "HOLD", [f"Error en llamada IA: {e}, se asume HOLD"]

# ======================================================
# PAPER TRADING (sin cambios)
# ======================================================

def paper_abrir_posicion(decision, precio, atr, razones, tiempo):
    global PAPER_BALANCE, PAPER_POSICIONES_ACTIVAS, PAPER_TRADES_TOTALES
    if len(PAPER_POSICIONES_ACTIVAS) >= MAX_SIMULTANEOUS_POSITIONS:
        return False

    riesgo_usd = PAPER_BALANCE * RISK_PER_TRADE
    if decision == "BUY":
        sl = precio - atr
        tp = precio + (atr * 2)
    else:
        sl = precio + atr
        tp = precio - (atr * 2)

    distancia_sl = abs(precio - sl)
    if distancia_sl == 0:
        return False
    size_btc = riesgo_usd / distancia_sl
    size_usd = size_btc * precio

    pos = {
        "id": len(PAPER_POSICIONES_ACTIVAS) + len(PAPER_TRADES_CERRADOS) + 1,
        "decision": decision,
        "entry_price": precio,
        "entry_time": tiempo,
        "sl": sl,
        "tp": tp,
        "size_btc": size_btc,
        "size_usd": size_usd,
        "razones": razones
    }
    PAPER_POSICIONES_ACTIVAS.append(pos)
    PAPER_TRADES_TOTALES += 1
    return True

def paper_calcular_pnl_posicion(pos, precio_actual):
    if pos["decision"] == "BUY":
        return (precio_actual - pos["entry_price"]) * pos["size_btc"]
    else:
        return (pos["entry_price"] - precio_actual) * pos["size_btc"]

def paper_revisar_sl_tp(precio_actual, tiempo_actual):
    global PAPER_BALANCE, PAPER_PNL_GLOBAL, PAPER_WIN, PAPER_LOSS
    global PAPER_TRADES_CERRADOS, PAPER_POSICIONES_ACTIVAS
    global PAPER_PEAK_BALANCE, PAPER_DRAWDOWN_PAUSED_UNTIL

    cerradas = []
    for pos in PAPER_POSICIONES_ACTIVAS[:]:
        cerrar = False
        motivo = None
        if pos["decision"] == "BUY":
            if precio_actual <= pos["sl"]:
                cerrar = True
                motivo = "SL"
            elif precio_actual >= pos["tp"]:
                cerrar = True
                motivo = "TP"
        else:
            if precio_actual >= pos["sl"]:
                cerrar = True
                motivo = "SL"
            elif precio_actual <= pos["tp"]:
                cerrar = True
                motivo = "TP"
        if cerrar:
            pnl = paper_calcular_pnl_posicion(pos, precio_actual)
            PAPER_BALANCE += pnl
            PAPER_PNL_GLOBAL += pnl
            if pnl > 0:
                PAPER_WIN += 1
            else:
                PAPER_LOSS += 1
            pos_cerrada = {
                **pos,
                "exit_price": precio_actual,
                "exit_time": tiempo_actual,
                "pnl": pnl,
                "motivo": motivo,
                "balance_after": PAPER_BALANCE
            }
            PAPER_TRADES_CERRADOS.append(pos_cerrada)
            PAPER_POSICIONES_ACTIVAS.remove(pos)
            cerradas.append(pos_cerrada)

            if PAPER_BALANCE > PAPER_PEAK_BALANCE:
                PAPER_PEAK_BALANCE = PAPER_BALANCE
            drawdown_pct = (PAPER_PEAK_BALANCE - PAPER_BALANCE) / PAPER_PEAK_BALANCE * 100
            if drawdown_pct >= MAX_DRAWDOWN_PERCENT and PAPER_DRAWDOWN_PAUSED_UNTIL is None:
                PAPER_DRAWDOWN_PAUSED_UNTIL = tiempo_actual + pd.Timedelta(seconds=PAUSE_ON_DRAWDOWN_SECONDS)
                return cerradas, True
    return cerradas, False

# ======================================================
# GRÁFICO DE VELAS DETALLADO
# ======================================================

def generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones, precio_entrada=None):
    try:
        df_plot = df.copy().tail(120)
        if df_plot.empty:
            return None
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
        ax.axhline(soporte, color='cyan', linestyle='--', linewidth=2, label=f"Soporte {soporte:.2f}")
        ax.axhline(resistencia, color='magenta', linestyle='--', linewidth=2, label=f"Resistencia {resistencia:.2f}")
        if 'ema20' in df_plot.columns:
            ax.plot(x, df_plot['ema20'].values, color='yellow', linewidth=2, label='EMA20')
        y_plot = df_plot['close'].values
        x_plot = np.arange(len(y_plot))
        slope_plot, intercept_plot = np.polyfit(x_plot, y_plot, 1)
        tendencia_linea = intercept_plot + slope_plot * x_plot
        ax.plot(x_plot, tendencia_linea, color='white', linewidth=2, linestyle='-', label=f"Tendencia slope {slope_plot:.4f}")
        entrada_index = len(df_plot) - 1
        if precio_entrada is None:
            precio_entrada = df_plot['close'].iloc[-1]
        if decision == 'BUY':
            ax.scatter(entrada_index, precio_entrada, s=200, marker='^', color='lime', edgecolors='black', label='Entrada BUY')
            ax.axvline(entrada_index, color='lime', linestyle=':', linewidth=2)
        elif decision == 'SELL':
            ax.scatter(entrada_index, precio_entrada, s=200, marker='v', color='red', edgecolors='black', label='Entrada SELL')
            ax.axvline(entrada_index, color='red', linestyle=':', linewidth=2)
        texto = f"{decision}\nPrecio: {precio_entrada:.2f}\nBalance: {PAPER_BALANCE:.2f} USD\nPnL Global: {PAPER_PNL_GLOBAL:.4f}\nRazones:\n" + "\n".join(razones[:4])
        ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='black', alpha=0.7, boxstyle='round'), color='white')
        ax.set_title(f"{SYMBOL} - Entrada {decision} (IA Gemini 3.1 Flash) - {INTERVAL}m")
        ax.set_xlabel("Velas")
        ax.set_ylabel("Precio")
        ax.grid(True, alpha=0.2)
        ax.legend(loc='lower left')
        step = max(1, int(len(df_plot)/10))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([t.strftime('%H:%M') for t in df_plot.index[::step]], rotation=45)
        plt.tight_layout()
        return fig
    except Exception as e:
        print(f"Error gráfico: {e}")
        return None

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

# ======================================================
# LOG EN CONSOLA
# ======================================================

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
    telegram_mensaje("🤖 BOT V90.2 INICIADO (Gemini 3.1 Flash Image Preview, 5m, max 3 posiciones, drawdown 20%)")
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

            decision, razones = obtener_decision_ia(df, soporte, resistencia, slope, tendencia)
            log_estado(df, tendencia, slope, soporte, resistencia, decision, razones)

            precio_actual = df['close'].iloc[-1]
            tiempo_actual = df.index[-1]

            cerradas, pausa_activada = paper_revisar_sl_tp(precio_actual, tiempo_actual)
            if pausa_activada:
                telegram_mensaje(f"⚠️ DRAWDOWN SUPERADO ({MAX_DRAWDOWN_PERCENT}%) - BOT PAUSADO 1 HORA")
                continue

            if PAPER_BALANCE > PAPER_PEAK_BALANCE:
                PAPER_PEAK_BALANCE = PAPER_BALANCE

            for c in cerradas:
                msg = (
                    f"📌 CIERRE PAPER {c['decision']} ({c['motivo']})\n"
                    f"Entrada: {c['entry_price']:.2f} | Salida: {c['exit_price']:.2f}\n"
                    f"PnL: {c['pnl']:.4f} USD\n"
                    f"Balance: {c['balance_after']:.2f} USD\n"
                    f"Win/Loss: {PAPER_WIN}/{PAPER_LOSS}"
                )
                telegram_mensaje(msg)

            if decision in ("BUY", "SELL") and len(PAPER_POSICIONES_ACTIVAS) < MAX_SIMULTANEOUS_POSITIONS:
                atr_actual = df['atr'].iloc[-1]
                abierta = paper_abrir_posicion(decision, precio_actual, atr_actual, razones, tiempo_actual)
                if abierta:
                    msg = (
                        f"📌 ENTRADA PAPER {decision} (IA Gemini 3.1 Flash)\n"
                        f"Precio: {precio_actual:.2f}\n"
                        f"SL: {PAPER_POSICIONES_ACTIVAS[-1]['sl']:.2f} | TP: {PAPER_POSICIONES_ACTIVAS[-1]['tp']:.2f}\n"
                        f"Size USD: {PAPER_POSICIONES_ACTIVAS[-1]['size_usd']:.2f}\n"
                        f"Balance: {PAPER_BALANCE:.2f} USD\n"
                        f"Posiciones activas: {len(PAPER_POSICIONES_ACTIVAS)}\n"
                        f"Razones: {', '.join(razones)}"
                    )
                    telegram_mensaje(msg)
                    fig = generar_grafico_entrada(df, decision, soporte, resistencia, slope, intercept, razones, precio_actual)
                    if fig:
                        telegram_grafico(fig)
                        plt.close(fig)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"🚨 ERROR: {e}")
            telegram_mensaje(f"🚨 ERROR BOT: {e}")
            time.sleep(60)

if __name__ == '__main__':
    run_bot()
