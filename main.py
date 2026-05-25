import os
import time
import base64
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import datetime
from pybit.unified_trading import HTTP
from ta.momentum import RSIIndicator
from ta.trend import MACD
from openai import OpenAI
from dotenv import load_dotenv

# =================== CONFIGURACIÓN ===================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SYMBOL = "BTCUSDT"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
DATA_DIR = "data"
SAMPLES_DIR = "samples"
INITIAL_BALANCE = 1000.0
COMMISSION = 0.001
MIN_MOVE_PERCENT = 1.2        # Movimiento mínimo para considerar muestra
LOOKAHEAD_VELAS = 10           # Velas adelante para calcular movimiento
MAX_SAMPLES = 50               # Máximo de muestras a generar

# =================== FUNCIONES AUXILIARES ===================
def send_telegram(text, parse_mode="Markdown"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram no configurado")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text[:4000], "parse_mode": parse_mode}, timeout=10)
    except Exception as e:
        print(f"Error enviando mensaje: {e}")

def send_telegram_image(image_buf, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {'photo': ('image.png', image_buf, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption[:1000]}
    try:
        requests.post(url, files=files, data=data, timeout=15)
    except Exception as e:
        print(f"Error enviando imagen: {e}")

def calcular_indicadores(df):
    if df.empty:
        return df
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    return df

def generar_grafico(df, titulo, entry_price=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3,1,1]})
    ax1.plot(df.index, df['close'], 'black', linewidth=1, label='Close')
    if 'ema20' in df:
        ax1.plot(df.index, df['ema20'], 'orange', linewidth=1, label='EMA20')
    if 'ema50' in df:
        ax1.plot(df.index, df['ema50'], 'blue', linewidth=1, label='EMA50')
    if entry_price:
        ax1.axhline(entry_price, color='green', linestyle='--', label='Entry')
    ax1.legend()
    ax1.set_title(titulo)
    if 'rsi' in df:
        ax2.plot(df.index, df['rsi'], 'purple')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_ylabel('RSI')
    if 'macd' in df:
        ax3.plot(df.index, df['macd'], label='MACD')
        ax3.plot(df.index, df['macd_signal'], label='Signal')
        ax3.bar(df.index, df['macd_diff'], label='Histogram')
        ax3.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

# =================== 1. DESCARGA DE DATOS (BYBIT) ===================
def fetch_klines(interval):
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = f"{DATA_DIR}/{SYMBOL}_{interval}m_{START_DATE}_{END_DATE}.csv"
    if os.path.exists(filepath):
        print(f"✅ Datos ya existen: {filepath}")
        return filepath
    session = HTTP(testnet=False)
    start_ms = int(datetime.strptime(START_DATE, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.strptime(END_DATE, "%Y-%m-%d").timestamp() * 1000)
    all_klines = []
    current_start = start_ms
    print(f"Descargando {SYMBOL} {interval}m...")
    while current_start < end_ms:
        resp = session.get_kline(
            category="linear",
            symbol=SYMBOL,
            interval=str(interval),
            start=current_start,
            end=end_ms,
            limit=1000
        )
        if resp['retCode'] != 0:
            print(f"Error: {resp}")
            break
        klines = resp['result']['list']
        if not klines:
            break
        all_klines.extend(klines)
        current_start = int(klines[-1][0]) + 1
        time.sleep(0.5)
    if not all_klines:
        return None
    df = pd.DataFrame(all_klines, columns=['timestamp','open','high','low','close','volume','turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp')
    df.to_csv(filepath, index=False)
    print(f"✅ Datos guardados: {filepath}")
    return filepath

# =================== 2. GENERAR MUESTRAS CON IMÁGENES ===================
def detectar_movimientos(df_10m):
    movimientos = []
    for i in range(len(df_10m) - LOOKAHEAD_VELAS):
        precio_actual = df_10m['close'].iloc[i]
        precio_futuro = df_10m['close'].iloc[i + LOOKAHEAD_VELAS]
        cambio = (precio_futuro - precio_actual) / precio_actual * 100
        if cambio > MIN_MOVE_PERCENT:
            movimientos.append((df_10m.index[i], 'BUY', cambio))
        elif cambio < -MIN_MOVE_PERCENT:
            movimientos.append((df_10m.index[i], 'SELL', -cambio))
    return movimientos

def generar_muestras(df_10m):
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    movimientos = detectar_movimientos(df_10m)
    print(f"Movimientos detectados: {len(movimientos)}")
    muestras = []
    for idx, direction, change in movimientos[:MAX_SAMPLES]:
        # ventana de 100 velas alrededor
        start_idx = max(0, df_10m.index.get_loc(idx) - 80)
        end_idx = df_10m.index.get_loc(idx)
        df_window = df_10m.iloc[start_idx:end_idx+1].copy()
        if len(df_window) < 30:
            continue
        img_buf = generar_grafico(df_window, f"{direction} {change:.1f}%", entry_price=df_10m.loc[idx]['close'])
        img_path = f"{SAMPLES_DIR}/{idx.strftime('%Y%m%d_%H%M')}_{direction}_{change:.1f}.png"
        with open(img_path, 'wb') as f:
            f.write(img_buf.getvalue())
        muestras.append({
            'timestamp': idx,
            'direction': direction,
            'cambio': change,
            'image_path': img_path,
            'image_buf': img_buf
        })
    print(f"✅ Generadas {len(muestras)} muestras en {SAMPLES_DIR}")
    return muestras

# =================== 3. (OPCIONAL) EXTRAER REGLAS CON IA ===================
# Solo ejecutar si se desea y se tiene OPENROUTER_API_KEY
def extraer_reglas_con_ia(muestras, max_imagenes=20):
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY no configurada, no se extraerán reglas con IA")
        return
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Backtest Rule Extractor"}
    )
    prompt = """
Eres un trader profesional. Analiza este gráfico de BTC/USDT (velas de 10 minutos).
En la línea verde vertical (momento indicado) el precio subió más de un 1.2% en las siguientes 10 velas.
Describe las condiciones que justifican una entrada COMPRA:
- Niveles de soporte/resistencia (precios)
- Patrones de velas (martillo, engulfing, etc.)
- Posición respecto a EMAs 20 y 50
- Valores de RSI y MACD (divergencias, sobreventa, cruce)
- Estructura de mercado
Responde como lista de condiciones objetivas y programables (ej: "RSI < 30", "precio cruza EMA20 al alza").
"""
    reglas = []
    for i, m in enumerate(muestras[:max_imagenes]):
        if m['direction'] != 'BUY':
            continue
        print(f"Enviando imagen {i+1}: {m['image_path']}")
        with open(m['image_path'], 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4.5",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
            ]}],
            temperature=0.2,
            max_tokens=1500
        )
        reglas.append(f"--- {m['timestamp']} ---\n{response.choices[0].message.content}")
        time.sleep(2)
    with open("reglas_extraidas.txt", "w", encoding='utf-8') as f:
        f.write("\n\n".join(reglas))
    print("✅ Reglas guardadas en reglas_extraidas.txt")

# =================== 4. BACKTEST CON REGLAS CODIFICADAS ===================
# Aquí debes codificar las reglas que extrajiste manualmente
def es_buy(df_10m, df_60m, idx):
    """
    Ejemplo de regla: compra si RSI < 30 y precio está cerca de EMA20 (rebote).
    Reemplaza con tus propias reglas basadas en el archivo reglas_extraidas.txt.
    """
    close = df_10m['close'].iloc[idx]
    rsi = df_10m['rsi'].iloc[idx]
    ema20 = df_10m['ema20'].iloc[idx]
    # Regla 1: RSI en sobreventa y precio por debajo de EMA20?
    if rsi < 30 and close < ema20:
        return True
    # Regla 2: MACD cruce alcista
    macd = df_10m['macd'].iloc[idx]
    signal = df_10m['macd_signal'].iloc[idx]
    macd_prev = df_10m['macd'].iloc[idx-1]
    signal_prev = df_10m['macd_signal'].iloc[idx-1]
    if macd > signal and macd_prev <= signal_prev:
        return True
    return False

def es_sell(df_10m, df_60m, idx):
    # Similar para ventas (puedes implementar si quieres)
    return False

def backtest(df_10m, df_60m):
    balance = INITIAL_BALANCE
    in_position = False
    position = None
    trades = []
    for i in range(100, len(df_10m)):  # empezar con suficientes velas
        if not in_position:
            if es_buy(df_10m, df_60m, i):
                entry = df_10m['close'].iloc[i]
                sl = entry * 0.995  # -0.5%
                tp1 = entry * 1.005  # +0.5%
                qty = (balance * 0.02) / entry  # riesgo 2%
                if qty * entry > balance * 0.1:  # no arriesgar más del 10% del balance
                    qty = (balance * 0.1) / entry
                in_position = True
                position = {
                    'entry_idx': i,
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'qty': qty,
                    'direction': 'BUY'
                }
        else:
            current = df_10m['close'].iloc[i]
            if position['direction'] == 'BUY':
                if current >= position['tp1']:
                    pnl = (position['tp1'] - position['entry']) * position['qty']
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({'pnl': pnl, 'exit': 'TP1'})
                    in_position = False
                elif current <= position['sl']:
                    pnl = (position['sl'] - position['entry']) * position['qty']
                    pnl -= abs(pnl) * COMMISSION
                    balance += pnl
                    trades.append({'pnl': pnl, 'exit': 'SL'})
                    in_position = False
    return trades, balance

# =================== 5. REPORTE A TELEGRAM (RESUMEN + 20 IMÁGENES) ===================
def enviar_resumen_y_graficos(trades, final_balance, muestras):
    total = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    losses = total - wins
    pnl_total = sum(t['pnl'] for t in trades)
    winrate = (wins / total * 100) if total else 0
    resumen = f"""📊 *BACKTEST COMPLETADO*
Trades totales: {total}
✅ Ganados: {wins}
❌ Perdidos: {losses}
📈 Winrate: {winrate:.1f}%
💰 PnL total: {pnl_total:+.2f} USDT
💵 Balance final: {final_balance:.2f} USDT
"""
    send_telegram(resumen)

    # Enviar hasta 20 imágenes de las muestras (señales generadas)
    for idx, m in enumerate(muestras[:20]):
        caption = f"🔔 Señal #{idx+1}: {m['direction']} {m['cambio']:.1f}%\n{m['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        # reutilizar el buffer de imagen guardado
        with open(m['image_path'], 'rb') as f:
            img_buf = BytesIO(f.read())
        send_telegram_image(img_buf, caption)

# =================== MAIN ===================
def main():
    print("🚀 Iniciando pipeline de backtest...")
    # 1. Descargar datos
    ltf_path = fetch_klines(10)
    htf_path = fetch_klines(60)
    if not ltf_path or not htf_path:
        print("❌ Error descargando datos")
        return

    df_10m = pd.read_csv(ltf_path, parse_dates=['timestamp'], index_col='timestamp')
    df_60m = pd.read_csv(htf_path, parse_dates=['timestamp'], index_col='timestamp')
    df_10m = calcular_indicadores(df_10m)
    df_60m = calcular_indicadores(df_60m)

    # 2. Generar muestras
    muestras = generar_muestras(df_10m)

    # 3. (Opcional) Extraer reglas con IA - descomentar si se tiene API key
    # extraer_reglas_con_ia(muestras, max_imagenes=20)

    # 4. Backtest con reglas codificadas
    trades, final_balance = backtest(df_10m, df_60m)

    # 5. Reportar a Telegram
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        enviar_resumen_y_graficos(trades, final_balance, muestras)
    else:
        print("⚠️ Telegram no configurado. Resultados:")
        print(f"Trades: {len(trades)}, PnL: {sum(t['pnl'] for t in trades):.2f}")

    print("✅ Pipeline finalizado")

if __name__ == "__main__":
    main()
