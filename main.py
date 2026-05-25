import os
import time
import json
import itertools
import base64
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
INITIAL_BALANCE = 1000.0
COMMISSION = 0.001
MIN_MOVE_PERCENT = 1.2
LOOKAHEAD_VELAS = 10
MAX_SAMPLES = 20                 
MAX_TRADES_BACKTEST = 30          
RISK_PERCENT = 0.02
MAX_POSITION_PCT = 0.10

# URL del archivo CSV comprimido en GitHub (RAW)
DATA_URL = "https://github.com/webosdd/aur/raw/refs/heads/main/BTCUSDT_15m_data.zip"

# =================== FUNCIONES AUXILIARES ===================
def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def send_telegram(text, parse_mode="Markdown"):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log("Telegram no configurado")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": text[:4000], "parse_mode": parse_mode}, timeout=10)
    except Exception as e:
        log(f"Error enviando mensaje: {e}")

def send_telegram_image(image_buf, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    files = {'photo': ('image.png', image_buf, 'image/png')}
    data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': caption[:1000]}
    try:
        requests.post(url, files=files, data=data, timeout=15)
    except Exception as e:
        log(f"Error enviando imagen: {e}")

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

# =================== 1. CARGAR DATOS DESDE GITHUB ===================
def cargar_datos():
    log(f"📥 Descargando datos desde: {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL, compression='zip')
    except Exception as e:
        raise Exception(f"Error al cargar el archivo: {e}. Verifica la URL y que el archivo sea un ZIP con un CSV dentro.")
    
    # Identificar la columna de tiempo
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    elif 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    elif 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
    else:
        # Si no encuentra, asume que la primera columna es el índice de tiempo
        df.index = pd.to_datetime(df.index)
    
    # Asegurar columnas numéricas
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            raise KeyError(f"El CSV debe contener una columna '{col}'. Verifica las columnas: {df.columns.tolist()}")
    
    df = df.sort_index()
    log(f"✅ Datos cargados: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
    return df

# =================== 2. GENERAR MUESTRAS ===================
def detectar_movimientos(df):
    movimientos = []
    for i in range(len(df) - LOOKAHEAD_VELAS):
        precio_actual = df['close'].iloc[i]
        precio_futuro = df['close'].iloc[i + LOOKAHEAD_VELAS]
        cambio = (precio_futuro - precio_actual) / precio_actual * 100
        if cambio > MIN_MOVE_PERCENT:
            movimientos.append((df.index[i], 'BUY', cambio))
        elif cambio < -MIN_MOVE_PERCENT:
            movimientos.append((df.index[i], 'SELL', -cambio))
    return movimientos

def generar_muestras(df, timeframe="15"):
    sample_dir = f"samples/{timeframe}m"
    os.makedirs(sample_dir, exist_ok=True)
    movimientos = detectar_movimientos(df)
    log(f"Movimientos detectados: {len(movimientos)}")
    muestras = []
    for idx, direction, change in movimientos[:MAX_SAMPLES]:
        start_idx = max(0, df.index.get_loc(idx) - 80)
        end_idx = df.index.get_loc(idx)
        df_window = df.iloc[start_idx:end_idx+1].copy()
        if len(df_window) < 30:
            continue
        img_buf = generar_grafico(df_window, f"{direction} {change:.1f}%", entry_price=df.loc[idx]['close'])
        img_path = f"{sample_dir}/{idx.strftime('%Y%m%d_%H%M')}_{direction}_{change:.1f}.png"
        with open(img_path, 'wb') as f:
            f.write(img_buf.getvalue())
        muestras.append({
            'timestamp': idx,
            'direction': direction,
            'cambio': change,
            'image_path': img_path,
            'image_buf': img_buf
        })
    log(f"✅ Generadas {len(muestras)} muestras")
    return muestras

# =================== 3. CONDICIONES ATÓMICAS ===================
CONDICIONES_ATOMICAS = {}

def definir_condiciones_atomicas():
    global CONDICIONES_ATOMICAS
    def rsi_lt_30(df, idx):
        return df['rsi'].iloc[idx] < 30
    def rsi_gt_70(df, idx):
        return df['rsi'].iloc[idx] > 70
    def price_lt_ema20(df, idx):
        return df['close'].iloc[idx] < df['ema20'].iloc[idx]
    def price_gt_ema20(df, idx):
        return df['close'].iloc[idx] > df['ema20'].iloc[idx]
    def macd_cross_above(df, idx):
        return (df['macd'].iloc[idx] > df['macd_signal'].iloc[idx]) and \
               (df['macd'].iloc[idx-1] <= df['macd_signal'].iloc[idx-1])
    def macd_cross_below(df, idx):
        return (df['macd'].iloc[idx] < df['macd_signal'].iloc[idx]) and \
               (df['macd'].iloc[idx-1] >= df['macd_signal'].iloc[idx-1])
    def close_near_support(df, idx):
        soporte = df['low'].iloc[max(0,idx-20):idx+1].min()
        return abs(df['close'].iloc[idx] - soporte) / df['close'].iloc[idx] < 0.002
    def close_near_resistance(df, idx):
        resistencia = df['high'].iloc[max(0,idx-20):idx+1].max()
        return abs(resistencia - df['close'].iloc[idx]) / df['close'].iloc[idx] < 0.002
    def price_above_ema50(df, idx):
        return df['close'].iloc[idx] > df['ema50'].iloc[idx]
    def price_below_ema50(df, idx):
        return df['close'].iloc[idx] < df['ema50'].iloc[idx]
    CONDICIONES_ATOMICAS = {
        "rsi_lt_30": rsi_lt_30,
        "rsi_gt_70": rsi_gt_70,
        "price_lt_ema20": price_lt_ema20,
        "price_gt_ema20": price_gt_ema20,
        "macd_cross_above": macd_cross_above,
        "macd_cross_below": macd_cross_below,
        "close_near_support": close_near_support,
        "close_near_resistance": close_near_resistance,
        "price_above_ema50": price_above_ema50,
        "price_below_ema50": price_below_ema50,
    }
definir_condiciones_atomicas()

# =================== 4. BACKTEST CON COMBINACIONES ===================
def backtest_con_regla(df, condicion_func):
    balance = INITIAL_BALANCE
    in_position = False
    position = None
    trades = []
    for i in range(100, len(df)):
        if not in_position:
            if condicion_func(df, i):
                entry = df['close'].iloc[i]
                sl = entry * 0.995
                tp1 = entry * 1.005
                qty = (balance * RISK_PERCENT) / (entry - sl) if (entry - sl) > 0 else 0
                max_qty = (balance * MAX_POSITION_PCT) / entry
                qty = min(qty, max_qty)
                if qty < 0.0001:
                    continue
                in_position = True
                position = {'entry': entry, 'sl': sl, 'tp1': tp1, 'qty': qty, 'direction': 'BUY'}
        else:
            current = df['close'].iloc[i]
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
            if len(trades) >= MAX_TRADES_BACKTEST:
                break
    return trades, balance

def evaluar_combinaciones(df, lista_condiciones):
    if not lista_condiciones:
        return None, []
    cond_funcs = [(nombre, CONDICIONES_ATOMICAS[nombre]) for nombre in lista_condiciones]
    resultados = []
    # Combinaciones de 2
    for (nom1, f1), (nom2, f2) in itertools.combinations(cond_funcs, 2):
        regla = lambda df, i, f1=f1, f2=f2: f1(df, i) and f2(df, i)
        trades, balance_final = backtest_con_regla(df, regla)
        if not trades:
            continue
        pnl_total = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        winrate = wins / len(trades) * 100
        resultados.append({
            'regla': f"{nom1} AND {nom2}",
            'trades': len(trades),
            'winrate': winrate,
            'pnl_total': pnl_total,
            'balance_final': balance_final
        })
    # Combinaciones de 3
    for (nom1, f1), (nom2, f2), (nom3, f3) in itertools.combinations(cond_funcs, 3):
        regla = lambda df, i, f1=f1, f2=f2, f3=f3: f1(df, i) and f2(df, i) and f3(df, i)
        trades, balance_final = backtest_con_regla(df, regla)
        if not trades:
            continue
        pnl_total = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        winrate = wins / len(trades) * 100
        resultados.append({
            'regla': f"{nom1} AND {nom2} AND {nom3}",
            'trades': len(trades),
            'winrate': winrate,
            'pnl_total': pnl_total,
            'balance_final': balance_final
        })
    if not resultados:
        return None, []
    resultados.sort(key=lambda x: x['winrate'], reverse=True)
    return resultados[0], resultados[:5]

# =================== 5. PROCESAR DATOS ===================
def procesar(df):
    log("Calculando indicadores...")
    df = calcular_indicadores(df)
    muestras = generar_muestras(df, timeframe="15")
    todas_condiciones = list(CONDICIONES_ATOMICAS.keys())
    log(f"Probando {len(todas_condiciones)} condiciones en combinaciones de 2 y 3...")
    mejor, top = evaluar_combinaciones(df, todas_condiciones)
    if mejor:
        log(f"✅ Mejor regla: {mejor['regla']} -> winrate {mejor['winrate']:.1f}%")
    else:
        log("⚠️ No se encontraron reglas con trades")
    return mejor, top, muestras

# =================== 6. REPORTE A TELEGRAM ===================
def enviar_resumen(mejor, muestras):
    if not mejor:
        send_telegram("❌ No se encontró ninguna regla con trades para los datos proporcionados.")
        return
    msg = f"📊 *RESULTADO BACKTEST (15m)*\n\n"
    msg += f"🏆 Mejor regla:\n{mejor['regla']}\n"
    msg += f"📈 Winrate: {mejor['winrate']:.1f}% ({mejor['trades']} trades)\n"
    msg += f"💰 PnL total: {mejor['pnl_total']:+.2f} USDT\n"
    msg += f"💵 Balance final: {mejor['balance_final']:.2f} USDT\n"
    send_telegram(msg)

    # Enviar gráficos de las muestras (hasta 15)
    for idx, m in enumerate(muestras[:15]):
        caption = f"🔔 Señal #{idx+1}: {m['direction']} {m['cambio']:.1f}%\n{m['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        with open(m['image_path'], 'rb') as f:
            img_buf = BytesIO(f.read())
        send_telegram_image(img_buf, caption)

# =================== MAIN ===================
def main():
    log("🚀 Iniciando backtest con datos desde GitHub")
    df = cargar_datos()
    mejor, top, muestras = procesar(df)
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        enviar_resumen(mejor, muestras)
    else:
        log("⚠️ Telegram no configurado. Resultados en consola:")
        if mejor:
            print(mejor)
    log("✅ Pipeline finalizado")

if __name__ == "__main__":
    main()
