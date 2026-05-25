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
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
DATA_DIR = "data"
SAMPLES_DIR = "samples"
INITIAL_BALANCE = 1000.0
COMMISSION = 0.001
MIN_MOVE_PERCENT = 1.2
LOOKAHEAD_VELAS = 10
MAX_SAMPLES = 50
MAX_TRADES_BACKTEST = 50          # Límite de trades en backtest
RISK_PERCENT = 0.02               # 2% del balance por trade
MAX_POSITION_PCT = 0.10           # Máx 10% del balance en una posición

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

# =================== 1. DESCARGA DE DATOS ===================
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
def extraer_reglas_con_ia(muestras, max_imagenes=20):
    if not OPENROUTER_API_KEY:
        print("❌ OPENROUTER_API_KEY no configurada, omitiendo extracción de reglas")
        return
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Rule Extractor"}
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

# =================== 4. BACKTEST CON MÚLTIPLES REGLAS ===================
def backtest_con_regla(df_10m, df_60m, condicion_buy):
    """
    Ejecuta backtest con una condición de compra (función que recibe df_10m, df_60m, índice).
    Devuelve lista de trades y balance final.
    """
    balance = INITIAL_BALANCE
    in_position = False
    position = None
    trades = []
    for i in range(100, len(df_10m)):
        if not in_position:
            if condicion_buy(df_10m, df_60m, i):
                entry = df_10m['close'].iloc[i]
                sl = entry * 0.995      # -0.5%
                tp1 = entry * 1.005     # +0.5%
                qty = (balance * RISK_PERCENT) / (entry - sl) if (entry - sl) > 0 else 0
                max_qty = (balance * MAX_POSITION_PCT) / entry
                qty = min(qty, max_qty)
                if qty * entry > balance * MAX_POSITION_PCT:
                    qty = (balance * MAX_POSITION_PCT) / entry
                if qty < 0.0001:   # mínimo BTC
                    continue
                in_position = True
                position = {
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'qty': qty,
                    'direction': 'BUY'
                }
        else:
            current = df_10m['close'].iloc[i]
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

def evaluar_reglas(df_10m, df_60m):
    """
    Define condiciones atómicas y prueba combinaciones en AND.
    Retorna la mejor combinación (tupla de condiciones) y sus métricas.
    """
    # Condiciones atómicas (todas reciben df_10m, idx)
    def cond_rsi_sobreventa(df, _, idx):
        return df['rsi'].iloc[idx] < 30
    def cond_rsi_sobrecompra(df, _, idx):
        return df['rsi'].iloc[idx] > 70
    def cond_precio_debajo_ema20(df, _, idx):
        return df['close'].iloc[idx] < df['ema20'].iloc[idx]
    def cond_precio_encima_ema20(df, _, idx):
        return df['close'].iloc[idx] > df['ema20'].iloc[idx]
    def cond_macd_cruce_alcista(df, _, idx):
        return (df['macd'].iloc[idx] > df['macd_signal'].iloc[idx]) and \
               (df['macd'].iloc[idx-1] <= df['macd_signal'].iloc[idx-1])
    def cond_macd_cruce_bajista(df, _, idx):
        return (df['macd'].iloc[idx] < df['macd_signal'].iloc[idx]) and \
               (df['macd'].iloc[idx-1] >= df['macd_signal'].iloc[idx-1])
    def cond_precio_cerca_soporte(df, _, idx):
        # soporte simple: mínimo de últimas 20 velas
        soporte = df['low'].iloc[max(0,idx-20):idx+1].min()
        return abs(df['close'].iloc[idx] - soporte) / df['close'].iloc[idx] < 0.002
    def cond_precio_cerca_resistencia(df, _, idx):
        resistencia = df['high'].iloc[max(0,idx-20):idx+1].max()
        return abs(resistencia - df['close'].iloc[idx]) / df['close'].iloc[idx] < 0.002

    condiciones = [
        ('RSI < 30', cond_rsi_sobreventa),
        ('RSI > 70', cond_rsi_sobrecompra),
        ('Precio < EMA20', cond_precio_debajo_ema20),
        ('Precio > EMA20', cond_precio_encima_ema20),
        ('MACD cruce alcista', cond_macd_cruce_alcista),
        ('MACD cruce bajista', cond_macd_cruce_bajista),
        ('Precio cerca de soporte', cond_precio_cerca_soporte),
        ('Precio cerca de resistencia', cond_precio_cerca_resistencia),
    ]

    resultados = []
    # Probar combinaciones de 2 condiciones en AND
    for (nombre1, c1), (nombre2, c2) in itertools.combinations(condiciones, 2):
        regla = lambda df, _, i: c1(df, _, i) and c2(df, _, i)
        trades, balance_final = backtest_con_regla(df_10m, df_60m, regla)
        if not trades:
            continue
        pnl_total = sum(t['pnl'] for t in trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        winrate = wins / len(trades) * 100 if trades else 0
        resultados.append({
            'regla': f"{nombre1} AND {nombre2}",
            'trades': len(trades),
            'winrate': winrate,
            'pnl_total': pnl_total,
            'balance_final': balance_final
        })
    # Ordenar por winrate descendente
    resultados.sort(key=lambda x: x['winrate'], reverse=True)
    if resultados:
        mejor = resultados[0]
        # Construir función para la mejor regla (necesitamos la función real para backtest final)
        # Pero para este resumen, solo enviamos el texto.
        return mejor, resultados[:5]  # top 5
    else:
        return None, []

# =================== 5. REPORTE A TELEGRAM (RESUMEN + GRÁFICOS) ===================
def enviar_resumen_y_graficos(mejor_regla, top_resultados, muestras):
    if mejor_regla:
        msg = f"🏆 *MEJOR REGLA ENCONTRADA*\n{mejor_regla['regla']}\n\n📊 Rendimiento:\n"
        msg += f"Trades: {mejor_regla['trades']}\n"
        msg += f"Winrate: {mejor_regla['winrate']:.1f}%\n"
        msg += f"PnL total: {mejor_regla['pnl_total']:+.2f} USDT\n"
        msg += f"Balance final: {mejor_regla['balance_final']:.2f} USDT\n\n"
        msg += "📈 *TOP 5 REGLAS*:\n"
        for r in top_resultados:
            msg += f"- {r['regla']}: {r['winrate']:.1f}% ({r['trades']} trades)\n"
        send_telegram(msg)
    else:
        send_telegram("❌ No se encontraron combinaciones de reglas con trades.")

    # Enviar hasta 20 imágenes de las muestras (primeras 20)
    for idx, m in enumerate(muestras[:20]):
        caption = f"🔔 Señal #{idx+1}: {m['direction']} {m['cambio']:.1f}%\n{m['timestamp'].strftime('%Y-%m-%d %H:%M')}"
        with open(m['image_path'], 'rb') as f:
            img_buf = BytesIO(f.read())
        send_telegram_image(img_buf, caption)

# =================== MAIN ===================
def main():
    print("🚀 Iniciando pipeline de optimización de reglas...")
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

    # 3. (Opcional) Extraer reglas con IA - Comentar si no se quiere usar
    # extraer_reglas_con_ia(muestras, max_imagenes=20)

    # 4. Evaluar combinaciones de reglas
    mejor_regla, top_resultados = evaluar_reglas(df_10m, df_60m)

    # 5. Enviar a Telegram
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        enviar_resumen_y_graficos(mejor_regla, top_resultados, muestras)
    else:
        print("⚠️ Telegram no configurado. Resultados impresos:")
        if mejor_regla:
            print(mejor_regla)
        else:
            print("No se encontraron reglas con trades.")

    print("✅ Pipeline finalizado")

if __name__ == "__main__":
    main()
