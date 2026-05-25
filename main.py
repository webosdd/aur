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
import mplfinance as mpf  # PARA GRÁFICOS DE VELAS JAPONESAS

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
MAX_TRADES_BACKTEST = 50
RISK_PERCENT = 0.02
MAX_POSITION_PCT = 0.10
MIN_WINRATE = 55.0          # Mínimo winrate para considerar un setup
MIN_TRADES = 10             # Mínimo número de trades
MAX_ITERATIONS = 30         # Máximo de iteraciones
DESIRED_SETUPS = 20         # Buscar hasta 20 setups

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

def generar_grafico_velas(df, titulo, soporte=None, resistencia=None, entry=None, sl=None, tp1=None):
    """
    Genera gráfico de velas japonesas con EMAs, soporte/resistencia, RSI y MACD.
    Retorna un objeto BytesIO con la imagen PNG.
    """
    # Preparar datos para mplfinance
    df_plot = df[['open', 'high', 'low', 'close', 'volume']].copy()
    df_plot.index = pd.to_datetime(df_plot.index)
    df_plot.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Añadir EMAs
    ema20 = df['ema20'].values
    ema50 = df['ema50'].values
    
    # Crear panel adicional para RSI y MACD
    apds = []
    # RSI
    if 'rsi' in df:
        rsi = df['rsi'].values
        apds.append(mpf.make_addplot(rsi, panel=1, color='purple', ylabel='RSI', ylim=(0,100)))
        # Líneas de sobrecompra/sobreventa
        apds.append(mpf.make_addplot([70]*len(df), panel=1, color='red', linestyle='--', secondary_y=False))
        apds.append(mpf.make_addplot([30]*len(df), panel=1, color='green', linestyle='--', secondary_y=False))
    # MACD
    if 'macd' in df:
        macd_line = df['macd'].values
        signal_line = df['macd_signal'].values
        histogram = df['macd_diff'].values
        apds.append(mpf.make_addplot(macd_line, panel=2, color='blue', ylabel='MACD'))
        apds.append(mpf.make_addplot(signal_line, panel=2, color='red'))
        apds.append(mpf.make_addplot(histogram, panel=2, type='bar', color='gray', alpha=0.5))
    
    # Líneas de soporte y resistencia
    if soporte:
        apds.append(mpf.make_addplot([soporte]*len(df), panel=0, color='cyan', linestyle='--', linewidth=1.5))
    if resistencia:
        apds.append(mpf.make_addplot([resistencia]*len(df), panel=0, color='magenta', linestyle='--', linewidth=1.5))
    
    # Líneas de entrada, SL, TP1
    if entry:
        apds.append(mpf.make_addplot([entry]*len(df), panel=0, color='orange', linestyle=':', linewidth=1.5))
    if sl:
        apds.append(mpf.make_addplot([sl]*len(df), panel=0, color='red', linestyle='--', linewidth=1.5))
    if tp1:
        apds.append(mpf.make_addplot([tp1]*len(df), panel=0, color='lime', linestyle='--', linewidth=1.5))
    
    # Configurar estilo
    mc = mpf.make_marketcolors(up='#00ff00', down='#ff0000', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridcolor='gray', facecolor='#121212', edgecolor='white')
    
    # Crear figura
    fig, axes = mpf.plot(df_plot,
                         type='candle',
                         style=s,
                         addplot=apds,
                         volume=False,
                         title=titulo,
                         ylabel='Price (USDT)',
                         figsize=(16, 12),
                         panel_ratios=(3, 1, 1),
                         returnfig=True)
    
    # Guardar a BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    plt.close(fig)
    return buf

# =================== 1. CARGAR DATOS DESDE GITHUB ===================
def cargar_datos():
    log(f"📥 Descargando datos desde: {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL, compression='zip')
    except Exception as e:
        raise Exception(f"Error al cargar el archivo: {e}. Verifica la URL y que el archivo sea un ZIP con un CSV dentro.")
    
    # Renombrar columnas al formato esperado
    rename_map = {
        'Open time': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    
    # Verificar columnas necesarias
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}. Columnas disponibles: {df.columns.tolist()}")
    
    # Convertir timestamp y establecer índice
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Asegurar valores numéricos
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.sort_index()
    log(f"✅ Datos cargados: {len(df)} velas desde {df.index[0]} hasta {df.index[-1]}")
    return df

# =================== 2. CONDICIONES ATÓMICAS (predefinidas) ===================
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
    def volume_spike(df, idx):
        avg_vol = df['volume'].iloc[max(0,idx-20):idx].mean()
        return df['volume'].iloc[idx] > avg_vol * 1.5
    def close_up_prev(df, idx):
        return df['close'].iloc[idx] > df['close'].iloc[idx-1]
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
        "volume_spike": volume_spike,
        "close_up_prev": close_up_prev,
    }
definir_condiciones_atomicas()

# =================== 3. BACKTEST CON UNA REGLA ===================
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

def evaluar_lista_condiciones(df, lista_nombres):
    """Prueba todas las combinaciones de 2, 3 y 4 condiciones en AND y devuelve la mejor."""
    if not lista_nombres:
        return None
    cond_funcs = [(nombre, CONDICIONES_ATOMICAS[nombre]) for nombre in lista_nombres if nombre in CONDICIONES_ATOMICAS]
    if len(cond_funcs) < 2:
        return None
    resultados = []
    # Combinaciones de 2,3,4
    for r in range(2, min(5, len(cond_funcs)+1)):
        for combo in itertools.combinations(cond_funcs, r):
            nombres = [c[0] for c in combo]
            regla = lambda df, i, fns=combo: all(f(df, i) for _, f in fns)
            trades, balance_final = backtest_con_regla(df, regla)
            if not trades:
                continue
            pnl_total = sum(t['pnl'] for t in trades)
            wins = sum(1 for t in trades if t['pnl'] > 0)
            winrate = wins / len(trades) * 100
            resultados.append({
                'regla': ' AND '.join(nombres),
                'trades': len(trades),
                'winrate': winrate,
                'pnl_total': pnl_total,
                'balance_final': balance_final,
                'combo': combo
            })
    if not resultados:
        return None
    # Filtrar por winrate mínimo y número mínimo de trades
    resultados = [r for r in resultados if r['winrate'] >= MIN_WINRATE and r['trades'] >= MIN_TRADES]
    if not resultados:
        return None
    resultados.sort(key=lambda x: (x['winrate'], x['pnl_total']), reverse=True)
    return resultados[0]

# =================== 4. IA PARA SUGERIR NUEVAS CONDICIONES ===================
def obtener_sugerencias_ia(historial_setups, iteration):
    if not OPENROUTER_API_KEY:
        return None
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Setups Optimizer"}
    )
    # Construir prompt con historial
    historial_texto = ""
    if historial_setups:
        historial_texto = "Historial de setups encontrados hasta ahora:\n"
        for i, s in enumerate(historial_setups[-5:], 1):
            historial_texto += f"{i}. {s['regla']} -> winrate {s['winrate']:.1f}% ({s['trades']} trades)\n"
    else:
        historial_texto = "Aún no se ha encontrado ningún setup rentable.\n"
    
    prompt = f"""
Eres un experto en trading algorítmico. Estás ayudando a optimizar un bot de trading para BTC/USDT en temporalidad 15 minutos.
Hasta ahora hemos probado combinaciones de las siguientes condiciones atómicas:
{list(CONDICIONES_ATOMICAS.keys())}

{historial_texto}

Basándote en el conocimiento del mercado, sugiere **nuevas condiciones atómicas** (entre 3 y 6) que podrían mejorar el rendimiento. Las condiciones deben ser objetivas y programables, usando datos de velas (open, high, low, close, volume, EMAs, RSI, MACD, etc.). Cada condición debe ser una expresión corta, como por ejemplo: "close < low.shift(1)" o "high - close < close * 0.002".

Responde ÚNICAMENTE con un JSON que contenga un array de strings, sin texto adicional. Ejemplo:
["condicion1", "condicion2", "condicion3"]
"""
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-sonnet-4.5",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content)
        if isinstance(data, list):
            return data
        else:
            return None
    except Exception as e:
        log(f"Error llamando a IA: {e}")
        return None

def agregar_nuevas_condiciones(nuevas_condiciones):
    """Agrega nuevas condiciones atómicas al diccionario global, convirtiendo las expresiones en funciones."""
    global CONDICIONES_ATOMICAS
    for expr in nuevas_condiciones:
        # Evitar duplicados
        if expr in CONDICIONES_ATOMICAS:
            continue
        # Crear una función lambda que evalúe la expresión usando pandas
        # ADVERTENCIA: eval es peligroso, pero aquí solo se usan expresiones de pandas con seguridad relativa
        # Limitamos los símbolos permitidos para minimizar riesgos
        try:
            # Compilar la expresión en una función que tome df e idx
            # Permitimos acceso a variables: df, idx, y funciones de pandas/numpy
            # Usamos un entorno de evaluación restringido
            allowed_globals = {
                'df': pd.DataFrame,
                'pd': pd,
                'np': np,
                'abs': abs,
                'max': max,
                'min': min,
            }
            # Compilar la condición
            code = compile(expr, '<string>', 'eval')
            def cond_func(df, idx):
                # El contexto local incluye df y idx
                local_env = {'df': df, 'idx': idx, 'pd': pd, 'np': np}
                return eval(code, allowed_globals, local_env)
            # Probar la función con un índice válido para verificar que no de error
            cond_func(None, 0)  # Solo para comprobar sintaxis; luego se usará con datos reales
            CONDICIONES_ATOMICAS[expr] = cond_func
            log(f"➕ Nueva condición añadida: {expr}")
        except Exception as e:
            log(f"❌ No se pudo añadir condición '{expr}': {e}")

# =================== 5. EJECUCIÓN PRINCIPAL ITERATIVA ===================
def main():
    log("🚀 Iniciando optimización iterativa de setups con IA")
    df = cargar_datos()
    df = calcular_indicadores(df)
    
    setups_encontrados = []
    iteration = 0
    while len(setups_encontrados) < DESIRED_SETUPS and iteration < MAX_ITERATIONS:
        iteration += 1
        log(f"\n🔁 Iteración {iteration}/{MAX_ITERATIONS} - Setups encontrados: {len(setups_encontrados)}/{DESIRED_SETUPS}")
        
        # Obtener lista actual de condiciones (todas las existentes)
        condiciones_actuales = list(CONDICIONES_ATOMICAS.keys())
        log(f"Probando {len(condiciones_actuales)} condiciones...")
        mejor = evaluar_lista_condiciones(df, condiciones_actuales)
        
        if mejor:
            log(f"✅ Mejor regla encontrada: {mejor['regla']} -> winrate {mejor['winrate']:.1f}% ({mejor['trades']} trades)")
            # Enviar a Telegram el setup encontrado con su gráfico
            # Para el gráfico, necesitamos un ejemplo de señal. Buscamos el primer evento de compra que cumple la regla.
            # Para simplificar, usamos la regla para encontrar un índice donde se cumple.
            # Construimos la función de regla a partir de la combinación
            combo = mejor['combo']
            regla_func = lambda df, i: all(f(df, i) for _, f in combo)
            # Buscar un índice donde se cumpla la regla y haya una entrada en el backtest (usamos el primer índice después de 100)
            sample_idx = None
            for i in range(100, len(df)):
                if regla_func(df, i):
                    sample_idx = i
                    break
            if sample_idx:
                # Obtener soporte y resistencia en ese momento (ventana de 20 velas)
                sop = df['low'].iloc[max(0,sample_idx-20):sample_idx+1].min()
                res = df['high'].iloc[max(0,sample_idx-20):sample_idx+1].max()
                entry = df['close'].iloc[sample_idx]
                sl = entry * 0.995
                tp1 = entry * 1.005
                # Ventana de 80 velas alrededor
                start = max(0, sample_idx - 80)
                end = min(len(df), sample_idx + 20)
                df_window = df.iloc[start:end].copy()
                img_buf = generar_grafico_velas(df_window, f"Iter {iteration}: {mejor['regla'][:50]}", sop, res, entry, sl, tp1)
                caption = f"🔔 *Setup #{len(setups_encontrados)+1}* (Iter {iteration})\nRegla: `{mejor['regla']}`\nWinrate: {mejor['winrate']:.1f}% ({mejor['trades']} trades)\nPnL: {mejor['pnl_total']:+.2f} USDT"
                send_telegram_image(img_buf, caption)
            else:
                send_telegram(f"🔔 *Setup #{len(setups_encontrados)+1}* (Iter {iteration})\nRegla: {mejor['regla']}\nWinrate: {mejor['winrate']:.1f}% ({mejor['trades']} trades)\nPnL: {mejor['pnl_total']:+.2f} USDT")
            setups_encontrados.append(mejor)
        else:
            log("⚠️ No se encontró ningún setup con winrate >= {}% y al menos {} trades.".format(MIN_WINRATE, MIN_TRADES))
            send_telegram(f"⚠️ Iteración {iteration}: no se encontró setup rentable. Se pedirán nuevas condiciones a la IA.")
        
        # Pedir a la IA nuevas condiciones (si no hemos alcanzado el objetivo)
        if len(setups_encontrados) < DESIRED_SETUPS:
            nuevas = obtener_sugerencias_ia(setups_encontrados, iteration)
            if nuevas and isinstance(nuevas, list):
                agregar_nuevas_condiciones(nuevas)
            else:
                log("No se recibieron sugerencias de IA o no son válidas.")
        
        time.sleep(2)  # Pequeña pausa entre iteraciones
    
    # Resumen final
    if setups_encontrados:
        resumen = "🏆 *RESUMEN DE SETUPS RENTABLES ENCONTRADOS*\n\n"
        for i, s in enumerate(setups_encontrados, 1):
            resumen += f"{i}. {s['regla']}\n   Winrate: {s['winrate']:.1f}% ({s['trades']} trades) | PnL: {s['pnl_total']:+.2f} USDT\n\n"
        send_telegram(resumen)
    else:
        send_telegram("❌ No se encontró ningún setup rentable después de las iteraciones.")
    
    log("✅ Proceso finalizado")

if __name__ == "__main__":
    main()
