import os
import time
import json
import itertools
import base64
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
from datetime import datetime
from pybit.unified_trading import HTTP
from ta.momentum import RSIIndicator
from ta.trend import MACD
from dotenv import load_dotenv

# =================== CONFIGURACIÓN ===================
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # not used now, but kept
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SYMBOL = "BTCUSDT"
INITIAL_BALANCE = 1000.0
COMMISSION = 0.001
MIN_MOVE_PERCENT = 1.2
LOOKAHEAD_VELAS = 10
# No max trades limit – we want full backtest
RISK_PERCENT = 0.02
MAX_POSITION_PCT = 0.10
MIN_WINRATE = 50.0          # minimum winrate to consider (can be adjusted)
MIN_TRADES = 5              # minimum number of trades required

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

def generar_grafico_equity(trades, equity_curve, titulo):
    """Genera gráfico de equity curve (capital acumulado) con puntos de trades."""
    if len(equity_curve) == 0:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_curve.index, equity_curve.values, color='blue', linewidth=2, label='Equity')
    # marcar winners y losers
    if trades:
        winners = [t for t in trades if t['pnl'] > 0]
        losers = [t for t in trades if t['pnl'] <= 0]
        if winners:
            win_times = [t['exit_time'] for t in winners]
            win_pnls = [t['cum_balance'] for t in winners]
            ax.scatter(win_times, win_pnls, color='green', marker='^', s=50, label='Winner')
        if losers:
            lose_times = [t['exit_time'] for t in losers]
            lose_pnls = [t['cum_balance'] for t in losers]
            ax.scatter(lose_times, lose_pnls, color='red', marker='v', s=50, label='Loser')
    ax.set_title(titulo, color='white')
    ax.set_xlabel('Fecha', color='white')
    ax.set_ylabel('Balance (USDT)', color='white')
    ax.tick_params(colors='white')
    ax.set_facecolor('#121212')
    ax.legend(loc='upper left', framealpha=0.5, facecolor='black', edgecolor='white', labelcolor='white')
    fig.patch.set_facecolor('#121212')
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100)
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
    
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas requeridas: {missing}. Columnas disponibles: {df.columns.tolist()}")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
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

# =================== 3. BACKTEST COMPLETO SIN LÍMITE DE TRADES ===================
def backtest_completo(df, condicion_func):
    """
    Ejecuta backtest sobre todo el DataFrame.
    Retorna: (trades_list, equity_curve_series, final_balance)
    Cada trade guarda: entry_time, entry_price, exit_time, exit_price, pnl, cum_balance, direction (always BUY)
    """
    balance = INITIAL_BALANCE
    in_position = False
    position = None
    trades = []
    equity = [balance]
    timestamps = [df.index[0]]
    balance_history = {df.index[0]: balance}
    
    # Empezamos desde índice 100 para tener suficientes indicadores
    for i in range(100, len(df)):
        current_time = df.index[i]
        if not in_position:
            if condicion_func(df, i):
                entry = df['close'].iloc[i]
                sl = entry * 0.995
                tp1 = entry * 1.005
                # Calcular cantidad basada en riesgo y balance actual
                risk_amount = balance * RISK_PERCENT
                risk_per_share = entry - sl
                if risk_per_share <= 0:
                    continue
                qty = risk_amount / risk_per_share
                max_qty = (balance * MAX_POSITION_PCT) / entry
                qty = min(qty, max_qty)
                if qty < 0.0001:
                    continue
                in_position = True
                position = {
                    'entry': entry,
                    'sl': sl,
                    'tp1': tp1,
                    'qty': qty,
                    'entry_time': current_time,
                    'entry_idx': i
                }
        else:
            current = df['close'].iloc[i]
            exit_reason = None
            exit_price = None
            if current >= position['tp1']:
                exit_price = position['tp1']
                exit_reason = 'TP1'
            elif current <= position['sl']:
                exit_price = position['sl']
                exit_reason = 'SL'
            
            if exit_reason is not None:
                pnl = (exit_price - position['entry']) * position['qty']
                pnl -= abs(pnl) * COMMISSION
                balance += pnl
                trades.append({
                    'entry_time': position['entry_time'],
                    'entry_price': position['entry'],
                    'exit_time': current_time,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'cum_balance': balance,
                    'exit_reason': exit_reason
                })
                balance_history[current_time] = balance
                in_position = False
    
    # Construir equity curve con todos los tiempos (rellenar forward)
    all_times = df.index[df.index >= df.index[100]]  # desde donde empezamos a operar
    equity_curve = pd.Series(index=all_times, dtype=float)
    last_balance = INITIAL_BALANCE
    last_time = all_times[0]
    for t in all_times:
        if t in balance_history:
            last_balance = balance_history[t]
        equity_curve[t] = last_balance
    # Agregar el balance final si no está ya
    if len(trades) > 0 and trades[-1]['exit_time'] not in balance_history:
        equity_curve[trades[-1]['exit_time']] = trades[-1]['cum_balance']
    
    return trades, equity_curve, balance

def evaluar_todas_combinaciones(df):
    """Genera todas las combinaciones de 2,3,4 condiciones y devuelve lista ordenada de resultados."""
    cond_nombres = list(CONDICIONES_ATOMICAS.keys())
    resultados = []
    total_combinaciones = 0
    for r in range(2, 5):
        total_combinaciones += len(list(itertools.combinations(cond_nombres, r)))
    log(f"Evaluando {total_combinaciones} combinaciones...")
    
    for r in range(2, 5):
        for combo_nombres in itertools.combinations(cond_nombres, r):
            combo_funcs = [CONDICIONES_ATOMICAS[n] for n in combo_nombres]
            regla_func = lambda df, i, fns=combo_funcs: all(f(df, i) for f in fns)
            trades, equity, final_balance = backtest_completo(df, regla_func)
            if len(trades) < MIN_TRADES:
                continue
            wins = sum(1 for t in trades if t['pnl'] > 0)
            winrate = wins / len(trades) * 100
            if winrate < MIN_WINRATE:
                continue
            total_pnl = sum(t['pnl'] for t in trades)
            gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
            gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else float('inf')
            # Max drawdown from equity curve
            peak = equity.cummax()
            drawdown = (peak - equity) / peak * 100
            max_drawdown = drawdown.max()
            resultados.append({
                'regla': ' AND '.join(combo_nombres),
                'trades': len(trades),
                'wins': wins,
                'losses': len(trades) - wins,
                'winrate': winrate,
                'total_pnl': total_pnl,
                'final_balance': final_balance,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'equity_curve': equity,
                'trades_list': trades
            })
    # Ordenar por winrate descendente, luego profit factor
    resultados.sort(key=lambda x: (x['winrate'], x['profit_factor']), reverse=True)
    return resultados

# =================== 4. ENVÍO DE RESULTADOS A TELEGRAM ===================
def enviar_mejor_setup(mejor):
    if mejor is None:
        send_telegram("❌ No se encontró ningún setup que cumpla con los requisitos mínimos (trades >= {MIN_TRADES}, winrate >= {MIN_WINRATE}%).")
        return
    
    # Mensaje de texto detallado
    msg = f"🏆 *MEJOR SETUP ENCONTRADO* 🏆\n\n"
    msg += f"📌 *Regla:* `{mejor['regla']}`\n"
    msg += f"📊 *Trades totales:* {mejor['trades']}\n"
    msg += f"✅ *Ganadores:* {mejor['wins']}   ❌ *Perdedores:* {mejor['losses']}\n"
    msg += f"📈 *Winrate:* {mejor['winrate']:.2f}%\n"
    msg += f"💰 *Profit total:* {mejor['total_pnl']:+.2f} USDT\n"
    msg += f"⚖️ *Profit Factor:* {mejor['profit_factor']:.2f}\n"
    msg += f"📉 *Máximo Drawdown:* {mejor['max_drawdown']:.2f}%\n"
    msg += f"💼 *Balance final:* {mejor['final_balance']:.2f} USDT (desde {INITIAL_BALANCE} USDT)"
    send_telegram(msg)
    
    # Enviar equity curve
    if mejor['equity_curve'] is not None and not mejor['equity_curve'].empty:
        img = generar_grafico_equity(mejor['trades_list'], mejor['equity_curve'], f"Equity Curve: {mejor['regla'][:60]}")
        if img:
            send_telegram_image(img, caption=f"Evolución del balance - Winrate {mejor['winrate']:.1f}%")
    
    # Enviar lista de trades (ganadores y perdedores) en lotes
    trades = mejor['trades_list']
    if trades:
        # Resumen de ganadores y perdedores
        winners = [t for t in trades if t['pnl'] > 0]
        losers = [t for t in trades if t['pnl'] <= 0]
        trade_msg = f"📋 *Detalle de trades:*\n"
        trade_msg += f"Ganadores: {len(winners)}  |  Perdedores: {len(losers)}\n\n"
        # Mostrar primeros 10 de cada tipo para no exceder límite
        if winners:
            trade_msg += "*Últimos ganadores:*\n"
            for w in winners[-5:]:
                trade_msg += f" +{w['pnl']:.2f} USDT ({w['exit_reason']}) en {w['exit_time'].strftime('%m-%d %H:%M')}\n"
        if losers:
            trade_msg += "\n*Últimos perdedores:*\n"
            for l in losers[-5:]:
                trade_msg += f" {l['pnl']:.2f} USDT ({l['exit_reason']}) en {l['exit_time'].strftime('%m-%d %H:%M')}\n"
        send_telegram(trade_msg[:4000])
        
        # Opcional: enviar todos los trades en un CSV? No necesario aquí.

# =================== 5. EJECUCIÓN PRINCIPAL ===================
def main():
    log("🚀 Iniciando evaluación de todas las combinaciones de condiciones (sin IA)")
    df = cargar_datos()
    df = calcular_indicadores(df)
    
    resultados = evaluar_todas_combinaciones(df)
    if not resultados:
        log("No se encontró ningún setup válido.")
        send_telegram("⚠️ No se encontró ningún setup con winrate >= {}% y al menos {} trades.".format(MIN_WINRATE, MIN_TRADES))
        return
    
    mejor = resultados[0]
    log(f"✅ Mejor setup encontrado: {mejor['regla']} -> winrate {mejor['winrate']:.2f}% ({mejor['trades']} trades)")
    enviar_mejor_setup(mejor)
    
    # Opcional: enviar también los top 5? Para no saturar, solo mejor.
    log("✅ Proceso finalizado")

if __name__ == "__main__":
    main()
