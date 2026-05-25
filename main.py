import os
import time
import json
import itertools
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from io import BytesIO
from datetime import datetime
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
RISK_PERCENT = 0.02
MAX_POSITION_PCT = 0.10
MIN_WINRATE = 50.0          # mínimo winrate para considerar una regla
MIN_TRADES = 5              # mínimo número de trades
MAX_ITERATIONS = 5          # iteraciones de mejora con IA
DESIRED_SETUPS = 1          # solo nos interesa la mejor regla final

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
    if len(equity_curve) == 0:
        return None
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(equity_curve.index, equity_curve.values, color='blue', linewidth=2, label='Equity')
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

# =================== CARGAR DATOS ===================
def cargar_datos():
    log(f"📥 Descargando datos desde: {DATA_URL}")
    try:
        df = pd.read_csv(DATA_URL, compression='zip')
    except Exception as e:
        raise Exception(f"Error al cargar el archivo: {e}")
    rename_map = {
        'Open time': 'timestamp', 'Open': 'open', 'High': 'high',
        'Low': 'low', 'Close': 'close', 'Volume': 'volume'
    }
    for old, new in rename_map.items():
        if old in df.columns:
            df.rename(columns={old: new}, inplace=True)
    required = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Faltan columnas: {missing}")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.sort_index()
    log(f"✅ Datos cargados: {len(df)} velas")
    return df

# =================== CONDICIONES ATÓMICAS INICIALES ===================
CONDICIONES_ATOMICAS = {}

def definir_condiciones_iniciales():
    global CONDICIONES_ATOMICAS
    def rsi_lt_30(df, idx): return df['rsi'].iloc[idx] < 30
    def rsi_gt_70(df, idx): return df['rsi'].iloc[idx] > 70
    def price_lt_ema20(df, idx): return df['close'].iloc[idx] < df['ema20'].iloc[idx]
    def price_gt_ema20(df, idx): return df['close'].iloc[idx] > df['ema20'].iloc[idx]
    def macd_cross_above(df, idx):
        return (df['macd'].iloc[idx] > df['macd_signal'].iloc[idx]) and (df['macd'].iloc[idx-1] <= df['macd_signal'].iloc[idx-1])
    def macd_cross_below(df, idx):
        return (df['macd'].iloc[idx] < df['macd_signal'].iloc[idx]) and (df['macd'].iloc[idx-1] >= df['macd_signal'].iloc[idx-1])
    def close_near_support(df, idx):
        soporte = df['low'].iloc[max(0,idx-20):idx+1].min()
        return abs(df['close'].iloc[idx] - soporte) / df['close'].iloc[idx] < 0.002
    def close_near_resistance(df, idx):
        resistencia = df['high'].iloc[max(0,idx-20):idx+1].max()
        return abs(resistencia - df['close'].iloc[idx]) / df['close'].iloc[idx] < 0.002
    def price_above_ema50(df, idx): return df['close'].iloc[idx] > df['ema50'].iloc[idx]
    def price_below_ema50(df, idx): return df['close'].iloc[idx] < df['ema50'].iloc[idx]
    def volume_spike(df, idx):
        avg_vol = df['volume'].iloc[max(0,idx-20):idx].mean()
        return df['volume'].iloc[idx] > avg_vol * 1.5
    def close_up_prev(df, idx): return df['close'].iloc[idx] > df['close'].iloc[idx-1]
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
definir_condiciones_iniciales()

# =================== BACKTEST COMPLETO ===================
def backtest_completo(df, condicion_func):
    balance = INITIAL_BALANCE
    in_position = False
    position = None
    trades = []
    balance_history = {}
    for i in range(100, len(df)):
        current_time = df.index[i]
        if not in_position:
            if condicion_func(df, i):
                entry = df['close'].iloc[i]
                sl = entry * 0.995
                tp1 = entry * 1.005
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
                    'entry': entry, 'sl': sl, 'tp1': tp1, 'qty': qty,
                    'entry_time': current_time
                }
        else:
            current = df['close'].iloc[i]
            exit_price = None
            exit_reason = None
            if current >= position['tp1']:
                exit_price = position['tp1']
                exit_reason = 'TP1'
            elif current <= position['sl']:
                exit_price = position['sl']
                exit_reason = 'SL'
            if exit_price is not None:
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
    # Construir equity curve
    all_times = df.index[df.index >= df.index[100]]
    equity_curve = pd.Series(index=all_times, dtype=float)
    last_balance = INITIAL_BALANCE
    for t in all_times:
        if t in balance_history:
            last_balance = balance_history[t]
        equity_curve[t] = last_balance
    return trades, equity_curve, balance

def evaluar_todas_combinaciones(df):
    cond_nombres = list(CONDICIONES_ATOMICAS.keys())
    resultados = []
    total_combos = 0
    for r in range(2, 5):
        total_combos += len(list(itertools.combinations(cond_nombres, r)))
    log(f"Evaluando {total_combos} combinaciones...")
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
    resultados.sort(key=lambda x: (x['winrate'], x['profit_factor']), reverse=True)
    return resultados

# =================== IA PARA SUGERIR NUEVAS CONDICIONES ===================
def sugerir_nuevas_condiciones_ia(mejor_setup, iteration):
    if not OPENROUTER_API_KEY:
        log("No hay API key de OpenRouter")
        return []
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Setups Optimizer"}
    )
    # Tomar una muestra de los últimos 5 trades perdedores (si existen)
    trades = mejor_setup['trades_list']
    losers = [t for t in trades if t['pnl'] < 0]
    losers_sample = losers[-5:] if len(losers) > 0 else []
    losers_text = "\n".join([f"  - {l['exit_time']}: PnL {l['pnl']:.2f}, exit {l['exit_reason']}" for l in losers_sample]) if losers_sample else "No hubo trades perdedores."
    
    prompt = f"""
Eres un experto en trading algorítmico para BTC/USDT en 15 minutos.

Hemos encontrado una regla que funciona bastante bien:
Regla: {mejor_setup['regla']}
Winrate: {mejor_setup['winrate']:.1f}% ({mejor_setup['trades']} trades)
Profit factor: {mejor_setup['profit_factor']:.2f}
Máximo drawdown: {mejor_setup['max_drawdown']:.1f}%

Los últimos trades perdedores (hasta 5):
{losers_text}

Basándote en este análisis, sugiere **3 nuevas condiciones atómicas** (expresiones en Python/pandas) que podrían filtrar esos perdedores y aumentar el winrate.
Cada condición debe ser una expresión que use `df` (DataFrame) y `idx` (índice de la vela actual). Ejemplos válidos:
- "df['close'].iloc[idx] < df['ema20'].iloc[idx]"
- "df['volume'].iloc[idx] > df['volume'].rolling(20).mean().iloc[idx] * 1.5"
- "(df['high'].iloc[idx] - df['low'].iloc[idx]) / df['close'].iloc[idx] > 0.01"

Responde ÚNICAMENTE con un JSON array de strings, sin texto adicional.
Ejemplo: ["cond1", "cond2", "cond3"]
"""
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp",  # modelo rápido y barato
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        if isinstance(data, list):
            return data
        else:
            return []
    except Exception as e:
        log(f"Error en IA: {e}")
        return []

def agregar_condicion_si_valida(expr):
    global CONDICIONES_ATOMICAS
    if expr in CONDICIONES_ATOMICAS:
        return False
    # Validar sintaxis básica
    try:
        compile(expr, '<string>', 'eval')
        # Crear función dinámica
        def nueva_cond(df, idx, expr=expr):
            return eval(expr, globals(), {'df': df, 'idx': idx, 'pd': pd, 'np': np})
        # Probar con un índice cualquiera (e.g., 150) para ver si no da error de ejecución
        # No lo hacemos aquí porque requiere df real; se probará en backtest.
        CONDICIONES_ATOMICAS[expr] = nueva_cond
        log(f"➕ Nueva condición añadida: {expr}")
        return True
    except Exception as e:
        log(f"❌ Condición inválida '{expr}': {e}")
        return False

# =================== EJECUCIÓN PRINCIPAL ITERATIVA ===================
def main():
    log("🚀 Iniciando optimización con IA (Gemini)")
    df = cargar_datos()
    df = calcular_indicadores(df)
    
    best_overall = None
    for iteration in range(1, MAX_ITERATIONS + 1):
        log(f"\n🔁 Iteración {iteration}/{MAX_ITERATIONS}")
        resultados = evaluar_todas_combinaciones(df)
        if not resultados:
            log("No hay combinaciones que cumplan los requisitos mínimos.")
            break
        mejor = resultados[0]
        log(f"✅ Mejor regla: {mejor['regla']} -> winrate {mejor['winrate']:.1f}% ({mejor['trades']} trades)")
        
        # Actualizar best_overall si es mejor
        if best_overall is None or (mejor['winrate'] > best_overall['winrate']):
            best_overall = mejor
            # Enviar a Telegram el nuevo mejor encontrado
            msg = f"🔔 *Iter {iteration} - Nuevo mejor setup*\nRegla: `{mejor['regla']}`\nWinrate: {mejor['winrate']:.1f}% ({mejor['trades']} trades)\nProfit factor: {mejor['profit_factor']:.2f}\nPnL: {mejor['total_pnl']:+.2f} USDT"
            send_telegram(msg)
            # Enviar equity curve
            img = generar_grafico_equity(mejor['trades_list'], mejor['equity_curve'], f"Equity Curve - {mejor['regla'][:60]}")
            if img:
                send_telegram_image(img, caption=f"Evolución del balance (winrate {mejor['winrate']:.1f}%)")
        
        # Si ya tenemos un winrate muy alto, podemos parar
        if mejor['winrate'] >= 80:
            log("Winrate >= 80%, deteniendo iteraciones.")
            break
        
        # Pedir a la IA nuevas condiciones
        log("Consultando IA para sugerir nuevas condiciones...")
        nuevas = sugerir_nuevas_condiciones_ia(mejor, iteration)
        if nuevas:
            any_added = False
            for expr in nuevas:
                if agregar_condicion_si_valida(expr):
                    any_added = True
            if any_added:
                log(f"Se añadieron {len([e for e in nuevas if e in CONDICIONES_ATOMICAS])} nuevas condiciones. Re-evaluando...")
            else:
                log("No se pudo añadir ninguna condición válida.")
        else:
            log("IA no devolvió sugerencias o hubo error.")
        
        time.sleep(2)  # pausa para no saturar la API
    
    # Resumen final
    if best_overall:
        final_msg = f"🏆 *MEJOR SETUP FINAL* 🏆\n\nRegla: `{best_overall['regla']}`\nWinrate: {best_overall['winrate']:.1f}% ({best_overall['trades']} trades)\nProfit factor: {best_overall['profit_factor']:.2f}\nDrawdown máx: {best_overall['max_drawdown']:.1f}%\nPnL total: {best_overall['total_pnl']:+.2f} USDT\nBalance final: {best_overall['final_balance']:.2f} USDT"
        send_telegram(final_msg)
        # Enviar trade list resumido
        trade_details = "📋 *Detalle de trades (últimos 10):*\n"
        for t in best_overall['trades_list'][-10:]:
            trade_details += f"{t['exit_time'].strftime('%m-%d %H:%M')} -> {t['pnl']:+.2f} USDT ({t['exit_reason']})\n"
        send_telegram(trade_details)
    else:
        send_telegram("❌ No se encontró ningún setup válido.")
    
    log("✅ Proceso finalizado")

if __name__ == "__main__":
    main()
