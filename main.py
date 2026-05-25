# BOT TRADING CON CLAUDE SONNET 4.5 - IA CON LIBERTAD TOTAL
# Temporalidades: 10m (ejecución) y 1h (tendencia)
# ==============================================================================
import os, time, requests, json, numpy as np, pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import io
import json_repair
import base64
from openai import OpenAI
import hashlib
import hmac
import logging
import re
from ta.momentum import RSIIndicator
from ta.trend import MACD

# =================== LOGGING ===================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("TradingBot")

# =================== SANITIZAR EMOJIS ===================
def sanitize_for_matplotlib(text):
    if not isinstance(text, str):
        return text
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F700-\U0001F77F"
        u"\U0001F780-\U0001F7FF"
        u"\U0001F800-\U0001F8FF"
        u"\U0001F900-\U0001F9FF"
        u"\U0001FA00-\U0001FA6F"
        u"\U0001FA70-\U0001FAFF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub('', text)

# =================== VARIABLES DE ENTORNO ===================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Falta OPENROUTER_API_KEY")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY", "dummy")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET", "dummy")

BASE_URL = "https://api.bybit.com"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Trading Bot"}
)
MODELO_VISION = "anthropic/claude-sonnet-4.5"

# =================== CONFIGURACIÓN ===================
PAPER_TRADE = True   # Cambiar a False para real
SYMBOL = "BTCUSDT"
INTERVAL_LTF = "10"   # 10 minutos
INTERVAL_HTF = "60"   # 1 hora
SLEEP_SECONDS = 120   # 2 minutos entre ciclos
GRAFICO_VELAS_LIMIT = 80  # 80 velas de 10m = ~13 horas
MAX_CONCURRENT_TRADES = 3
MIN_MARGIN_PER_TRADE = 3.0
TP1_PERCENT = 0.5          # 50% del tamaño en TP1
TRAILING_PERCENT = 0.0015  # 0.15% trailing
LEVERAGE = 34

# Estado paper
paper_balance = 1000.0
paper_positions = {}
paper_trade_counter = 0
paper_win_count = 0
paper_loss_count = 0
paper_total_trades = 0
paper_trade_history = []
PAPER_COMMISSION_PCT = 0.001

# Estado real
REAL_BALANCE = None
REAL_ACTIVE_TRADES = {}
TRADE_COUNTER = 0
WIN_COUNT = 0
LOSS_COUNT = 0
TOTAL_TRADES = 0
TRADE_HISTORY = []

# =================== FUNCIONES BYBIT ===================
def bybit_request(endpoint, method="GET", params=None, body=None):
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    query_string = ""
    if params:
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    if body:
        body_str = json.dumps(body)
        payload = timestamp + BYBIT_API_KEY + recv_window + body_str
    else:
        payload = timestamp + BYBIT_API_KEY + recv_window + query_string
    signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json"
    }
    url = f"{BASE_URL}{endpoint}"
    if method == "GET":
        resp = requests.get(url, headers=headers, params=params)
    else:
        resp = requests.post(url, headers=headers, json=body)
    return resp.json()

def set_leverage():
    if PAPER_TRADE:
        return
    try:
        body = {"category": "linear", "symbol": "BTCUSDT", "buyLeverage": str(LEVERAGE), "sellLeverage": str(LEVERAGE)}
        bybit_request("/v5/position/set-leverage", method="POST", body=body)
        logger.info("Apalancamiento configurado")
    except Exception as e:
        logger.error(f"Error leverage: {e}")

def get_real_balance():
    if PAPER_TRADE:
        return paper_balance
    try:
        params = {"accountType": "UNIFIED", "coin": "USDT"}
        result = bybit_request("/v5/account/wallet-balance", method="GET", params=params)
        return float(result['result']['list'][0]['coin'][0]['walletBalance'])
    except Exception as e:
        logger.error(f"Error balance: {e}")
        return None

def get_free_margin():
    if PAPER_TRADE:
        margin_used = sum((t['qty_original'] * t['entrada']) / LEVERAGE for t in paper_positions.values())
        return max(0.0, paper_balance - margin_used)
    try:
        params = {"accountType": "UNIFIED"}
        result = bybit_request("/v5/account/wallet-balance", method="GET", params=params)
        if result.get('retCode') == 0:
            for coin in result['result']['list'][0]['coin']:
                if coin['coin'] == 'USDT':
                    wallet = float(coin['walletBalance'])
                    used = float(coin.get('usedMargin', 0))
                    return wallet - used
    except Exception as e:
        logger.error(f"Error free margin: {e}")
    return 0.0

def get_real_position_size():
    if PAPER_TRADE:
        return sum(t['qty_restante'] for t in paper_positions.values())
    try:
        params = {"category": "linear", "symbol": "BTCUSDT"}
        result = bybit_request("/v5/position/list", method="GET", params=params)
        if result.get('retCode') == 0:
            for pos in result['result']['list']:
                if pos['symbol'] == "BTCUSDT":
                    return abs(float(pos['size']))
        return 0.0
    except Exception as e:
        logger.error(f"Error position size: {e}")
        return 0.0

def place_market_order(side, qty):
    if PAPER_TRADE:
        logger.info(f"PAPER: Orden {side} {qty} BTC")
        return f"paper_{int(time.time())}"
    try:
        body = {"category": "linear", "symbol": "BTCUSDT", "side": side.capitalize(), "orderType": "Market", "qty": str(qty), "timeInForce": "GTC"}
        result = bybit_request("/v5/order/create", method="POST", body=body)
        if result.get('retCode') == 0:
            return result['result']['orderId']
        else:
            logger.error(f"Error orden: {result}")
            return None
    except Exception as e:
        logger.error(f"Excepción order: {e}")
        return None

def close_position_qty(qty, side_to_close):
    if PAPER_TRADE:
        logger.info(f"PAPER: Cierre {qty} BTC")
        return f"paper_close_{int(time.time())}"
    try:
        real_size = get_real_position_size()
        if real_size <= 0:
            return "already_closed"
        qty_to_close = min(qty, real_size)
        if qty_to_close < 0.001:
            return "already_closed"
        close_side = "Sell" if side_to_close == "Buy" else "Buy"
        body = {"category": "linear", "symbol": "BTCUSDT", "side": close_side, "orderType": "Market", "qty": str(round(qty_to_close, 3)), "timeInForce": "GTC", "reduceOnly": True}
        result = bybit_request("/v5/order/create", method="POST", body=body)
        if result.get('retCode') == 0:
            return result['result']['orderId']
        else:
            logger.error(f"Error cierre: {result}")
            return None
    except Exception as e:
        logger.error(f"Excepción cierre: {e}")
        return None

def close_position_qty_confirm(qty, side_to_close, max_wait=5):
    if PAPER_TRADE:
        return f"confirm_{int(time.time())}"
    size_before = get_real_position_size()
    if size_before <= 0:
        return "already_closed"
    qty_to_close = min(qty, size_before)
    if qty_to_close < 0.001:
        return "already_closed"
    order_id = close_position_qty(qty_to_close, side_to_close)
    if not order_id or order_id == "already_closed":
        return None
    for _ in range(max_wait * 2):
        time.sleep(0.5)
        size_after = get_real_position_size()
        if size_before - size_after >= qty_to_close * 0.99:
            return order_id
    logger.error("No se confirmó cierre")
    return None

# =================== MEMORIA PERSISTENTE ===================
MEMORY_FILE = "memoria_bot_paper.json" if PAPER_TRADE else "memoria_bot_real.json"

def guardar_memoria():
    data = {}
    if PAPER_TRADE:
        data = {
            "balance": paper_balance,
            "win": paper_win_count,
            "loss": paper_loss_count,
            "total": paper_total_trades,
            "history": paper_trade_history,
            "positions": {tid: {k: v for k, v in t.items() if k != 'analisis'} for tid, t in paper_positions.items()}
        }
    else:
        data = {
            "balance": REAL_BALANCE,
            "win": WIN_COUNT,
            "loss": LOSS_COUNT,
            "total": TOTAL_TRADES,
            "history": TRADE_HISTORY,
            "positions": {tid: {k: v for k, v in t.items() if k != 'analisis'} for tid, t in REAL_ACTIVE_TRADES.items()}
        }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(data, f, indent=4)
        logger.info("💾 Memoria guardada")
    except Exception as e:
        logger.error(f"Error guardando: {e}")

def cargar_memoria():
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    global REAL_BALANCE, WIN_COUNT, LOSS_COUNT, TOTAL_TRADES, TRADE_HISTORY, REAL_ACTIVE_TRADES
    if not os.path.exists(MEMORY_FILE):
        return
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        if PAPER_TRADE:
            paper_balance = data.get("balance", 1000.0)
            paper_win_count = data.get("win", 0)
            paper_loss_count = data.get("loss", 0)
            paper_total_trades = data.get("total", 0)
            paper_trade_history = data.get("history", [])
            paper_positions = {int(k): v for k, v in data.get("positions", {}).items()}
        else:
            REAL_BALANCE = data.get("balance")
            WIN_COUNT = data.get("win", 0)
            LOSS_COUNT = data.get("loss", 0)
            TOTAL_TRADES = data.get("total", 0)
            TRADE_HISTORY = data.get("history", [])
            REAL_ACTIVE_TRADES = {int(k): v for k, v in data.get("positions", {}).items()}
        logger.info("📂 Memoria cargada")
    except Exception as e:
        logger.error(f"Error cargando: {e}")

# =================== TELEGRAM (con emojis) ===================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto[:4000]}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def telegram_enviar_imagen(ruta, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        with open(ruta, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                          data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1000]},
                          files={"photo": foto}, timeout=15)
    except Exception as e:
        logger.error(f"Imagen error: {e}")

def enviar_mensaje_trade(trade_id, decision, entrada, sl, tp1, qty, razon, analisis, modo, balance):
    emoji_decision = "✅ COMPRAR" if decision == "Buy" else "🔴 VENDER"
    emoji_entry = "🚪"
    emoji_sl = "🛑"
    emoji_tp = "🎯"
    emoji_qty = "📊"
    emoji_balance = "💰"
    msg = (
        f"{emoji_decision} *{modo} #{trade_id}*\n"
        f"{emoji_entry} *Entrada:* {entrada:.2f} USDT\n"
        f"{emoji_sl} *Stop Loss:* {sl:.2f} USDT\n"
        f"{emoji_tp} *Take Profit 1:* {tp1:.2f} USDT\n"
        f"{emoji_qty} *Cantidad:* {qty:.3f} BTC\n"
        f"{emoji_balance} *Balance:* {balance:.2f} USDT\n"
        f"📝 *Razón:* {razon[:200]}\n"
        f"🔍 *Análisis:* {analisis[:300]}..."
    )
    telegram_mensaje(msg)

def enviar_mensaje_cierre(trade_id, pnl, balance, motivo):
    emoji = "✅" if pnl > 0 else "❌"
    msg = f"{emoji} *CIERRE #{trade_id}*\n📈 PnL: {pnl:+.2f} USDT\n💰 Balance: {balance:.2f} USDT\n📌 Motivo: {motivo}"
    telegram_mensaje(msg)

# =================== INDICADORES Y ANÁLISIS TÉCNICO ===================
def obtener_velas(interval, limit=150):
    try:
        r = requests.get(f"{BASE_URL}/v5/market/kline", params={"category": "linear", "symbol": SYMBOL, "interval": interval, "limit": limit}, timeout=20)
        data = r.json()
        if data.get("retCode") != 0:
            return pd.DataFrame()
        lista = data.get("result")["list"][::-1]
        df = pd.DataFrame(lista, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error velas: {e}")
        return pd.DataFrame()

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

def detectar_zonas_mercado(df, idx=-2):
    if df.empty or len(df) < 20:
        return 0, 0, 0, 0, "LATERAL", "LATERAL"
    df_eval = df if idx == -1 else df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(20).min().iloc[-1]
    resistencia = df_eval['high'].rolling(20).max().iloc[-1]
    y = df_eval['close'].values[-120:]
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    tend = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    micro = 'SUBIENDO' if micro_slope > 0.2 else 'CAYENDO' if micro_slope < -0.2 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tend, micro

def generar_grafico(df, titulo, soporte=None, resistencia=None, slope=None, intercept=None,
                    entry=None, sl=None, tp1=None, side=None, excluir_actual=False):
    if df.empty:
        return None
    if excluir_actual and len(df) > 1:
        df_plot = df.iloc[:-1].tail(GRAFICO_VELAS_LIMIT).copy()
    else:
        df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    if len(df_plot) < 3:
        return None
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})
    x = np.arange(len(df_plot))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax1.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax1.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
    if soporte:
        ax1.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte')
    if resistencia:
        ax1.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia')
    if 'ema20' in df_plot.columns:
        ax1.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    if 'ema50' in df_plot.columns:
        ax1.plot(x, df_plot['ema50'], 'orange', lw=2, label='EMA50')
    if slope is not None and intercept is not None and slope != 0:
        x_trend = np.array([0, len(df_plot)-1])
        y_trend = intercept + slope * x_trend
        ax1.plot(x_trend, y_trend, color='white', linestyle='-.', lw=2, label='Tendencia', alpha=0.7)
    if entry:
        ax1.axhline(entry, color='orange', ls=':', lw=1.5, label='Entry')
        ax1.scatter(x[-1], entry, color='lime' if side == 'Buy' else 'red', s=100, edgecolors='white', zorder=5)
    if sl:
        ax1.axhline(sl, color='red', ls='--', lw=1.5, label='SL')
    if tp1:
        ax1.axhline(tp1, color='lime', ls='--', lw=1.5, label='TP1')
    ax1.set_title(sanitize_for_matplotlib(titulo), color='white')
    ax1.set_ylabel('Precio (USDT)', color='white')
    ax1.tick_params(colors='white')
    ax1.set_facecolor('#121212')
    # RSI
    ax2.plot(x, df_plot['rsi'], 'cyan', lw=1)
    ax2.axhline(70, color='red', ls='--', alpha=0.5)
    ax2.axhline(30, color='green', ls='--', alpha=0.5)
    ax2.set_ylabel('RSI', color='white')
    ax2.set_ylim(0, 100)
    ax2.set_facecolor('#121212')
    # MACD
    ax3.plot(x, df_plot['macd'], 'blue', lw=1, label='MACD')
    ax3.plot(x, df_plot['macd_signal'], 'red', lw=1, label='Signal')
    ax3.bar(x, df_plot['macd_diff'], color='gray', alpha=0.5, label='Histogram')
    ax3.set_ylabel('MACD', color='white')
    ax3.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.5, facecolor='black', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close()
    return img

def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# =================== PROMPT CORREGIDO (LIBRE Y SIN DOGMAS) ===================
PROMPT_TRADING = """
Eres un trader profesional con décadas de experiencia. Analiza los gráficos de BTC/USDT:
- Gráfico 1: Velas de 10 minutos (ejecución)
- Gráfico 2: Velas de 1 hora (tendencia)

Interpreta libremente todo lo que ves: velas, niveles de soporte/resistencia (estáticos o dinámicos), EMAs 20/50, RSI, MACD, volumen implícito, patrones, estructura de mercado.

Principios generales (son guías, no reglas fijas):
- Un soporte puede actuar como rebote o ser roto. Observa la fuerza de las velas, las mechas, el cierre.
- Una resistencia puede rechazar o ser superada. Lo mismo.
- Las EMAs cambian de rol según el precio: si se rompen pueden pasar de soporte a resistencia o viceversa.
- Los patrones de velas (martillo, engulfing, doji, etc.) ganan importancia en zonas clave.
- Las trampas de liquidez son comunes: rupturas falsas que luego revierten. Confirma con cierres fuera del nivel.
- La tendencia de 1h da sesgo principal, pero el precio puede tener microestructura contraria.
- RSI y MACD son confirmación, no señales primarias. Busca divergencias.
- Coloca el stop loss detrás del último pivote o nivel relevante. El take profit 1 debe ser alcanzable.
- Si no ves una setup clara con buena relación riesgo/recompensa, no operes.

Tu decisión debe ser COMPRAR (Buy), VENDER (Sell) o NO HACER NADA (Hold).
Si decides Buy o Sell, proporciona:
- entry_price: precio exacto o nivel límite
- sl_price: stop loss
- tp1_price: primer objetivo parcial (puede ser un 50% del tamaño)
Además, una breve razón (máx 150 caracteres) y un análisis completo en español.

Eres libre de interpretar cualquier situación. No hay reglas prohibidas. Evalúa cada caso con criterio.

Responde ÚNICAMENTE con un JSON válido:
{"decision": "Buy/Sell/Hold", "entry_price": 0.0, "sl_price": 0.0, "tp1_price": 0.0, "razon": "...", "analisis": "..."}
Si Hold, precios 0.0.
"""

def analizar_con_claude(img_ltf, img_htf, reintentos=2):
    for intento in range(reintentos + 1):
        try:
            img_ltf_b64 = pil_to_base64(img_ltf)
            img_htf_b64 = pil_to_base64(img_htf)
            response = client.chat.completions.create(
                model=MODELO_VISION,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT_TRADING},
                    {"type": "image_url", "image_url": {"url": img_ltf_b64}},
                    {"type": "image_url", "image_url": {"url": img_htf_b64}}
                ]}],
                temperature=0.2,
                timeout=60
            )
            raw = response.choices[0].message.content
            logger.debug(f"Respuesta cruda (intento {intento+1}): {raw[:300]}")
            # Limpiar markdown
            clean = re.sub(r'^```json\s*', '', raw, flags=re.IGNORECASE)
            clean = re.sub(r'\s*```$', '', clean)
            match = re.search(r'\{.*?"decision"\s*:\s*"(?:Buy|Sell|Hold)".*?\}', clean, re.DOTALL)
            if not match:
                match = re.search(r'\{.*\}', clean, re.DOTALL)
            if match:
                json_str = match.group()
                try:
                    data = json.loads(json_str)
                except:
                    data = json_repair.repair_json(json_str, return_objects=True)
                decision = data.get("decision", "Hold")
                razon = data.get("razon", "")[:150]
                analisis = data.get("analisis", "")
                entry = float(data.get("entry_price", 0.0))
                sl = float(data.get("sl_price", 0.0))
                tp1 = float(data.get("tp1_price", 0.0))
                return decision, razon, analisis, entry, sl, tp1
            else:
                raise ValueError("No se encontró JSON")
        except Exception as e:
            logger.error(f"Intento {intento+1} falló: {e}")
            if intento == reintentos:
                return "Hold", f"Error: {str(e)[:50]}", "", 0.0, 0.0, 0.0
            time.sleep(2)
    return "Hold", "Error crítico", "", 0.0, 0.0, 0.0

# =================== APERTURA DE POSICIÓN ===================
def abrir_posicion(decision, precio_actual, razon, analisis, entry_ia, sl_ia, tp1_ia, df_ltf, soporte, resistencia, slope, intercept):
    global paper_balance, paper_positions, paper_trade_counter, REAL_BALANCE, TRADE_COUNTER, REAL_ACTIVE_TRADES

    if decision not in ["Buy", "Sell"]:
        return

    if PAPER_TRADE:
        balance = paper_balance
        positions = paper_positions
        paper = True
    else:
        if REAL_BALANCE is None:
            REAL_BALANCE = get_real_balance()
            if REAL_BALANCE is None:
                logger.error("No se pudo obtener balance real")
                return
        balance = REAL_BALANCE
        positions = REAL_ACTIVE_TRADES
        paper = False

    max_trades = min(MAX_CONCURRENT_TRADES, int(balance // MIN_MARGIN_PER_TRADE) or 1)
    if len(positions) >= max_trades:
        logger.warning(f"Máximo trades alcanzado ({max_trades})")
        return

    free_margin = get_free_margin()
    if free_margin <= 0:
        logger.error("Margen libre insuficiente")
        return

    entrada = entry_ia if entry_ia and entry_ia > 0 else precio_actual
    sl = sl_ia if sl_ia and sl_ia > 0 else (entrada * 0.995 if decision == "Buy" else entrada * 1.005)
    tp1 = tp1_ia if tp1_ia and tp1_ia > 0 else (entrada * 1.005 if decision == "Buy" else entrada * 0.995)

    # Ajustes mínimos de seguridad
    if decision == "Buy":
        if sl >= entrada:
            sl = entrada * 0.995
        if tp1 <= entrada:
            tp1 = entrada * 1.005
    else:
        if sl <= entrada:
            sl = entrada * 1.005
        if tp1 >= entrada:
            tp1 = entrada * 0.995

    riesgo_usdt = min(3.0, free_margin * 0.1)
    distancia_sl = abs(entrada - sl)
    if distancia_sl <= 0:
        distancia_sl = entrada * 0.002
    qty_btc = riesgo_usdt / distancia_sl
    max_qty = (balance * LEVERAGE) / entrada
    qty_btc = min(qty_btc, max_qty)
    if qty_btc < 0.001:
        logger.warning(f"Cantidad muy pequeña: {qty_btc:.4f}")
        return
    qty_btc = round(qty_btc, 3)
    nocional = qty_btc * entrada
    if nocional < 100:
        qty_btc = round(100.0 / entrada, 3)
    margen_necesario = (qty_btc * entrada) / LEVERAGE
    if margen_necesario > free_margin * 0.98:
        logger.error("Margen insuficiente")
        return

    if paper:
        comision = nocional * PAPER_COMMISSION_PCT
        paper_balance -= comision
        paper_trade_counter += 1
        trade_id = paper_trade_counter
        positions[trade_id] = {
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_actual": sl, "tp1": tp1,
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "analisis": analisis,
            "trailing_activado": False, "best_price": entrada, "trailing_stop": None
        }
        modo = "PAPER"
        balance_mostrar = paper_balance
    else:
        order_id = place_market_order(decision, qty_btc)
        if not order_id:
            logger.error("Orden real falló")
            return
        TRADE_COUNTER += 1
        trade_id = TRADE_COUNTER
        positions[trade_id] = {
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_actual": sl, "tp1": tp1,
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "analisis": analisis,
            "trailing_activado": False, "best_price": entrada, "trailing_stop": None
        }
        modo = "REAL"
        balance_mostrar = REAL_BALANCE

    enviar_mensaje_trade(trade_id, decision, entrada, sl, tp1, qty_btc, razon, analisis, modo, balance_mostrar)

    img = generar_grafico(df_ltf, f"{modo} #{trade_id} {decision}", soporte, resistencia, slope, intercept,
                          entry=entrada, sl=sl, tp1=tp1, side=decision)
    if img:
        img.save("/tmp/entrada.png")
        caption = f"📊 {modo} #{trade_id} {decision}\nEntry: {entrada:.2f} | SL: {sl:.2f} | TP1: {tp1:.2f}"
        telegram_enviar_imagen("/tmp/entrada.png", caption)

    guardar_memoria()

# =================== GESTIÓN DE TRADES ACTIVOS ===================
def gestionar_trades_simulado(df):
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    if not paper_positions:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    cerrar = []
    for tid, t in paper_positions.items():
        # TP1
        if not t['tp1_ejecutado'] and t['tp1'] > 0:
            if (t['decision']=="Buy" and h >= t['tp1']) or (t['decision']=="Sell" and l <= t['tp1']):
                qty = round(t['qty_original'] * TP1_PERCENT, 3)
                if qty >= 0.001 and t['qty_restante'] > 0:
                    pnl = (t['tp1'] - t['entrada']) * qty if t['decision']=="Buy" else (t['entrada'] - t['tp1']) * qty
                    pnl -= abs(pnl) * PAPER_COMMISSION_PCT
                    t['pnl_parcial'] += pnl
                    t['qty_restante'] = round(t['qty_original'] - qty, 3)
                    t['tp1_ejecutado'] = True
                    t['trailing_activado'] = True
                    t['best_price'] = t['entrada']
                    t['trailing_stop'] = t['entrada'] * (1 - TRAILING_PERCENT) if t['decision']=="Buy" else t['entrada'] * (1 + TRAILING_PERCENT)
                    logger.info(f"✅ TP1 #{tid} +{pnl:.2f} USDT")
                    if t['qty_restante'] <= 0:
                        cerrar.append(tid)
                else:
                    cerrar.append(tid)
        # Trailing stop
        if t.get('trailing_activado', False) and t['qty_restante'] > 0:
            if t['decision'] == 'Buy':
                if h > t['best_price']:
                    t['best_price'] = h
                    t['trailing_stop'] = t['best_price'] * (1 - TRAILING_PERCENT)
                if l <= t['trailing_stop']:
                    qty = t['qty_restante']
                    pnl = (t['trailing_stop'] - t['entrada']) * qty
                    pnl -= abs(pnl) * PAPER_COMMISSION_PCT
                    total = t['pnl_parcial'] + pnl
                    paper_balance += total
                    paper_total_trades += 1
                    if total > 0:
                        paper_win_count += 1
                    else:
                        paper_loss_count += 1
                    paper_trade_history.append({"pnl": total, "win": total>0, "decision": t['decision'], "razon": t['razon']})
                    cerrar.append(tid)
                    enviar_mensaje_cierre(tid, total, paper_balance, "Trailing Stop")
            else:
                if l < t['best_price']:
                    t['best_price'] = l
                    t['trailing_stop'] = t['best_price'] * (1 + TRAILING_PERCENT)
                if h >= t['trailing_stop']:
                    qty = t['qty_restante']
                    pnl = (t['entrada'] - t['trailing_stop']) * qty
                    pnl -= abs(pnl) * PAPER_COMMISSION_PCT
                    total = t['pnl_parcial'] + pnl
                    paper_balance += total
                    paper_total_trades += 1
                    if total > 0:
                        paper_win_count += 1
                    else:
                        paper_loss_count += 1
                    paper_trade_history.append({"pnl": total, "win": total>0, "decision": t['decision'], "razon": t['razon']})
                    cerrar.append(tid)
                    enviar_mensaje_cierre(tid, total, paper_balance, "Trailing Stop")
        # Stop loss inicial
        if not t.get('tp1_ejecutado', False) and t['qty_restante'] > 0:
            if (t['decision']=="Buy" and l <= t['sl_actual']) or (t['decision']=="Sell" and h >= t['sl_actual']):
                qty = t['qty_restante']
                pnl = (t['sl_actual'] - t['entrada']) * qty if t['decision']=="Buy" else (t['entrada'] - t['sl_actual']) * qty
                pnl -= abs(pnl) * PAPER_COMMISSION_PCT
                total = t['pnl_parcial'] + pnl
                paper_balance += total
                paper_total_trades += 1
                if total > 0:
                    paper_win_count += 1
                else:
                    paper_loss_count += 1
                paper_trade_history.append({"pnl": total, "win": total>0, "decision": t['decision'], "razon": t['razon']})
                cerrar.append(tid)
                enviar_mensaje_cierre(tid, total, paper_balance, "Stop Loss")
    for tid in cerrar:
        del paper_positions[tid]
    guardar_memoria()

def gestionar_trades_real(df):
    # Implementar si se necesita
    pass

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global REAL_BALANCE, paper_balance, ultima_vela
    cargar_memoria()
    set_leverage()
    telegram_mensaje("🤖 Bot de Trading Iniciado - Claude Sonnet 4.5\n📊 Temporalidades: 10m / 1h\n📜 Principios generales, sin reglas fijas")
    logger.info("Bot iniciado con temporalidades 10m/1h y prompt libre")
    if PAPER_TRADE:
        logger.info(f"Saldo paper: {paper_balance:.2f}")
        telegram_mensaje(f"📄 Modo PAPER TRADE - Saldo inicial: {paper_balance:.2f} USDT")
    else:
        REAL_BALANCE = get_real_balance()
        logger.info(f"Saldo real: {REAL_BALANCE:.2f}")
        telegram_mensaje(f"💰 Modo REAL - Saldo: {REAL_BALANCE:.2f} USDT")

    ultima_vela = None
    while True:
        try:
            df10 = obtener_velas(INTERVAL_LTF)
            df60 = obtener_velas(INTERVAL_HTF)
            if df10.empty or df60.empty:
                time.sleep(SLEEP_SECONDS)
                continue
            df10 = calcular_indicadores(df10)
            df60 = calcular_indicadores(df60)
            precio_actual = df10['close'].iloc[-1]
            vela_actual = df10.index[-1]

            if ultima_vela is None or ultima_vela != vela_actual:
                ultima_vela = vela_actual
                sop10, res10, slope10, inter10, _, _ = detectar_zonas_mercado(df10)
                sop60, res60, slope60, inter60, _, _ = detectar_zonas_mercado(df60)

                img10 = generar_grafico(df10, "BTC 10m", soporte=sop10, resistencia=res10, slope=slope10, intercept=inter10, excluir_actual=True)
                img60 = generar_grafico(df60, "BTC 1h", soporte=sop60, resistencia=res60, slope=slope60, intercept=inter60, excluir_actual=True)

                if img10 and img60:
                    dec, razon, analisis, entry, sl, tp1 = analizar_con_claude(img10, img60)
                    logger.info(f"🤖 IA decide: {dec} - {razon[:100]}")
                    if dec in ["Buy", "Sell"]:
                        abrir_posicion(dec, precio_actual, razon, analisis, entry, sl, tp1, df10, sop10, res10, slope10, inter10)
                    else:
                        logger.info(f"Hold. Razón: {razon[:100]}")
                else:
                    logger.error("No se generaron gráficos")

            if PAPER_TRADE and paper_positions:
                gestionar_trades_simulado(df10)

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            logger.error(f"Error grave: {e}", exc_info=True)
            time.sleep(30)

if __name__ == "__main__":
    run_bot()
