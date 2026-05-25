# BOT TRADING CON CLAUDE SONNET 4.5 - SCALPING 3M/30M CON GESTIÓN INTELIGENTE Y CACHÉ
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
from collections import deque
import pickle

# =================== CONFIGURACIÓN DE LOGGING MEJORADO ===================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =================== SANITIZAR EMOJIS PARA MATPLOTLIB ===================
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

# =================== CONFIGURACIÓN DE APIS ===================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Falta OPENROUTER_API_KEY")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": "https://railway.app",
        "X-Title": "Trading Bot",
    }
)
MODELO_VISION = "anthropic/claude-sonnet-4.5"  # Mejor modelo para análisis técnico

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("Faltan BYBIT_API_KEY o BYBIT_API_SECRET")

# =================== MODO PAPER TRADE ===================
PAPER_TRADE = True   # Cambiar a False para real

paper_balance = 1000.0
paper_positions = {}
paper_trade_counter = 0
paper_win_count = 0
paper_loss_count = 0
paper_total_trades = 0
paper_trade_history = []

PAPER_COMMISSION_PCT = 0.001

# =================== CONFIGURACIÓN DEL BOT (SCALPING 3M) ===================
SYMBOL = "BTCUSDT"
INTERVAL_LTF = "3"
INTERVAL_HTF = "30"
RISK_PER_TRADE_MAX = 3.0   # Riesgo máximo por trade en USDT
LEVERAGE = 20               # Reducido por seguridad
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 2   # Máximo de trades simultáneos
MIN_MARGIN_PER_TRADE = 3.0
TP1_PERCENT = 0.5           # 50% del tamaño se cierra en TP1
TRAILING_PERCENT = 0.002    # 0.2% de trailing dinámico

# Parámetros para limitar llamadas en mercado lateral
SIDEWAYS_THRESHOLD = 0.002   # Si el rango de precio en las últimas 20 velas es <0.2%, se considera lateral
SIDEWAYS_SKIP_CYCLES = 3     # Saltar análisis durante 3 ciclos si está lateral

REAL_BALANCE = None
REAL_ACTIVE_TRADES = {}
TRADE_COUNTER = 0
WIN_COUNT = 0
LOSS_COUNT = 0
TOTAL_TRADES = 0
TRADE_HISTORY = []

MAX_DAILY_DRAWDOWN_PCT = 0.20
DAILY_START_BALANCE = None
STOPPED_TODAY = False
CURRENT_DAY = None

ULTIMO_APRENDIZAJE = 0
ULTIMO_PROFIT_FACTOR = 1.0
REGLAS_APRENDIDAS = "Aún no hay lecciones."
TOKENS_ACUMULADOS = 0

# =================== CACHÉ DE DECISIONES ===================
DECISION_CACHE = {}  # clave: hash de características, valor: (decisión, timestamp, expiración)
CACHE_EXPIRY_SECONDS = 1800  # 30 minutos

def get_cache_key(df_ltf, df_htf):
    """Genera clave única basada en los últimos valores de precio, RSI, MACD y volumen."""
    last_ltf = df_ltf.iloc[-1]
    last_htf = df_htf.iloc[-1]
    # Tomar características relevantes
    features = (
        round(last_ltf['close'], 2),
        round(last_ltf.get('rsi', 50), 1),
        round(last_ltf.get('macd_hist', 0), 4),
        round(last_htf['close'], 2),
        round(last_htf.get('rsi', 50), 1),
        round(last_htf.get('macd_hist', 0), 4),
        round(df_ltf['close'].std(), 2),  # volatilidad reciente
    )
    key_str = str(features)
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cached_decision(cache_key):
    if cache_key in DECISION_CACHE:
        decision, timestamp, expiry = DECISION_CACHE[cache_key]
        if time.time() - timestamp < expiry:
            logger.info(f"✅ Usando decisión en caché (válida hasta {expiry/60:.0f} min)")
            return decision
        else:
            del DECISION_CACHE[cache_key]
    return None

def store_cached_decision(cache_key, decision):
    DECISION_CACHE[cache_key] = (decision, time.time(), CACHE_EXPIRY_SECONDS)

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
        logger.info("📊 Paper trade: apalancamiento simulado 20x")
        return
    try:
        body = {"category": "linear", "symbol": "BTCUSDT", "buyLeverage": str(LEVERAGE), "sellLeverage": str(LEVERAGE)}
        result = bybit_request("/v5/position/set-leverage", method="POST", body=body)
        ret_code = result.get('retCode')
        if ret_code == 0 or ret_code == 110043:
            logger.info(f"🔧 Apalancamiento {LEVERAGE}x configurado")
        else:
            logger.warning(f"Error configurando apalancamiento: {result}")
    except Exception as e:
        logger.error(f"Excepción configurando apalancamiento: {e}")

def get_real_balance():
    if PAPER_TRADE:
        return paper_balance
    try:
        params = {"accountType": "UNIFIED", "coin": "USDT"}
        result = bybit_request("/v5/account/wallet-balance", method="GET", params=params)
        return float(result['result']['list'][0]['coin'][0]['walletBalance'])
    except Exception as e:
        logger.error(f"Error obteniendo saldo: {e}")
        return None

def get_free_margin():
    if PAPER_TRADE:
        margin_used = 0.0
        for t in paper_positions.values():
            margin_used += (t['qty_original'] * t['entrada']) / LEVERAGE
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
        logger.error(f"Error obteniendo margen libre: {e}")
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
        logger.error(f"Error get_real_position_size: {e}")
        return 0.0

def place_market_order(side, qty):
    if PAPER_TRADE:
        logger.info(f"📄 PAPER: Orden {side} {qty} BTC simulada")
        return f"paper_order_{int(time.time())}"
    try:
        body = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "side": side.capitalize(),
            "orderType": "Market",
            "qty": str(qty),
            "timeInForce": "GTC"
        }
        result = bybit_request("/v5/order/create", method="POST", body=body)
        if result.get('retCode') == 0:
            return result['result']['orderId']
        else:
            logger.error(f"Error orden market: {result}")
            return None
    except Exception as e:
        logger.error(f"Excepción place_market_order: {e}")
        return None

def close_position_qty(qty, side_to_close):
    if PAPER_TRADE:
        logger.info(f"📄 PAPER: Cierre simulado de {qty} BTC lado {side_to_close}")
        return f"paper_close_{int(time.time())}"
    try:
        real_size = get_real_position_size()
        if real_size <= 0.0:
            return "already_closed"
        qty_to_close = min(qty, real_size)
        if qty_to_close <= 0.0 or qty_to_close < 0.001:
            return "already_closed"
        close_side = "Sell" if side_to_close == "Buy" else "Buy"
        body = {
            "category": "linear",
            "symbol": "BTCUSDT",
            "side": close_side,
            "orderType": "Market",
            "qty": str(round(qty_to_close, 3)),
            "timeInForce": "GTC",
            "reduceOnly": True
        }
        result = bybit_request("/v5/order/create", method="POST", body=body)
        if result.get('retCode') == 0:
            logger.info(f"Orden de cierre enviada: {qty_to_close} BTC")
            return result['result']['orderId']
        else:
            logger.error(f"Error close_position_qty: {result}")
            return None
    except Exception as e:
        logger.error(f"Excepción close_position_qty: {e}")
        return None

def close_position_qty_confirm(qty, side_to_close, max_wait=5):
    if PAPER_TRADE:
        return f"paper_confirm_{int(time.time())}"
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
            logger.info(f"Confirmada reducción: {size_before:.4f} -> {size_after:.4f}")
            return order_id
    logger.error(f"No se confirmó reducción tras {max_wait}s")
    return None

# =================== MEMORIA PERSISTENTE (optimizada) ===================
MEMORY_FILE = "memoria_bot_paper.json" if PAPER_TRADE else "memoria_bot_real.json"

def convertir_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convertir_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convertir_serializable(item) for item in obj]
    return obj

def guardar_memoria():
    global ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS
    if PAPER_TRADE:
        active_trades_meta = {}
        for tid, t in paper_positions.items():
            active_trades_meta[tid] = {
                "id": t["id"], "decision": t["decision"], "entrada": t["entrada"],
                "razon": t.get("razon", ""), "tp1_ejecutado": t["tp1_ejecutado"],
                "sl_actual": t.get("sl_actual"), "sl_inicial": t.get("sl_inicial"),
                "tp1": t.get("tp1"),
                "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
                "breakeven_activado": t.get("breakeven_activado", False),
                "trailing_activado": t.get("trailing_activado", False),
                "best_price": t.get("best_price"),
                "trailing_stop": t.get("trailing_stop"),
                "pnl_parcial": t.get("pnl_parcial", 0.0)
            }
        data = {
            "TRADE_HISTORY": paper_trade_history,
            "REGLAS_APRENDIDAS": REGLAS_APRENDIDAS,
            "REAL_BALANCE": paper_balance,
            "WIN_COUNT": paper_win_count,
            "LOSS_COUNT": paper_loss_count,
            "TOTAL_TRADES": paper_total_trades,
            "ULTIMO_APRENDIZAJE": ULTIMO_APRENDIZAJE,
            "TOKENS_ACUMULADOS": TOKENS_ACUMULADOS,
            "ACTIVE_TRADES_META": active_trades_meta,
            "ULTIMO_PROFIT_FACTOR": ULTIMO_PROFIT_FACTOR
        }
    else:
        active_trades_meta = {}
        for tid, t in REAL_ACTIVE_TRADES.items():
            active_trades_meta[tid] = {
                "id": t["id"], "decision": t["decision"], "entrada": t["entrada"],
                "razon": t.get("razon", ""), "tp1_ejecutado": t["tp1_ejecutado"],
                "sl_actual": t.get("sl_actual"), "sl_inicial": t.get("sl_inicial"),
                "tp1": t.get("tp1"),
                "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
                "breakeven_activado": t.get("breakeven_activado", False),
                "trailing_activado": t.get("trailing_activado", False),
                "best_price": t.get("best_price"),
                "trailing_stop": t.get("trailing_stop"),
                "pnl_parcial": t.get("pnl_parcial", 0.0)
            }
        data = {
            "TRADE_HISTORY": TRADE_HISTORY,
            "REGLAS_APRENDIDAS": REGLAS_APRENDIDAS,
            "REAL_BALANCE": REAL_BALANCE,
            "WIN_COUNT": WIN_COUNT,
            "LOSS_COUNT": LOSS_COUNT,
            "TOTAL_TRADES": TOTAL_TRADES,
            "ULTIMO_APRENDIZAJE": ULTIMO_APRENDIZAJE,
            "TOKENS_ACUMULADOS": TOKENS_ACUMULADOS,
            "ACTIVE_TRADES_META": active_trades_meta,
            "ULTIMO_PROFIT_FACTOR": ULTIMO_PROFIT_FACTOR
        }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(convertir_serializable(data), f, indent=4)
        logger.info("💾 Memoria guardada")
    except Exception as e:
        logger.error(f"Error guardando memoria: {e}")

def cargar_memoria():
    global TRADE_HISTORY, REGLAS_APRENDIDAS, REAL_BALANCE, WIN_COUNT, LOSS_COUNT
    global TOTAL_TRADES, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, REAL_ACTIVE_TRADES
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    if not os.path.exists(MEMORY_FILE):
        return
    try:
        with open(MEMORY_FILE, "r") as f:
            data = json.load(f)
        if PAPER_TRADE:
            paper_trade_history = data.get("TRADE_HISTORY", [])
            paper_balance = data.get("REAL_BALANCE", 1000.0)
            paper_win_count = data.get("WIN_COUNT", 0)
            paper_loss_count = data.get("LOSS_COUNT", 0)
            paper_total_trades = data.get("TOTAL_TRADES", 0)
            active_meta = data.get("ACTIVE_TRADES_META", {})
            for tid, meta in active_meta.items():
                paper_positions[int(tid)] = meta
        else:
            TRADE_HISTORY = data.get("TRADE_HISTORY", [])
            REAL_BALANCE = data.get("REAL_BALANCE", None)
            WIN_COUNT = data.get("WIN_COUNT", 0)
            LOSS_COUNT = data.get("LOSS_COUNT", 0)
            TOTAL_TRADES = data.get("TOTAL_TRADES", 0)
            active_meta = data.get("ACTIVE_TRADES_META", {})
            for tid, meta in active_meta.items():
                REAL_ACTIVE_TRADES[int(tid)] = meta
        REGLAS_APRENDIDAS = data.get("REGLAS_APRENDIDAS", REGLAS_APRENDIDAS)
        ULTIMO_APRENDIZAJE = data.get("ULTIMO_APRENDIZAJE", 0)
        TOKENS_ACUMULADOS = data.get("TOKENS_ACUMULADOS", 0)
        ULTIMO_PROFIT_FACTOR = data.get("ULTIMO_PROFIT_FACTOR", 1.0)
        logger.info(f"📂 Memoria cargada. Trades: {paper_total_trades if PAPER_TRADE else TOTAL_TRADES}")
    except Exception as e:
        logger.error(f"Error cargando memoria: {e}")

def parse_json_seguro(raw):
    if not raw or raw.strip() == "":
        return None
    try:
        repaired = json_repair.repair_json(raw)
        return json.loads(repaired)
    except:
        return None

# =================== TELEGRAM ===================
def telegram_mensaje_largo(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram no configurado")
        return
    MAX_LEN = 4000
    if len(texto) <= MAX_LEN:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
        except Exception as e:
            logger.error(f"Excepción telegram: {e}")
        return
    for i in range(0, len(texto), MAX_LEN):
        parte = texto[i:i+MAX_LEN]
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": parte}, timeout=10)
        except Exception as e:
            logger.error(f"Excepción telegram: {e}")

def telegram_mensaje(texto):
    telegram_mensaje_largo(texto)

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram no configurado para imagen")
        return
    try:
        if not os.path.exists(ruta_imagen):
            logger.warning(f"Imagen no encontrada: {ruta_imagen}")
            return
        if len(caption) > 1000:
            caption = caption[:997] + "..."
        with open(ruta_imagen, 'rb') as foto:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
        if resp.status_code != 200:
            logger.error(f"Error imagen Telegram: {resp.status_code} - {resp.text[:200]}")
        else:
            logger.info("🖼️ Imagen enviada a Telegram")
    except Exception as e:
        logger.error(f"Excepción en telegram_enviar_imagen: {e}")

def reporte_estado():
    if PAPER_TRADE:
        balance = paper_balance
        win_count = paper_win_count
        loss_count = paper_loss_count
        total_trades = paper_total_trades
        active_trades = len(paper_positions)
    else:
        if REAL_BALANCE is None:
            return
        balance = REAL_BALANCE
        win_count = WIN_COUNT
        loss_count = LOSS_COUNT
        total_trades = TOTAL_TRADES
        active_trades = len(REAL_ACTIVE_TRADES)
    pnl_global = balance - (DAILY_START_BALANCE or balance)
    winrate = (win_count / total_trades * 100) if total_trades > 0 else 0
    max_din = get_dynamic_max_trades()
    modo = "📄 PAPER" if PAPER_TRADE else "💰 REAL"
    mensaje = (
        f"{modo} ESTADO BTC\n"
        f"💰 Balance: {balance:.2f} USDT\n"
        f"📈 PnL día: {pnl_global:+.2f} USDT\n"
        f"🏆 Winrate: {winrate:.1f}%\n"
        f"🎯 Activos: {active_trades}/{max_din}\n"
        f"⚖️ PF (10t): {ULTIMO_PROFIT_FACTOR:.2f}"
    )
    telegram_mensaje(mensaje)

# =================== INDICADORES Y GRÁFICOS (con RSI, MACD) ===================
def obtener_velas(interval="3", limit=150):
    try:
        r = requests.get(f"{BASE_URL}/v5/market/kline",
                         params={"category": "linear", "symbol": SYMBOL, "interval": interval, "limit": limit},
                         timeout=20)
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
        logger.error(f"Error obteniendo velas {interval}: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    """Agrega EMA, RSI, MACD al DataFrame."""
    if df.empty:
        return df
    df = df.copy()
    # EMAs
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # MACD (12,26,9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    # ATR ya no se calcula (lo dejamos fuera)
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2):
    if df.empty or len(df) < 20:
        return 0,0,0,0,"LATERAL","LATERAL"
    df_eval = df if idx == -1 else df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(20).min().iloc[-1]
    resistencia = df_eval['high'].rolling(20).max().iloc[-1]
    y = df_eval['close'].values[-120:]
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    tend = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    micro = 'SUBIENDO' if micro_slope > 0.2 else 'CAYENDO' if micro_slope < -0.2 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tend, micro

def is_sideways_market(df_ltf, umbral=SIDEWAYS_THRESHOLD):
    """Detecta si el mercado está lateral mirando el rango de las últimas 20 velas."""
    if len(df_ltf) < 20:
        return False
    last_20 = df_ltf.iloc[-20:]
    price_range = (last_20['high'].max() - last_20['low'].min()) / last_20['close'].iloc[-1]
    return price_range < umbral

def generar_grafico_para_vision(df, titulo, soporte=None, resistencia=None, slope=None, intercept=None,
                                entry_price=None, sl_price=None, tp1_price=None, side=None, excluir_actual=False):
    """Genera gráfico con EMAs, RSI, MACD y niveles. Optimizado para memoria."""
    if df.empty:
        return None
    if excluir_actual and len(df) > 1:
        df_plot = df.iloc[:-1].tail(GRAFICO_VELAS_LIMIT).copy()
    else:
        df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    if len(df_plot) < 3:
        return None
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(df_plot))
    # Velas
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
    # EMAs
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    if 'ema50' in df_plot.columns:
        ax.plot(x, df_plot['ema50'], 'orange', lw=2, label='EMA50')
    # Soporte / Resistencia
    if soporte:
        ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte')
    if resistencia:
        ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia')
    # Tendencia
    if slope is not None and intercept is not None and slope != 0:
        x_trend = np.array([0, len(df_plot)-1])
        y_trend = intercept + slope * x_trend
        ax.plot(x_trend, y_trend, color='white', linestyle='-.', lw=2, label='Tendencia', alpha=0.7)
    # Entrada/SL/TP
    if entry_price is not None:
        ax.axhline(entry_price, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Entry')
        circle_color = 'lime' if side == 'Buy' else 'red'
        ax.scatter(x[-1], entry_price, color=circle_color, s=100, edgecolors='white', zorder=5)
        ax.annotate(f'Entry {entry_price:.0f}', xy=(x[-1], entry_price), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, color='white',
                    bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
    if sl_price is not None:
        ax.axhline(sl_price, color='red', linestyle='--', linewidth=2, label=f'SL {sl_price:.0f}')
    if tp1_price is not None:
        ax.axhline(tp1_price, color='lime', linestyle='--', linewidth=2, label=f'TP1 {tp1_price:.0f}')
    # Indicadores RSI y MACD en subplots
    if 'rsi' in df_plot.columns and 'macd' in df_plot.columns:
        ax2 = ax.twinx()
        ax2.plot(x, df_plot['rsi'], color='purple', alpha=0.5, label='RSI')
        ax2.axhline(70, color='red', linestyle=':', alpha=0.5)
        ax2.axhline(30, color='green', linestyle=':', alpha=0.5)
        ax2.set_ylabel('RSI', color='purple')
        # Para no saturar, no mostramos MACD aquí, pero la IA lo verá en los datos
    titulo_limpio = sanitize_for_matplotlib(titulo)
    ax.set_title(titulo_limpio, color='white', fontsize=14)
    ax.set_xlabel('Tiempo (velas)', color='white')
    ax.set_ylabel('Precio (USDT)', color='white')
    ax.tick_params(colors='white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white')
    ax.spines['left'].set_color('white')
    ax.spines['right'].set_color('white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.5, facecolor='black', edgecolor='white', labelcolor='white')
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

# =================== PROMPT PARA CLAUDE SONNET 4.5 (con libertad total) ===================
def analizar_con_claude(img_ltf, img_htf, df_ltf, df_htf):
    global TOKENS_ACUMULADOS, REGLAS_APRENDIDAS
    try:
        img_ltf_b64 = pil_to_base64(img_ltf)
        img_htf_b64 = pil_to_base64(img_htf)
        
        # Extraer datos numéricos recientes para dar contexto adicional
        ultima_vela_ltf = df_ltf.iloc[-1]
        ultima_vela_htf = df_htf.iloc[-1]
        rsi_ltf = ultima_vela_ltf.get('rsi', 50)
        rsi_htf = ultima_vela_htf.get('rsi', 50)
        macd_ltf = ultima_vela_ltf.get('macd_hist', 0)
        macd_htf = ultima_vela_htf.get('macd_hist', 0)
        close_ltf = ultima_vela_ltf['close']
        
        lecciones = REGLAS_APRENDIDAS if REGLAS_APRENDIDAS != "Aún no hay lecciones." else "No hay lecciones previas. Opera solo cuando haya claras señales de soporte/resistencia."
        
        prompt = f"""
Eres un trader profesional de crypto con años de experiencia en scalping, especializado en análisis técnico profundo.
Tienes TOTAL LIBERTAD para analizar los gráficos que se te muestran y decidir si COMPRAR, VENDER o MANTENER.

INSTRUCCIONES:
- Analiza AMBOS gráficos: el de 3 minutos (LTF) y el de 30 minutos (HTF).
- Busca patrones de velas (martillo, estrella fugaz, engulfing, etc.), niveles de soporte/resistencia, divergencias en RSI/MACD, cruces de EMAs, estructura de mercado.
- Puedes usar también los datos numéricos que te proporciono: RSI actual (LTF: {rsi_ltf:.1f}, HTF: {rsi_htf:.1f}), MACD histograma (LTF: {macd_ltf:.4f}, HTF: {macd_htf:.4f}), precio actual {close_ltf:.2f}.
- Decide si es un buen momento para entrar (Buy o Sell) o si es mejor Hold.
- Si decides entrar, debes proporcionar:
   * entry_price: el precio exacto al que quieres entrar (puede ser el actual o uno ligeramente diferente si esperas un pullback).
   * sl_price: el stop loss (debe ser razonable, entre 0.1% y 0.6% de distancia).
   * tp1_price: primer take profit (entre 0.3% y 0.8% de distancia). Luego el bot cerrará el 50% en TP1 y dejará el resto con trailing.
- También debes dar un nivel de confianza (0-100).
- Explica brevemente tu razonamiento (máx 200 caracteres en "razon", y una explicación más detallada en "explicacion").

Lecciones aprendidas anteriormente (puedes usarlas como guía):
{lecciones}

Responde ÚNICAMENTE con un JSON en una sola línea con esta estructura exacta:
{{"decision": "Buy/Sell/Hold", "razon": "texto corto", "explicacion": "análisis detallado", "entry_price": 0.0, "sl_price": 0.0, "tp1_price": 0.0, "confianza": 0}}
Si la decisión es Hold, todos los precios deben ser 0.0.
No añadas texto adicional fuera del JSON.
"""
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_ltf_b64}},
                        {"type": "image_url", "image_url": {"url": img_htf_b64}}
                    ]
                }
            ],
            temperature=0.3,
            timeout=60
        )
        TOKENS_ACUMULADOS += response.usage.total_tokens if response.usage else 0
        contenido = response.choices[0].message.content
        json_match = re.search(r'\{.*\}(?=\s*$)', contenido, re.DOTALL)
        if json_match:
            datos = parse_json_seguro(json_match.group())
        else:
            datos = parse_json_seguro(contenido)
        if not datos:
            return "Hold", "Error parsing", "", None, None, None, 0

        decision = datos.get("decision", "Hold")
        razon = datos.get("razon", "")
        explicacion = datos.get("explicacion", "")
        entry_price = datos.get("entry_price")
        sl_price = datos.get("sl_price")
        tp1_price = datos.get("tp1_price")
        confianza = datos.get("confianza", 0)
        
        # Validar que los precios sean razonables (no negativos, etc.)
        if decision != "Hold":
            if entry_price is None or entry_price <= 0:
                decision = "Hold"
            if sl_price is None or sl_price <= 0:
                sl_price = entry_price * 0.995 if decision == "Buy" else entry_price * 1.005
            if tp1_price is None or tp1_price <= 0:
                tp1_price = entry_price * 1.005 if decision == "Buy" else entry_price * 0.995
        
        return decision, razon, explicacion, entry_price, sl_price, tp1_price, confianza
    except Exception as e:
        logger.error(f"Error en IA: {e}")
        return "Hold", f"Error: {str(e)[:50]}", "", None, None, None, 0

# =================== GESTIÓN DE RIESGO Y APERTURA (con SL/TP dados por IA) ===================
def calcular_riesgo_dinamico(free_margin):
    if free_margin >= 20:
        return RISK_PER_TRADE_MAX
    elif free_margin >= 10:
        return 1.5
    else:
        return 1.0

def abrir_posicion_con_ia(decision, precio_actual, razon, contexto, sl_ia, tp1_ia, df_ltf, sop, res, slope, inter):
    global paper_balance, paper_positions, paper_trade_counter, REAL_BALANCE, TRADE_COUNTER, REAL_ACTIVE_TRADES

    if decision not in ["Buy", "Sell"]:
        logger.info(f"Decisión {decision} no es operativa. Ignorando.")
        return

    if PAPER_TRADE:
        balance = paper_balance
        positions = paper_positions
        is_paper = True
    else:
        if REAL_BALANCE is None:
            REAL_BALANCE = get_real_balance()
            if REAL_BALANCE is None:
                logger.error("No se pudo obtener balance real")
                return
        balance = REAL_BALANCE
        positions = REAL_ACTIVE_TRADES
        is_paper = False

    max_trades = get_dynamic_max_trades()
    if len(positions) >= max_trades:
        logger.warning(f"Máximo dinámico de trades ({max_trades}) alcanzado.")
        return

    free_margin = get_free_margin()
    if free_margin <= 0:
        logger.error("Margen libre insuficiente.")
        return

    risk_per_trade = calcular_riesgo_dinamico(free_margin)
    logger.info(f"Riesgo fijado en {risk_per_trade} USDT (margen libre: {free_margin:.2f})")

    entrada = sl_ia if sl_ia else precio_actual
    if decision == "Buy":
        # Si la IA no dio SL, usar 0.3% debajo de entrada
        sl = sl_ia if sl_ia else entrada * 0.997
        tp1 = tp1_ia if tp1_ia else entrada * 1.004
    else:
        sl = sl_ia if sl_ia else entrada * 1.003
        tp1 = tp1_ia if tp1_ia else entrada * 0.996

    # Asegurar que SL no esté demasiado lejos o cerca
    if decision == "Buy":
        distancia_sl = entrada - sl
        if distancia_sl <= 0 or distancia_sl > entrada * 0.01:
            sl = entrada * 0.997
            distancia_sl = entrada - sl
    else:
        distancia_sl = sl - entrada
        if distancia_sl <= 0 or distancia_sl > entrada * 0.01:
            sl = entrada * 1.003
            distancia_sl = sl - entrada

    qty_btc = risk_per_trade / distancia_sl
    max_qty = (balance * LEVERAGE) / entrada
    qty_btc = min(qty_btc, max_qty)
    if qty_btc < 0.001:
        logger.warning(f"Cantidad muy pequeña ({qty_btc:.4f} BTC). No se abre.")
        return

    margen_necesario = (qty_btc * entrada) / LEVERAGE
    if margen_necesario > free_margin * 0.98:
        logger.error(f"Margen insuficiente: necesario {margen_necesario:.2f} > libre {free_margin:.2f}")
        return

    qty_btc = round(qty_btc, 3)
    nocional = qty_btc * entrada
    if nocional < 100.0:
        qty_btc = round(100.0 / entrada, 3)
        logger.warning(f"Ajustado a nocional mínimo: {qty_btc} BTC")
        margen_necesario = (qty_btc * entrada) / LEVERAGE
        if margen_necesario > free_margin:
            logger.error("Tras ajuste, margen excedido.")
            return

    if is_paper:
        order_id = f"paper_{paper_trade_counter+1}"
        logger.info(f"📄 PAPER: Orden MARKET {decision} {qty_btc} BTC a {entrada:.2f}")
        comision = nocional * PAPER_COMMISSION_PCT
        paper_balance -= comision
        paper_trade_counter += 1
        trade_id = paper_trade_counter
        positions[trade_id] = {
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_inicial": sl, "sl_actual": sl,
            "tp1": tp1,
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto,
            "trailing_activado": False,
            "best_price": entrada,
            "trailing_stop": entrada * (1 - TRAILING_PERCENT) if decision == "Buy" else entrada * (1 + TRAILING_PERCENT)
        }
        modo = "📄 PAPER"
    else:
        order_id = place_market_order(decision, qty_btc)
        if not order_id:
            logger.error("No se pudo abrir orden real.")
            return
        TRADE_COUNTER += 1
        trade_id = TRADE_COUNTER
        positions[trade_id] = {
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_inicial": sl, "sl_actual": sl,
            "tp1": tp1,
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto,
            "trailing_activado": False,
            "best_price": entrada,
            "trailing_stop": entrada * (1 - TRAILING_PERCENT) if decision == "Buy" else entrada * (1 + TRAILING_PERCENT)
        }
        modo = "💰 REAL"

    msg = (f"{modo} [#{trade_id}] {decision} MARKET en {entrada:.2f} | Qty {qty_btc} BTC (riesgo {risk_per_trade} USDT)\n"
           f"🛑 SL: {sl:.2f} (dist {distancia_sl:.1f} USD)\n"
           f"🎯 TP1: {tp1:.2f} | Trailing tras TP1: {TRAILING_PERCENT*100:.2f}%\n"
           f"📝 Razón: {razon}\n"
           f"💰 Margen requerido: {margen_necesario:.2f} USDT | Libre disponible: {free_margin:.2f} USDT")
    logger.info(msg)
    telegram_mensaje(msg)

    titulo_grafico = f"Entrada - {modo} #{trade_id}"
    img_completa = generar_grafico_para_vision(df_ltf, titulo_grafico, sop, res, slope, inter,
                                               entry_price=entrada, sl_price=sl,
                                               tp1_price=tp1, side=decision, excluir_actual=False)
    if img_completa:
        img_completa.save("/tmp/in_completo.png")
        caption = (f"{modo} #{trade_id} {decision}\n"
                   f"Entry: {entrada:.2f} | SL: {sl:.2f} | TP1: {tp1:.2f}")
        telegram_enviar_imagen("/tmp/in_completo.png", caption)

    guardar_memoria()

def get_dynamic_max_trades():
    if PAPER_TRADE:
        balance = paper_balance
    else:
        if REAL_BALANCE is None:
            return 1
        balance = REAL_BALANCE
    max_by_balance = int(balance // MIN_MARGIN_PER_TRADE)
    if max_by_balance < 1:
        max_by_balance = 1
    return min(MAX_CONCURRENT_TRADES, max_by_balance)

def sync_active_trades_with_bybit():
    if PAPER_TRADE:
        return
    global REAL_ACTIVE_TRADES
    real_size = get_real_position_size()
    if real_size == 0.0 and REAL_ACTIVE_TRADES:
        logger.info("🔄 Sincronización: No hay posición real. Limpiando trades fantasmas.")
        REAL_ACTIVE_TRADES.clear()
        guardar_memoria()
    elif real_size > 0.0 and not REAL_ACTIVE_TRADES:
        logger.warning("⚠️ Hay posición real pero el bot no la registra.")
    else:
        mem_size = sum(t['qty_restante'] for t in REAL_ACTIVE_TRADES.values())
        if abs(mem_size - real_size) > 0.002:
            logger.warning(f"⚠️ Discrepancia de tamaño: memoria {mem_size:.3f} BTC, real {real_size:.3f} BTC.")
            if REAL_ACTIVE_TRADES:
                tid = list(REAL_ACTIVE_TRADES.keys())[0]
                REAL_ACTIVE_TRADES[tid]['qty_restante'] = real_size
                for other in list(REAL_ACTIVE_TRADES.keys())[1:]:
                    del REAL_ACTIVE_TRADES[other]
                guardar_memoria()

# =================== GESTIÓN DE TRADES ACTIVOS (con trailing automático mejorado) ===================
def revisar_sl_tp_simulado(df, precio_actual):
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions, ULTIMO_APRENDIZAJE
    if not paper_positions:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    cerrar_ids = []
    for tid, t in list(paper_positions.items()):
        # TP1: cerrar 50% si no se ha ejecutado y se alcanza el precio
        if not t['tp1_ejecutado'] and t['tp1'] > 0:
            if (t['decision']=="Buy" and h >= t['tp1']) or (t['decision']=="Sell" and l <= t['tp1']):
                qty_tp1 = round(t['qty_original'] * TP1_PERCENT, 3)
                if qty_tp1 >= 0.001 and t['qty_restante'] > 0:
                    pnl_parcial = (t['tp1'] - t['entrada']) * qty_tp1 if t['decision']=="Buy" else (t['entrada'] - t['tp1']) * qty_tp1
                    comision = abs(pnl_parcial) * PAPER_COMMISSION_PCT
                    pnl_parcial -= comision
                    t['pnl_parcial'] += pnl_parcial
                    t['qty_restante'] = round(t['qty_original'] - qty_tp1, 3)
                    t['tp1_ejecutado'] = True
                    t['trailing_activado'] = True
                    t['best_price'] = t['entrada']
                    t['trailing_stop'] = t['entrada'] * (1 - TRAILING_PERCENT) if t['decision']=="Buy" else t['entrada'] * (1 + TRAILING_PERCENT)
                    logger.info(f"✅ PAPER: TP1 #{tid} +{pnl_parcial:.2f} USDT, trailing activado")
                    telegram_mensaje(f"✅ PAPER TP1 #{tid}: +{pnl_parcial:.2f} USDT. Trailing stop activado.")
                    if t['qty_restante'] <= 0.0001:
                        cerrar_ids.append(tid)
                else:
                    cerrar_ids.append(tid)

        # Trailing stop dinámico (solo si activado después de TP1)
        if t.get('trailing_activado', False) and t['qty_restante'] > 0:
            if t['decision'] == 'Buy':
                # Actualizar mejor precio
                if h > t['best_price']:
                    t['best_price'] = h
                    t['trailing_stop'] = t['best_price'] * (1 - TRAILING_PERCENT)
                if l <= t['trailing_stop']:
                    qty_restante = t['qty_restante']
                    pnl_resto = (t['trailing_stop'] - t['entrada']) * qty_restante
                    comision = abs(pnl_resto) * PAPER_COMMISSION_PCT
                    pnl_resto -= comision
                    pnl_total = t['pnl_parcial'] + pnl_resto
                    paper_balance += pnl_total
                    paper_total_trades += 1
                    if pnl_total > 0:
                        paper_win_count += 1
                    else:
                        paper_loss_count += 1
                    paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                    cerrar_ids.append(tid)
                    msg = f"📉 PAPER CIERRE #{tid} por Trailing Stop - PnL: {pnl_total:+.2f} USDT"
                    logger.info(msg)
                    telegram_mensaje(msg)
                    reporte_estado()
            else:  # Sell
                if l < t['best_price']:
                    t['best_price'] = l
                    t['trailing_stop'] = t['best_price'] * (1 + TRAILING_PERCENT)
                if h >= t['trailing_stop']:
                    qty_restante = t['qty_restante']
                    pnl_resto = (t['entrada'] - t['trailing_stop']) * qty_restante
                    comision = abs(pnl_resto) * PAPER_COMMISSION_PCT
                    pnl_resto -= comision
                    pnl_total = t['pnl_parcial'] + pnl_resto
                    paper_balance += pnl_total
                    paper_total_trades += 1
                    if pnl_total > 0:
                        paper_win_count += 1
                    else:
                        paper_loss_count += 1
                    paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                    cerrar_ids.append(tid)
                    msg = f"📉 PAPER CIERRE #{tid} por Trailing Stop - PnL: {pnl_total:+.2f} USDT"
                    logger.info(msg)
                    telegram_mensaje(msg)
                    reporte_estado()

        # Stop loss inicial (solo si no se ha alcanzado TP1)
        if not t['tp1_ejecutado'] and t['qty_restante'] > 0:
            if (t['decision']=="Buy" and l <= t['sl_actual']) or (t['decision']=="Sell" and h >= t['sl_actual']):
                qty_restante = t['qty_restante']
                pnl_resto = (t['sl_actual'] - t['entrada']) * qty_restante if t['decision']=="Buy" else (t['entrada'] - t['sl_actual']) * qty_restante
                comision = abs(pnl_resto) * PAPER_COMMISSION_PCT
                pnl_resto -= comision
                pnl_total = t['pnl_parcial'] + pnl_resto
                paper_balance += pnl_total
                paper_total_trades += 1
                if pnl_total>0:
                    paper_win_count+=1
                else:
                    paper_loss_count+=1
                paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                cerrar_ids.append(tid)
                motivo = "Stop Loss inicial"
                msg = f"🛑 PAPER CIERRE #{tid} por {motivo} - PnL: {pnl_total:+.2f} USDT"
                logger.info(msg)
                telegram_mensaje(msg)
                reporte_estado()

    for tid in cerrar_ids:
        del paper_positions[tid]

    if paper_total_trades - ULTIMO_APRENDIZAJE >= 10:
        aprender_de_trades()

def real_revisar_sl_tp(df, precio_actual):
    global REAL_BALANCE, WIN_COUNT, LOSS_COUNT, TOTAL_TRADES, TRADE_HISTORY, REAL_ACTIVE_TRADES, ULTIMO_APRENDIZAJE
    if not REAL_ACTIVE_TRADES:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    cerrar_ids = []
    for tid, t in list(REAL_ACTIVE_TRADES.items()):
        # TP1
        if not t['tp1_ejecutado'] and t['tp1']>0:
            if (t['decision']=="Buy" and h>=t['tp1']) or (t['decision']=="Sell" and l<=t['tp1']):
                qty_tp1 = round(t['qty_original'] * TP1_PERCENT, 3)
                if qty_tp1>=0.001 and t['qty_restante']>0:
                    result = close_position_qty_confirm(qty_tp1, t['decision'])
                    if result and result!="already_closed":
                        pnl_parcial = (t['tp1']-t['entrada'])*qty_tp1 if t['decision']=="Buy" else (t['entrada']-t['tp1'])*qty_tp1
                        t['pnl_parcial']+=pnl_parcial
                        t['qty_restante']=round(t['qty_original']-qty_tp1,3)
                        t['tp1_ejecutado']=True
                        t['trailing_activado'] = True
                        t['best_price'] = t['entrada']
                        t['trailing_stop'] = t['entrada'] * (1 - TRAILING_PERCENT) if t['decision']=="Buy" else t['entrada'] * (1 + TRAILING_PERCENT)
                        logger.info(f"✅ TP1 #{tid} +{pnl_parcial:.2f} USDT, trailing activado")
                        telegram_mensaje(f"✅ TP1 #{tid}: +{pnl_parcial:.2f} USDT. Trailing activado.")
                        if t['qty_restante']<=0.0001:
                            cerrar_ids.append(tid)
                    else:
                        logger.warning(f"TP1 no confirmado #{tid}")
                else:
                    cerrar_ids.append(tid)

        # Trailing stop dinámico
        if t.get('trailing_activado', False) and t['qty_restante'] > 0:
            if t['decision'] == 'Buy':
                if h > t['best_price']:
                    t['best_price'] = h
                    t['trailing_stop'] = t['best_price'] * (1 - TRAILING_PERCENT)
                if l <= t['trailing_stop']:
                    qty_restante = t['qty_restante']
                    result = close_position_qty_confirm(qty_restante, t['decision'])
                    if result and result != "already_closed":
                        pnl_resto = (t['trailing_stop'] - t['entrada']) * qty_restante
                        pnl_total = t['pnl_parcial'] + pnl_resto
                        REAL_BALANCE = get_real_balance()
                        TOTAL_TRADES += 1
                        if pnl_total > 0:
                            WIN_COUNT += 1
                        else:
                            LOSS_COUNT += 1
                        TRADE_HISTORY.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                        cerrar_ids.append(tid)
                        msg = f"📉 CIERRE #{tid} por Trailing Stop - PnL: {pnl_total:+.2f} USDT"
                        logger.info(msg)
                        telegram_mensaje(msg)
                        reporte_estado()
                    else:
                        logger.error(f"Falló cierre por trailing #{tid}")
            else:
                if l < t['best_price']:
                    t['best_price'] = l
                    t['trailing_stop'] = t['best_price'] * (1 + TRAILING_PERCENT)
                if h >= t['trailing_stop']:
                    qty_restante = t['qty_restante']
                    result = close_position_qty_confirm(qty_restante, t['decision'])
                    if result and result != "already_closed":
                        pnl_resto = (t['entrada'] - t['trailing_stop']) * qty_restante
                        pnl_total = t['pnl_parcial'] + pnl_resto
                        REAL_BALANCE = get_real_balance()
                        TOTAL_TRADES += 1
                        if pnl_total > 0:
                            WIN_COUNT += 1
                        else:
                            LOSS_COUNT += 1
                        TRADE_HISTORY.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                        cerrar_ids.append(tid)
                        msg = f"📉 CIERRE #{tid} por Trailing Stop - PnL: {pnl_total:+.2f} USDT"
                        logger.info(msg)
                        telegram_mensaje(msg)
                        reporte_estado()
                    else:
                        logger.error(f"Falló cierre por trailing #{tid}")

        # Stop loss inicial antes de TP1
        if not t['tp1_ejecutado'] and t['qty_restante'] > 0:
            if (t['decision']=="Buy" and l<=t['sl_actual']) or (t['decision']=="Sell" and h>=t['sl_actual']):
                qty_restante = t['qty_restante']
                result = close_position_qty_confirm(qty_restante, t['decision'])
                if result and result!="already_closed":
                    pnl_resto = (t['sl_actual']-t['entrada'])*qty_restante if t['decision']=="Buy" else (t['entrada']-t['sl_actual'])*qty_restante
                    pnl_total = t['pnl_parcial']+pnl_resto
                    REAL_BALANCE = get_real_balance()
                    TOTAL_TRADES+=1
                    if pnl_total>0:
                        WIN_COUNT+=1
                    else:
                        LOSS_COUNT+=1
                    TRADE_HISTORY.append(convertir_serializable({"pnl":pnl_total, "resultado_win":pnl_total>0, "decision":t['decision'], "razon":t['razon']}))
                    cerrar_ids.append(tid)
                    motivo = "Stop Loss inicial"
                    msg = f"🛑 CIERRE #{tid} por {motivo} - PnL: {pnl_total:+.2f} USDT"
                    logger.info(msg)
                    telegram_mensaje(msg)
                    reporte_estado()
                else:
                    logger.error(f"Falló cierre por stop #{tid}")

    for tid in cerrar_ids:
        del REAL_ACTIVE_TRADES[tid]

    if TOTAL_TRADES - ULTIMO_APRENDIZAJE >= 10:
        aprender_de_trades()

def aprender_de_trades():
    global REGLAS_APRENDIDAS, ULTIMO_APRENDIZAJE, ULTIMO_PROFIT_FACTOR
    try:
        if PAPER_TRADE:
            ult = paper_trade_history[-10:] if len(paper_trade_history)>=10 else paper_trade_history
            gan = sum(t['pnl'] for t in ult if t['pnl']>0)
            per = abs(sum(t['pnl'] for t in ult if t['pnl']<0))
            ULTIMO_PROFIT_FACTOR = gan/per if per>0 else 1.0
            winrate = (paper_win_count/paper_total_trades*100) if paper_total_trades>0 else 0
            resumen = f"📚 APRENDIZAJE PAPER #{paper_total_trades}\n🏆 Winrate: {winrate:.1f}% | ⚖️ PF: {ULTIMO_PROFIT_FACTOR:.2f}"
        else:
            ult = TRADE_HISTORY[-10:] if len(TRADE_HISTORY)>=10 else TRADE_HISTORY
            gan = sum(t['pnl'] for t in ult if t['pnl']>0)
            per = abs(sum(t['pnl'] for t in ult if t['pnl']<0))
            ULTIMO_PROFIT_FACTOR = gan/per if per>0 else 1.0
            winrate = (WIN_COUNT/TOTAL_TRADES*100) if TOTAL_TRADES>0 else 0
            resumen = f"📚 APRENDIZAJE #{TOTAL_TRADES}\n🏆 Winrate: {winrate:.1f}% | ⚖️ PF: {ULTIMO_PROFIT_FACTOR:.2f}"
        telegram_mensaje(resumen)

        ejemplos_exitosos = """
        Ejemplos de trades que funcionaron bien:
        - Buy en soporte claro con RSI sobrevendido y divergencia alcista.
        - Sell en resistencia con RSI sobrecomprado y patrón de estrella fugaz.
        - Mantener cuando el mercado está lateral entre EMAs.
        """
        try:
            ult_serial = convertir_serializable(ult)
            prompt = f"""Analiza estos últimos trades y extrae una lección detallada (máx 500 palabras) en español sobre qué condiciones funcionaron mejor y cuáles fallaron. Basa la lección en los ejemplos exitosos y en los datos.

Ejemplos exitosos:
{ejemplos_exitosos}

Historial reciente (últimos 10 trades):
{json.dumps(ult_serial, indent=2)}
"""
            # Usamos un modelo de texto más barato para el aprendizaje
            resp = client.chat.completions.create(
                model="google/gemini-2.0-flash",  # más económico
                messages=[{"role":"user","content":prompt}],
                timeout=45
            )
            REGLAS_APRENDIDAS = resp.choices[0].message.content
            telegram_mensaje(f"🧠 Nueva lección IA:\n{REGLAS_APRENDIDAS[:500]}...")
            with open("lecciones_aprendidas.txt", "a", encoding="utf-8") as f:
                f.write(f"\n\n--- Trade #{ULTIMO_APRENDIZAJE+1} ---\n{REGLAS_APRENDIDAS}\n")
        except Exception as e:
            logger.error(f"Error en aprendizaje detallado: {e}")
            REGLAS_APRENDIDAS = "Aún no hay lección detallada."
    except Exception as e:
        logger.error(f"Error aprendizaje: {e}")
    finally:
        ULTIMO_APRENDIZAJE = paper_total_trades if PAPER_TRADE else TOTAL_TRADES
        guardar_memoria()

def risk_management_check():
    global DAILY_START_BALANCE, STOPPED_TODAY, CURRENT_DAY
    hoy = datetime.now(timezone.utc).date()
    if CURRENT_DAY != hoy:
        CURRENT_DAY = hoy
        balance = paper_balance if PAPER_TRADE else (REAL_BALANCE or get_real_balance())
        DAILY_START_BALANCE = balance
        STOPPED_TODAY = False
        logger.info(f"📅 Nuevo día: {hoy}. Balance inicial: {balance:.2f}")
    balance_actual = paper_balance if PAPER_TRADE else REAL_BALANCE
    if balance_actual is not None and DAILY_START_BALANCE is not None:
        drawdown = (balance_actual - DAILY_START_BALANCE) / DAILY_START_BALANCE
        if drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
            STOPPED_TODAY = True
            logger.warning("⚠️ Drawdown diario superado. Operaciones detenidas.")
    return not STOPPED_TODAY

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global REAL_BALANCE, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, TRADE_HISTORY, REAL_ACTIVE_TRADES
    global paper_balance, paper_trade_counter, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions, ultima_vela
    global DECISION_CACHE

    cargar_memoria()
    set_leverage()
    sync_active_trades_with_bybit()

    telegram_mensaje("🤖 Bot iniciado - Claude Sonnet 4.5 - Scalping 3m/30m con caché y gestión inteligente")
    logger.info("Bot iniciado")

    if PAPER_TRADE:
        logger.info(f"📄 PAPER TRADE - Saldo: {paper_balance:.2f} USDT")
        telegram_mensaje(f"📄 Bot Paper Trade Online - Saldo simulado: {paper_balance:.2f} USDT")
    else:
        REAL_BALANCE = get_real_balance()
        if REAL_BALANCE is None:
            logger.error("No se pudo obtener saldo real. Abortando.")
            return
        logger.info(f"💰 BOT REAL - Balance: {REAL_BALANCE:.2f} USDT")
        telegram_mensaje(f"💰 Bot Real Online - Balance: {REAL_BALANCE:.2f} USDT")

    ultima_vela = None
    sideways_counter = 0

    while True:
        try:
            if int(time.time()) % 300 < 5:
                logger.info("❤️ Bot heartbeat")

            df_ltf_raw = obtener_velas(INTERVAL_LTF)
            df_htf_raw = obtener_velas(INTERVAL_HTF)
            if df_ltf_raw.empty or df_htf_raw.empty:
                time.sleep(SLEEP_SECONDS)
                continue
            df_ltf = calcular_indicadores(df_ltf_raw)
            df_htf = calcular_indicadores(df_htf_raw)
            if df_ltf.empty or df_htf.empty:
                time.sleep(SLEEP_SECONDS)
                continue

            precio_actual = df_ltf['close'].iloc[-1]
            if not PAPER_TRADE:
                REAL_BALANCE = get_real_balance()
            max_trades_actual = get_dynamic_max_trades()
            active_count = len(paper_positions) if PAPER_TRADE else len(REAL_ACTIVE_TRADES)

            # Detectar mercado lateral
            if is_sideways_market(df_ltf):
                sideways_counter += 1
                if sideways_counter >= SIDEWAYS_SKIP_CYCLES:
                    logger.info("📉 Mercado lateral detectado. Saltando análisis para ahorrar API.")
                    time.sleep(SLEEP_SECONDS)
                    continue
            else:
                sideways_counter = 0

            vela_actual_time = df_ltf.index[-1]
            es_vela_nueva = (ultima_vela is None) or (ultima_vela != vela_actual_time)

            if es_vela_nueva and active_count < max_trades_actual and risk_management_check():
                # Generar clave de caché
                cache_key = get_cache_key(df_ltf, df_htf)
                decision_cache = get_cached_decision(cache_key)
                if decision_cache:
                    dec, raz, explicacion, entry_ia, sl_ia, tp1_ia, conf = decision_cache
                else:
                    sop_ltf, res_ltf, slope_ltf, inter_ltf, _, _ = detectar_zonas_mercado(df_ltf)
                    sop_htf, res_htf, slope_htf, inter_htf, _, _ = detectar_zonas_mercado(df_htf)

                    img_ltf = generar_grafico_para_vision(df_ltf, "BTC/USDT 3m (LTF)", sop_ltf, res_ltf, slope_ltf, inter_ltf, excluir_actual=True)
                    img_htf = generar_grafico_para_vision(df_htf, "BTC/USDT 30m (HTF)", sop_htf, res_htf, slope_htf, inter_htf, excluir_actual=True)

                    if img_ltf and img_htf:
                        dec, raz, explicacion, entry_ia, sl_ia, tp1_ia, conf = analizar_con_claude(img_ltf, img_htf, df_ltf, df_htf)
                        # Guardar en caché
                        store_cached_decision(cache_key, (dec, raz, explicacion, entry_ia, sl_ia, tp1_ia, conf))
                        logger.info(f"🧠 Decisión IA: {dec} - Razón: {raz} - Confianza: {conf}")
                    else:
                        logger.error("No se pudieron generar los gráficos")
                        time.sleep(SLEEP_SECONDS)
                        continue

                if dec != "Hold" and conf >= 40:  # Umbral de confianza más bajo para Claude
                    abrir_posicion_con_ia(dec, precio_actual, raz, explicacion, sl_ia, tp1_ia,
                                          df_ltf, sop_ltf, res_ltf, slope_ltf, inter_ltf)
                elif dec != "Hold":
                    logger.warning(f"🚫 Señal {dec} rechazada: confianza baja ({conf})")
                    telegram_mensaje(f"🚫 Señal {dec} rechazada: confianza baja ({conf})")
                else:
                    logger.info(f"IA decidió HOLD. Motivo: {raz[:100]}")

                ultima_vela = vela_actual_time

            # Gestión de trades activos (siempre)
            if PAPER_TRADE and paper_positions:
                revisar_sl_tp_simulado(df_ltf, precio_actual)
            elif not PAPER_TRADE and REAL_ACTIVE_TRADES:
                real_revisar_sl_tp(df_ltf, precio_actual)

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            logger.error(f"ERROR CRÍTICO: {e}", exc_info=True)
            time.sleep(30)

if __name__ == '__main__':
    run_bot()
