# BOT TRADING CON CLAUDE SONNET 4.5 - SCALPING 3M/30M CON IA LIBRE Y GESTIÓN AVANZADA
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
from collections import OrderedDict
from ta.momentum import RSIIndicator
from ta.trend import MACD

# =================== CONFIGURACIÓN DE LOGGING MEJORADA ===================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_trading.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TradingBot")

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
        "X-Title": "Trading Bot Claude",
    }
)
MODELO_VISION = "anthropic/claude-sonnet-4.5"  # Modelo definitivo

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

# =================== CONFIGURACIÓN DEL BOT (SIN REGLAS FIJAS) ===================
SYMBOL = "BTCUSDT"
INTERVAL_LTF = "3"
INTERVAL_HTF = "30"
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 3
MIN_MARGIN_PER_TRADE = 3.0
TP1_PERCENT = 0.5          # Porcentaje del tamaño a cerrar en TP1
TRAILING_PERCENT = 0.0015  # 0.15% trailing stop

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

# =================== CACHÉ DE DECISIONES (para evitar IA repetitiva) ===================
class DecisionCache:
    def __init__(self, max_size=20, similarity_threshold=0.005):  # 0.5% de similitud
        self.cache = OrderedDict()
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold

    def _hash(self, df_ltf, df_htf):
        # Usar último precio y RSI/MACD aproximados como clave
        last_close = df_ltf['close'].iloc[-1]
        last_rsi = df_ltf['rsi'].iloc[-1] if 'rsi' in df_ltf.columns else 50
        last_macd = df_ltf['macd'].iloc[-1] if 'macd' in df_ltf.columns else 0
        # Redondear para agrupar condiciones similares
        close_bin = round(last_close / 100) * 100
        rsi_bin = round(last_rsi / 5) * 5
        macd_bin = round(last_macd / 50) * 50
        return f"{close_bin}_{rsi_bin}_{macd_bin}"

    def get(self, df_ltf, df_htf):
        key = self._hash(df_ltf, df_htf)
        if key in self.cache:
            logger.info(f"✅ Usando decisión en caché para condiciones similares (clave {key})")
            return self.cache[key]
        return None

    def set(self, df_ltf, df_htf, decision_data):
        key = self._hash(df_ltf, df_htf)
        self.cache[key] = decision_data
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)
        logger.debug(f"Decisión guardada en caché con clave {key}")

decision_cache = DecisionCache()

# =================== DETECCIÓN DE MERCADO LATERAL (para ahorrar costes) ===================
def is_sideways_market(df, threshold_pct=0.003):
    """
    Determina si el mercado está lateral basado en el rango de precios reciente.
    threshold_pct: 0.3% de rango en las últimas 20 velas.
    """
    if len(df) < 20:
        return False
    recent_high = df['high'].tail(20).max()
    recent_low = df['low'].tail(20).min()
    range_pct = (recent_high - recent_low) / recent_low
    return range_pct < threshold_pct

# =================== FUNCIONES BYBIT (sin cambios significativos) ===================
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
        logger.info("📊 Paper trade: apalancamiento simulado 34x")
        return
    try:
        body = {"category": "linear", "symbol": "BTCUSDT", "buyLeverage": "34", "sellLeverage": "34"}
        result = bybit_request("/v5/position/set-leverage", method="POST", body=body)
        ret_code = result.get('retCode')
        if ret_code == 0 or ret_code == 110043:
            logger.info("🔧 Apalancamiento 34x configurado")
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
            margin_used += (t['qty_original'] * t['entrada']) / LEVERAGE  # LEVERAGE está definido antes? Lo añadimos
        LEVERAGE = 34
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

# =================== MEMORIA PERSISTENTE OPTIMIZADA (sin duplicar dataframes) ===================
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
    if PAPER_TRADE:
        active_trades_meta = {}
        for tid, t in paper_positions.items():
            active_trades_meta[tid] = {
                "id": t["id"], "decision": t["decision"], "entrada": t["entrada"],
                "razon": t.get("razon", ""), "tp1_ejecutado": t["tp1_ejecutado"],
                "sl_actual": t.get("sl_actual"),
                "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
                "breakeven_activado": t.get("breakeven_activado", False),
                "trailing_activado": t.get("trailing_activado", False),
                "best_price": t.get("best_price"),
                "trailing_stop": t.get("trailing_stop"),
                "pnl_parcial": t.get("pnl_parcial", 0.0)
            }
        data = {
            "TRADE_HISTORY": paper_trade_history,
            "REAL_BALANCE": paper_balance,
            "WIN_COUNT": paper_win_count,
            "LOSS_COUNT": paper_loss_count,
            "TOTAL_TRADES": paper_total_trades,
            "ACTIVE_TRADES_META": active_trades_meta
        }
    else:
        active_trades_meta = {}
        for tid, t in REAL_ACTIVE_TRADES.items():
            active_trades_meta[tid] = {
                "id": t["id"], "decision": t["decision"], "entrada": t["entrada"],
                "razon": t.get("razon", ""), "tp1_ejecutado": t["tp1_ejecutado"],
                "sl_actual": t.get("sl_actual"),
                "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
                "breakeven_activado": t.get("breakeven_activado", False),
                "trailing_activado": t.get("trailing_activado", False),
                "best_price": t.get("best_price"),
                "trailing_stop": t.get("trailing_stop"),
                "pnl_parcial": t.get("pnl_parcial", 0.0)
            }
        data = {
            "TRADE_HISTORY": TRADE_HISTORY,
            "REAL_BALANCE": REAL_BALANCE,
            "WIN_COUNT": WIN_COUNT,
            "LOSS_COUNT": LOSS_COUNT,
            "TOTAL_TRADES": TOTAL_TRADES,
            "ACTIVE_TRADES_META": active_trades_meta
        }
    try:
        with open(MEMORY_FILE, "w") as f:
            json.dump(convertir_serializable(data), f, indent=4)
        logger.info("💾 Memoria guardada")
    except Exception as e:
        logger.error(f"Error guardando memoria: {e}")

def cargar_memoria():
    global TRADE_HISTORY, REAL_BALANCE, WIN_COUNT, LOSS_COUNT, TOTAL_TRADES, REAL_ACTIVE_TRADES
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
            paper_positions = {int(k): v for k, v in active_meta.items()}
        else:
            TRADE_HISTORY = data.get("TRADE_HISTORY", [])
            REAL_BALANCE = data.get("REAL_BALANCE", None)
            WIN_COUNT = data.get("WIN_COUNT", 0)
            LOSS_COUNT = data.get("LOSS_COUNT", 0)
            TOTAL_TRADES = data.get("TOTAL_TRADES", 0)
            active_meta = data.get("ACTIVE_TRADES_META", {})
            REAL_ACTIVE_TRADES = {int(k): v for k, v in active_meta.items()}
        logger.info(f"📂 Memoria cargada. Trades: {paper_total_trades if PAPER_TRADE else TOTAL_TRADES}")
    except Exception as e:
        logger.error(f"Error cargando memoria: {e}")

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
    max_trades = get_dynamic_max_trades()
    modo = "📄 PAPER" if PAPER_TRADE else "💰 REAL"
    mensaje = (
        f"{modo} ESTADO BTC\n"
        f"💰 Balance: {balance:.2f} USDT\n"
        f"📈 PnL día: {pnl_global:+.2f} USDT\n"
        f"🏆 Winrate: {winrate:.1f}%\n"
        f"🎯 Activos: {active_trades}/{max_trades}"
    )
    telegram_mensaje(mensaje)

# =================== INDICADORES (RSI, MACD, EMA) ===================
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
    if df.empty:
        return df
    # EMAs
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    # RSI (14 periodos)
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    # MACD (12,26,9)
    macd_ind = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    df['macd_diff'] = macd_ind.macd_diff()
    return df

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

# =================== GENERACIÓN DE GRÁFICO (con RSI y MACD incluidos) ===================
def generar_grafico_para_vision(df, titulo, soporte=None, resistencia=None, slope=None, intercept=None,
                                entry_price=None, sl_price=None, tp1_price=None, side=None, excluir_actual=False):
    if df.empty:
        return None
    if excluir_actual and len(df) > 1:
        df_plot = df.iloc[:-1].tail(GRAFICO_VELAS_LIMIT).copy()
    else:
        df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
    if len(df_plot) < 3:
        return None
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), sharex=True, 
                                        gridspec_kw={'height_ratios': [3, 1, 1]})
    # Gráfico de velas
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
    if entry_price is not None:
        ax1.axhline(entry_price, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='Entry')
        circle_color = 'lime' if side == 'Buy' else 'red'
        ax1.scatter(x[-1], entry_price, color=circle_color, s=100, edgecolors='white', zorder=5)
        ax1.annotate(f'Entry {entry_price:.0f}', xy=(x[-1], entry_price), xytext=(5, 5),
                     textcoords='offset points', fontsize=9, color='white',
                     bbox=dict(boxstyle='round,pad=0.2', fc='black', alpha=0.6))
    if sl_price is not None:
        ax1.axhline(sl_price, color='red', linestyle='--', linewidth=2, label=f'SL {sl_price:.0f}')
    if tp1_price is not None:
        ax1.axhline(tp1_price, color='lime', linestyle='--', linewidth=2, label=f'TP1 {tp1_price:.0f}')
    
    # RSI
    if 'rsi' in df_plot.columns:
        ax2.plot(x, df_plot['rsi'], color='cyan', lw=1.5)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI', color='white')
        ax2.set_ylim(0, 100)
    
    # MACD
    if 'macd' in df_plot.columns:
        ax3.plot(x, df_plot['macd'], color='blue', lw=1.5, label='MACD')
        ax3.plot(x, df_plot['macd_signal'], color='red', lw=1.5, label='Signal')
        ax3.bar(x, df_plot['macd_diff'], color='gray', alpha=0.5, label='Histogram')
        ax3.set_ylabel('MACD', color='white')
        ax3.legend(loc='upper left')
    
    titulo_limpio = sanitize_for_matplotlib(titulo)
    ax1.set_title(titulo_limpio, color='white', fontsize=14)
    ax1.set_ylabel('Precio (USDT)', color='white')
    ax1.tick_params(colors='white')
    ax2.tick_params(colors='white')
    ax3.tick_params(colors='white')
    for spine in ax1.spines.values():
        spine.set_color('white')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')
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

# =================== PROMPT PARA CLAUDE (LIBERTAD TOTAL) ===================
def analizar_con_claude(img_ltf, img_htf):
    try:
        img_ltf_b64 = pil_to_base64(img_ltf)
        img_htf_b64 = pil_to_base64(img_htf)

        prompt = f"""
Eres un trader profesional de crypto con décadas de experiencia. Tu tarea es analizar los gráficos de BTC/USDT en dos temporalidades:
- Gráfico 1: Velas de 3 minutos (LTF) – muestra la acción de precio reciente.
- Gráfico 2: Velas de 30 minutos (HTF) – muestra la tendencia de mayor plazo.

Tienes libertad total para interpretar lo que ves. No existen reglas fijas. Eres un humano observando el mercado. Debes considerar:
- Acción del precio: patrones de velas (martillo, estrella, engulfing, doji, etc.), mechas largas, cierres.
- Niveles clave: soportes y resistencias claros (dinámicos o estáticos), zonas de oferta/demanda.
- Indicadores mostrados: EMA20, EMA50, RSI, MACD.
- Estructura de mercado: ¿tendencia alcista, bajista, lateral? ¿Microestructura?
- Volumen (aunque no se ve directamente, puedes intuir por la fuerza de los movimientos).
- Contexto: ¿el precio está rechazando un nivel? ¿Está rompiendo con fuerza? ¿Hay divergencias en RSI/MACD?
- Confluencias: si múltiples factores apuntan en la misma dirección, mejor.

Decide si es momento de **COMPRAR (Buy)**, **VENDER (Sell)** o **NO HACER NADA (Hold)**.

Si decides Buy o Sell, debes proporcionar:
- Precio de entrada (puede ser el precio actual o un nivel específico que veas en el gráfico).
- Stop loss: nivel donde la operación quedaría invalidada.
- Take profit 1 (TP1): primer objetivo parcial (luego el bot cerrará una parte y activará trailing stop).

Importante: NO uses porcentajes fijos. Ajusta SL y TP según la estructura del mercado (volatilidad, soportes/resistencias cercanos, mechas, etc.). Sé preciso.

Además, explica detalladamente tu razonamiento: qué ves, por qué tomas esa decisión, qué te da confianza.

Responde ÚNICAMENTE con un JSON válido en una línea con esta estructura:
{{
  "decision": "Buy/Sell/Hold",
  "entry_price": 0.0,
  "sl_price": 0.0,
  "tp1_price": 0.0,
  "razon": "explicación breve (max 150 chars)",
  "analisis_completo": "análisis detallado en español (sin límite)"
}}
Si la decisión es Hold, los precios deben ser 0.0.
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
        contenido = response.choices[0].message.content
        # Extraer JSON
        json_match = re.search(r'\{.*\}(?=\s*$)', contenido, re.DOTALL)
        if json_match:
            datos = json.loads(json_match.group())
        else:
            datos = json.loads(contenido)
        decision = datos.get("decision", "Hold")
        entry = datos.get("entry_price", 0.0)
        sl = datos.get("sl_price", 0.0)
        tp1 = datos.get("tp1_price", 0.0)
        razon = datos.get("razon", "")
        analisis = datos.get("analisis_completo", "")
        return decision, razon, analisis, entry, sl, tp1
    except Exception as e:
        logger.error(f"Error en IA: {e}")
        return "Hold", f"Error: {str(e)[:50]}", "", 0.0, 0.0, 0.0

# =================== APERTURA DE POSICIÓN (SIN REGLAS FIJAS DE SL/TP) ===================
def abrir_posicion_con_ia(decision, precio_actual, razon, analisis, entry_ia, sl_ia, tp1_ia, df_ltf):
    global paper_balance, paper_positions, paper_trade_counter, REAL_BALANCE, TRADE_COUNTER, REAL_ACTIVE_TRADES

    if decision not in ["Buy", "Sell"]:
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

    # Usar los precios dados por la IA (si son válidos), si no, usar precio actual + lógica por defecto
    entrada = entry_ia if entry_ia and entry_ia > 0 else precio_actual
    sl = sl_ia if sl_ia and sl_ia > 0 else (entrada * 0.995 if decision == "Buy" else entrada * 1.005)
    tp1 = tp1_ia if tp1_ia and tp1_ia > 0 else (entrada * 1.005 if decision == "Buy" else entrada * 0.995)

    # Asegurar que SL y TP tengan sentido
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

    distancia_sl = abs(entrada - sl)
    qty_btc = risk_per_trade / distancia_sl
    LEVERAGE = 34
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
            "analisis_ia": analisis,
            "trailing_activado": False,
            "best_price": entrada,
            "trailing_stop": None
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
            "analisis_ia": analisis,
            "trailing_activado": False,
            "best_price": entrada,
            "trailing_stop": None
        }
        modo = "💰 REAL"

    msg = (f"{modo} [#{trade_id}] {decision} en {entrada:.2f} | Qty {qty_btc} BTC (riesgo {risk_per_trade} USDT)\n"
           f"🛑 SL: {sl:.2f} | 🎯 TP1: {tp1:.2f}\n"
           f"📝 Razón: {razon}\n"
           f"💰 Margen: {margen_necesario:.2f} USDT | Libre: {free_margin:.2f} USDT")
    logger.info(msg)
    telegram_mensaje(msg)

    # Enviar gráfico con niveles
    sop, res, slope, inter, _, _ = detectar_zonas_mercado(df_ltf)
    img_completa = generar_grafico_para_vision(df_ltf, f"Entrada {modo} #{trade_id}", sop, res, slope, inter,
                                               entry_price=entrada, sl_price=sl, tp1_price=tp1, side=decision)
    if img_completa:
        img_completa.save("/tmp/in_completo.png")
        caption = f"{modo} #{trade_id} {decision}\nEntry: {entrada:.2f} | SL: {sl:.2f} | TP1: {tp1:.2f}"
        telegram_enviar_imagen("/tmp/in_completo.png", caption)

    guardar_memoria()

def calcular_riesgo_dinamico(free_margin):
    if free_margin >= 20:
        return 3.0
    elif free_margin >= 10:
        return 1.5
    else:
        return 1.0

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

# =================== GESTIÓN DE TRADES ACTIVOS (TRAILING TRAS TP1) ===================
def revisar_sl_tp_simulado(df, precio_actual):
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    if not paper_positions:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    cerrar_ids = []
    for tid, t in list(paper_positions.items()):
        # TP1: cerrar % definido (TP1_PERCENT)
        if not t['tp1_ejecutado'] and t['tp1'] is not None and t['tp1'] > 0:
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

        # Trailing stop (solo si activado después de TP1)
        if t.get('trailing_activado', False) and t['qty_restante'] > 0:
            if t['decision'] == 'Buy':
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
        if not t.get('tp1_ejecutado', False) and t['qty_restante'] > 0:
            cond = (t['decision']=="Buy" and l <= t['sl_actual']) or (t['decision']=="Sell" and h >= t['sl_actual'])
            if cond:
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
    guardar_memoria()

def real_revisar_sl_tp(df, precio_actual):
    global REAL_BALANCE, WIN_COUNT, LOSS_COUNT, TOTAL_TRADES, TRADE_HISTORY, REAL_ACTIVE_TRADES
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

        # Trailing stop
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
            cond = (t['decision']=="Buy" and l<=t['sl_actual']) or (t['decision']=="Sell" and h>=t['sl_actual'])
            if cond:
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

# =================== LOOP PRINCIPAL CON CACHÉ Y DETECCIÓN DE LATERALIDAD ===================
def run_bot():
    global REAL_BALANCE, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, TRADE_HISTORY, REAL_ACTIVE_TRADES
    global paper_balance, paper_trade_counter, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions, ultima_vela

    cargar_memoria()
    set_leverage()
    sync_active_trades_with_bybit()

    telegram_mensaje("🤖 Bot iniciado - Claude Sonnet 4.5 - Scalping 3m/30m con IA libre")
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

            vela_actual_time = df_ltf.index[-1]
            es_vela_nueva = (ultima_vela is None) or (ultima_vela != vela_actual_time)

            # SALTAR SI MERCADO LATERAL PARA AHORRAR COSTES
            if is_sideways_market(df_ltf, threshold_pct=0.003):
                logger.info("⏸️ Mercado lateral detectado. Saltando análisis para ahorrar costes.")
                time.sleep(SLEEP_SECONDS * 2)
                continue

            if es_vela_nueva and active_count < max_trades_actual and risk_management_check():
                # Verificar caché primero
                cached = decision_cache.get(df_ltf, df_htf)
                if cached:
                    dec, raz, analisis, entry, sl, tp1 = cached
                    logger.info(f"♻️ Usando decisión cachead: {dec}")
                else:
                    sop_ltf, res_ltf, slope_ltf, inter_ltf, _, _ = detectar_zonas_mercado(df_ltf)
                    sop_htf, res_htf, slope_htf, inter_htf, _, _ = detectar_zonas_mercado(df_htf)

                    img_ltf = generar_grafico_para_vision(df_ltf, "BTC/USDT 3m (LTF)", sop_ltf, res_ltf, slope_ltf, inter_ltf, excluir_actual=True)
                    img_htf = generar_grafico_para_vision(df_htf, "BTC/USDT 30m (HTF)", sop_htf, res_htf, slope_htf, inter_htf, excluir_actual=True)

                    if img_ltf and img_htf:
                        dec, raz, analisis, entry, sl, tp1 = analizar_con_claude(img_ltf, img_htf)
                        # Guardar en caché
                        decision_cache.set(df_ltf, df_htf, (dec, raz, analisis, entry, sl, tp1))
                    else:
                        logger.error("No se pudieron generar los gráficos")
                        time.sleep(SLEEP_SECONDS)
                        continue

                if dec != "Hold":
                    abrir_posicion_con_ia(dec, precio_actual, raz, analisis, entry, sl, tp1, df_ltf)
                else:
                    logger.info(f"IA decidió HOLD. Razón: {raz[:100]}")

                ultima_vela = vela_actual_time

            # Gestión de trades activos
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
