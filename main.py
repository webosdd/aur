# BOT TRADING CON GEMINI 3.1 FLASH + TA-Lib (61 patrones de velas)
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
import signal
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =================== TA-LIB (patrones de velas) ===================
try:
    import talib
    TA_LIB_AVAILABLE = True
    logger.info("✅ TA-Lib cargado correctamente. 61 patrones de velas disponibles.")
except ImportError:
    TA_LIB_AVAILABLE = False
    logger.warning("⚠️ TA-Lib no instalado. Se omitirá la detección de patrones.")
    # Definimos stub para evitar errores
    class talib:
        @staticmethod
        def CDLDOJI(*args): return np.array([0])

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
MODELO_VISION = "google/gemini-3.1-flash-image-preview"

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

# =================== CONFIGURACIÓN DEL BOT ===================
SYMBOL = "BTCUSDT"
INTERVAL_LTF = "3"
INTERVAL_HTF = "30"
RISK_PER_TRADE_MAX = 3.0
LEVERAGE = 34
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 3
MIN_MARGIN_PER_TRADE = 3.0
TP1_PERCENT = 0.5
TRAILING_PERCENT = 0.0015
MIN_SL_DIST_PCT = 0.0005
MAX_SL_DIST_PCT = 0.005
MIN_TP1_DIST_PCT = 0.002
MAX_TP1_DIST_PCT = 0.006

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

ULTIMO_REPORTE = 0

# =================== MEMORIA PERSISTENTE ===================
MEMORY_FILE = "memoria_bot_paper.json" if PAPER_TRADE else "memoria_bot_real.json"

def convertir_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: convertir_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convertir_serializable(item) for item in obj]
    return obj

def normalizar_trade_activo(trade):
    defaults = {
        'tp1': 0.0,
        'tp1_ejecutado': False,
        'trailing_activado': False,
        'best_price': trade.get('entrada', 0.0),
        'trailing_stop': None,
        'pnl_parcial': 0.0,
        'qty_restante': trade.get('qty_original', 0.0),
        'sl_actual': trade.get('sl_inicial', 0.0),
        'razon': '',
        'decision': 'Hold',
        'entrada': 0.0,
        'qty_original': 0.0,
        'order_id': None,
        'breakeven_activado': False,
        'contexto_pensamiento': ''
    }
    for key, val in defaults.items():
        if key not in trade:
            trade[key] = val
    return trade

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
                "pnl_parcial": t.get("pnl_parcial", 0.0),
                "tp1": t.get("tp1", 0.0),
                "sl_inicial": t.get("sl_inicial", 0.0),
                "order_id": t.get("order_id")
            }
        data = {
            "TRADE_HISTORY": paper_trade_history,
            "REAL_BALANCE": paper_balance,
            "WIN_COUNT": paper_win_count,
            "LOSS_COUNT": paper_loss_count,
            "TOTAL_TRADES": paper_total_trades,
            "ACTIVE_TRADES_META": active_trades_meta,
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
                "pnl_parcial": t.get("pnl_parcial", 0.0),
                "tp1": t.get("tp1", 0.0),
                "sl_inicial": t.get("sl_inicial", 0.0),
                "order_id": t.get("order_id")
            }
        data = {
            "TRADE_HISTORY": TRADE_HISTORY,
            "REAL_BALANCE": REAL_BALANCE,
            "WIN_COUNT": WIN_COUNT,
            "LOSS_COUNT": LOSS_COUNT,
            "TOTAL_TRADES": TOTAL_TRADES,
            "ACTIVE_TRADES_META": active_trades_meta,
        }
    try:
        tmp_file = MEMORY_FILE + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(convertir_serializable(data), f, indent=4)
        os.replace(tmp_file, MEMORY_FILE)
        logger.info("💾 Memoria guardada (escritura atómica)")
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
            paper_positions = {}
            for tid, meta in active_meta.items():
                trade = normalizar_trade_activo(meta)
                paper_positions[int(tid)] = trade
        else:
            TRADE_HISTORY = data.get("TRADE_HISTORY", [])
            REAL_BALANCE = data.get("REAL_BALANCE", None)
            WIN_COUNT = data.get("WIN_COUNT", 0)
            LOSS_COUNT = data.get("LOSS_COUNT", 0)
            TOTAL_TRADES = data.get("TOTAL_TRADES", 0)
            active_meta = data.get("ACTIVE_TRADES_META", {})
            REAL_ACTIVE_TRADES = {}
            for tid, meta in active_meta.items():
                trade = normalizar_trade_activo(meta)
                REAL_ACTIVE_TRADES[int(tid)] = trade
        logger.info(f"📂 Memoria cargada. Trades: {paper_total_trades if PAPER_TRADE else TOTAL_TRADES}")
    except Exception as e:
        logger.error(f"Error cargando memoria: {e}")

ULTIMO_GUARDADO = 0

def guardado_periodico():
    global ULTIMO_GUARDADO
    ahora = time.time()
    if ahora - ULTIMO_GUARDADO > 300:
        guardar_memoria()
        ULTIMO_GUARDADO = ahora

def guardar_y_salir(signum, frame):
    logger.info("Señal SIGTERM recibida. Guardando memoria antes de salir...")
    guardar_memoria()
    sys.exit(0)

signal.signal(signal.SIGTERM, guardar_y_salir)

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
        f"🎯 Activos: {active_trades}/{max_din}"
    )
    telegram_mensaje(mensaje)

# =================== INDICADORES Y GRÁFICOS ===================
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
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
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
    fig, ax = plt.subplots(figsize=(16,8))
    x = np.arange(len(df_plot))
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
    if soporte:
        ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte')
    if resistencia:
        ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    if slope is not None and intercept is not None and slope != 0:
        x_trend = np.array([0, len(df_plot)-1])
        y_trend = intercept + slope * x_trend
        ax.plot(x_trend, y_trend, color='white', linestyle='-.', lw=2, label='Tendencia', alpha=0.7)
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
    plt.close(fig)
    plt.close('all')
    return img

def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

def parse_json_seguro(raw):
    if not raw or raw.strip() == "":
        return None
    try:
        repaired = json_repair.repair_json(raw)
        return json.loads(repaired)
    except:
        return None

# =================== DETECCIÓN DE PATRONES CON TA-Lib (61 patrones) ===================
def obtener_todos_patrones_talib(df: pd.DataFrame) -> dict:
    """
    Ejecuta todas las funciones de patrones de TA-Lib sobre el DataFrame.
    Retorna un diccionario con {nombre_patron: señal (100/-100/0)} para la última vela.
    """
    if not TA_LIB_AVAILABLE or df.empty or len(df) < 3:
        return {}
    
    # Asegurar nombres de columnas en mayúsculas como requiere TA-Lib
    open_ = df['open'].values.astype(float)
    high = df['high'].values.astype(float)
    low = df['low'].values.astype(float)
    close = df['close'].values.astype(float)
    volume = df['volume'].values.astype(float) if 'volume' in df else None
    
    # Lista de todas las funciones de patrones (61 según documentación)
    patrones = {
        # 1. Doji
        'CDLDOJI': talib.CDLDOJI,
        'CDLDOJISTAR': talib.CDLDOJISTAR,
        'CDLDRAGONFLYDOJI': talib.CDLDRAGONFLYDOJI,
        'CDLGRAVESTONEDOJI': talib.CDLGRAVESTONEDOJI,
        'CDLLONGLEGGEDDOJI': talib.CDLLONGLEGGEDDOJI,
        'CDLRICKSHAWMAN': talib.CDLRICKSHAWMAN,
        # 2. Patrones de una vela
        'CDLHAMMER': talib.CDLHAMMER,
        'CDLSHOOTINGSTAR': talib.CDLSHOOTINGSTAR,
        'CDLHANGINGMAN': talib.CDLHANGINGMAN,
        'CDLINVERTEDHAMMER': talib.CDLINVERTEDHAMMER,
        'CDLMARUBOZU': talib.CDLMARUBOZU,
        'CDLBELTHOLD': talib.CDLBELTHOLD,
        'CDLTAKURI': talib.CDLTAKURI,
        'CDLSPINNINGTOP': talib.CDLSPINNINGTOP,
        # 3. Patrones de dos velas
        'CDLENGULFING': talib.CDLENGULFING,
        'CDLHARAMI': talib.CDLHARAMI,
        'CDLHARAMICROSS': talib.CDLHARAMICROSS,
        'CDLPIERCING': talib.CDLPIERCING,
        'CDLDARKCLOUDCOVER': talib.CDLDARKCLOUDCOVER,
        'CDLTAKURI': talib.CDLTAKURI,  # repetido pero se ejecuta
        'CDLKICKING': talib.CDLKICKING,
        'CDLKICKINGBYLENGTH': talib.CDLKICKINGBYLENGTH,
        'CDL2CROWS': talib.CDL2CROWS,
        'CDL3WHITESOLDIERS': talib.CDL3WHITESOLDIERS,
        'CDL3BLACKCROWS': talib.CDL3BLACKCROWS,
        'CDLEVENINGSTAR': talib.CDLEVENINGSTAR,
        'CDLMORNINGSTAR': talib.CDLMORNINGSTAR,
        'CDLEVENINGDOJISTAR': talib.CDLEVENINGDOJISTAR,
        'CDLMORNINGDOJISTAR': talib.CDLMORNINGDOJISTAR,
        'CDL3INSIDE': talib.CDL3INSIDE,
        'CDL3OUTSIDE': talib.CDL3OUTSIDE,
        'CDL3LINESTRIKE': talib.CDL3LINESTRIKE,
        'CDLABANDONEDBABY': talib.CDLABANDONEDBABY,
        'CDLIDENTICAL3CROWS': talib.CDLIDENTICAL3CROWS,
        'CDLUPSIDEGAP2CROWS': talib.CDLUPSIDEGAP2CROWS,
        'CDLUNIQUE3RIVER': talib.CDLUNIQUE3RIVER,
        'CDLLADDERBOTTOM': talib.CDLLADDERBOTTOM,
        'CDLCONCEALBABYSWALL': talib.CDLCONCEALBABYSWALL,
        'CDLBREAKAWAY': talib.CDLBREAKAWAY,
        'CDLMATHOLD': talib.CDLMATHOLD,
        'CDLHIKKAKE': talib.CDLHIKKAKE,
        'CDLHIKKAKEMOD': talib.CDLHIKKAKEMOD,
        'CDLHIGHWAVE': talib.CDLHIGHWAVE,
        'CDLTHRUSTING': talib.CDLTHRUSTING,
        'CDLCLOSINGMARUBOZU': talib.CDLCLOSINGMARUBOZU,
        'CDLSTALLEDPATTERN': talib.CDLSTALLEDPATTERN,
        'CDLTASUKIGAP': talib.CDLTASUKIGAP,
        'CDLINNECK': talib.CDLINNECK,
        'CDLNECK': talib.CDLNECK,
        'CDLONNECK': talib.CDLONNECK,
        'CDLRISEFALL3METHODS': talib.CDLRISEFALL3METHODS,
        'CDLSEPARATINGLINES': talib.CDLSEPARATINGLINES,
        'CDLPIERCING': talib.CDLPIERCING,  # ya estaba
        'CDLLONGLINE': talib.CDLLONGLINE,
        'CDLSHORTLINE': talib.CDLSHORTLINE,
        'CDLSTICKSANDWICH': talib.CDLSTICKSANDWICH,
        'CDLKEEPINGLINE': talib.CDLKEEPINGLINE,
        'CDLKANNA': talib.CDLKANNA,
        'CDLRIFFMAN': talib.CDLRIFFMAN
    }
    
    resultados = {}
    for nombre, func in patrones.items():
        try:
            # Llamar a la función correspondiente
            if volume is not None and nombre in ['CDLDOJI', 'CDLHAMMER']:  # algunas no usan volumen
                señal_array = func(open_, high, low, close)
            else:
                señal_array = func(open_, high, low, close)
            if len(señal_array) > 0:
                señal = señal_array[-1]  # última vela
                if señal != 0:
                    resultados[nombre] = int(señal)  # 100, -100, o 200/ -200 en algunos
        except Exception as e:
            logger.debug(f"Error en patrón {nombre}: {e}")
            continue
    return resultados

def resumen_patrones_texto(patrones_dict: dict) -> str:
    """Convierte el diccionario de patrones en un texto legible para el prompt."""
    if not patrones_dict:
        return "No se detectaron patrones de velas significativos."
    bullish = [p for p, s in patrones_dict.items() if s > 0]
    bearish = [p for p, s in patrones_dict.items() if s < 0]
    texto = ""
    if bullish:
        texto += f"Patrones ALCISTAS detectados (última vela): {', '.join(bullish)}.\n"
    if bearish:
        texto += f"Patrones BAJISTAS detectados (última vela): {', '.join(bearish)}.\n"
    if not bullish and not bearish:
        texto = "Patrón neutro o sin patrón claro.\n"
    return texto

# =================== PROMPT PARA GEMINI CON PATRONES ===================
def analizar_con_gemini(img_ltf, img_htf, patrones_ltf_texto, patrones_htf_texto):
    try:
        img_ltf_b64 = pil_to_base64(img_ltf)
        img_htf_b64 = pil_to_base64(img_htf)

        prompt = f"""
Eres un trader profesional de crypto especializado en scalping con velas de 3 minutos.
Te voy a mostrar dos gráficos de BTC/USDT: el primero es de 3 minutos (velas recientes), el segundo es de 30 minutos (tendencia de mayor plazo).

Además, se ha realizado un análisis automático de patrones de velas japonesas (TA-Lib) en ambos marcos:

🔍 **Patrones detectados en gráfico de 3m (LTF):**
{patrones_ltf_texto}

🔍 **Patrones detectados en gráfico de 30m (HTF):**
{patrones_htf_texto}

Analiza los gráficos junto con la información de patrones. Debes usar los patrones como CONFIRMACIÓN adicional, no como única señal. Prioriza los niveles de SOPORTE/RESISTENCIA y la relación con EMA20.

Reglas:
- SOLO recomienda COMPRAR si: precio está cerca de SOPORTE (distancia <0.3%) Y hay señal alcista en patrones O el precio rebota claramente en el gráfico.
- SOLO recomienda VENDER si: precio está cerca de RESISTENCIA (distancia <0.3%) Y hay señal bajista en patrones O el precio rechaza claramente.
- Si los patrones contradicen lo que ves en el gráfico (ej: patrón alcista pero precio cayendo sin soporte), la decisión debe ser HOLD.
- Si no hay patrones claros pero los niveles son muy fuertes, puedes operar con menos confianza.

Devuelve ÚNICAMENTE un JSON válido en una línea con esta estructura:
{{
  "decision": "Buy/Sell/Hold",
  "razon": "explicación breve (max 150 chars)",
  "explicacion": "análisis detallado en español incluyendo qué patrón influyó",
  "entry_price": 0.0,
  "sl_price": 0.0,
  "tp1_price": 0.0,
  "confianza": 0-100
}}
Si la decisión es Hold, los precios deben ser 0.0.
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
        
        return decision, razon, explicacion, entry_price, sl_price, tp1_price, confianza
    except Exception as e:
        logger.error(f"Error en IA: {e}")
        return "Hold", f"Error: {str(e)[:50]}", "", None, None, None, 0

# =================== VALIDACIÓN Y AJUSTE DE PRECIOS ===================
def ajustar_sl_tp_para_scalping(decision, entrada, sl_ia, tp1_ia):
    if sl_ia is None or sl_ia <= 0:
        distancia_sl = entrada * 0.002
        sl_ajustado = entrada - distancia_sl if decision == "Buy" else entrada + distancia_sl
    else:
        if decision == "Buy":
            distancia_raw = entrada - sl_ia
        else:
            distancia_raw = sl_ia - entrada
        min_sl = entrada * MIN_SL_DIST_PCT
        max_sl = entrada * MAX_SL_DIST_PCT
        if distancia_raw < min_sl:
            distancia_final = min_sl
        elif distancia_raw > max_sl:
            distancia_final = max_sl
        else:
            distancia_final = distancia_raw
        sl_ajustado = entrada - distancia_final if decision == "Buy" else entrada + distancia_final

    if tp1_ia is None or tp1_ia <= 0:
        distancia_tp1 = entrada * 0.003
        tp1_ajustado = entrada + distancia_tp1 if decision == "Buy" else entrada - distancia_tp1
    else:
        if decision == "Buy":
            distancia_raw = tp1_ia - entrada
        else:
            distancia_raw = entrada - tp1_ia
        min_tp = entrada * MIN_TP1_DIST_PCT
        max_tp = entrada * MAX_TP1_DIST_PCT
        if distancia_raw < min_tp:
            distancia_final_tp = min_tp
        elif distancia_raw > max_tp:
            distancia_final_tp = max_tp
        else:
            distancia_final_tp = distancia_raw
        tp1_ajustado = entrada + distancia_final_tp if decision == "Buy" else entrada - distancia_final_tp
    return sl_ajustado, tp1_ajustado

# =================== GESTIÓN DE RIESGO Y APERTURA ===================
def calcular_riesgo_dinamico(free_margin):
    if free_margin >= 20:
        return RISK_PER_TRADE_MAX
    elif free_margin >= 10:
        return 1.5
    else:
        return 1.0

def abrir_posicion_con_ia(decision, precio_actual, razon, contexto, sl_ia, tp1_ia, df_ltf, sop, res, slope, inter):
    global paper_balance, paper_positions, paper_trade_counter, REAL_BALANCE, TRADE_COUNTER, REAL_ACTIVE_TRADES
    global paper_total_trades, TOTAL_TRADES

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

    entrada = precio_actual
    sl_ajustado, tp1_ajustado = ajustar_sl_tp_para_scalping(decision, entrada, sl_ia, tp1_ia)

    if decision == "Buy":
        distancia_final = entrada - sl_ajustado
    else:
        distancia_final = sl_ajustado - entrada

    if distancia_final <= 0:
        logger.error("Distancia SL inválida. Cancelando.")
        return

    qty_btc = risk_per_trade / distancia_final
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
        positions[trade_id] = normalizar_trade_activo({
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_inicial": sl_ajustado, "sl_actual": sl_ajustado,
            "tp1": tp1_ajustado,
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto,
            "trailing_activado": False,
            "best_price": entrada,
            "trailing_stop": None
        })
        modo = "📄 PAPER"
    else:
        order_id = place_market_order(decision, qty_btc)
        if not order_id:
            logger.error("No se pudo abrir orden real.")
            return
        TRADE_COUNTER += 1
        trade_id = TRADE_COUNTER
        positions[trade_id] = normalizar_trade_activo({
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_inicial": sl_ajustado, "sl_actual": sl_ajustado,
            "tp1": tp1_ajustado,
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto,
            "trailing_activado": False,
            "best_price": entrada,
            "trailing_stop": None
        })
        modo = "💰 REAL"

    msg = (f"{modo} [#{trade_id}] {decision} MARKET en {entrada:.2f} | Qty {qty_btc} BTC (riesgo {risk_per_trade} USDT)\n"
           f"🛑 SL: {sl_ajustado:.2f} (dist {distancia_final:.1f} USD)\n"
           f"🎯 TP1: {tp1_ajustado:.2f} | Trailing tras TP1: {TRAILING_PERCENT*100:.2f}%\n"
           f"📝 Razón: {razon}\n"
           f"💰 Margen requerido: {margen_necesario:.2f} USDT | Libre disponible: {free_margin:.2f} USDT")
    logger.info(msg)
    telegram_mensaje(msg)

    titulo_grafico = f"Entrada - {modo} #{trade_id}"
    img_completa = generar_grafico_para_vision(df_ltf, titulo_grafico, sop, res, slope, inter,
                                               entry_price=entrada, sl_price=sl_ajustado,
                                               tp1_price=tp1_ajustado, side=decision, excluir_actual=False)
    if img_completa:
        img_completa.save("/tmp/in_completo.png")
        caption = (f"{modo} #{trade_id} {decision}\n"
                   f"Entry: {entrada:.2f} | SL: {sl_ajustado:.2f} | TP1: {tp1_ajustado:.2f}")
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

# =================== REPORTE CADA 10 TRADES ===================
def enviar_reporte_cada_10_trades():
    global ULTIMO_REPORTE
    if PAPER_TRADE:
        total = paper_total_trades
        win = paper_win_count
        loss = paper_loss_count
        balance = paper_balance
        history = paper_trade_history
    else:
        total = TOTAL_TRADES
        win = WIN_COUNT
        loss = LOSS_COUNT
        balance = REAL_BALANCE
        history = TRADE_HISTORY

    if total - ULTIMO_REPORTE >= 10 and total > 0:
        ultimos = history[-10:] if len(history) >= 10 else history
        ganancias = sum(t['pnl'] for t in ultimos if t['pnl'] > 0)
        perdidas = abs(sum(t['pnl'] for t in ultimos if t['pnl'] < 0))
        pf = ganancias / perdidas if perdidas > 0 else 0.0
        winrate_10 = (sum(1 for t in ultimos if t['pnl'] > 0) / len(ultimos)) * 100 if ultimos else 0
        modo = "📄 PAPER" if PAPER_TRADE else "💰 REAL"
        msg = (f"{modo} - REPORTE CADA 10 TRADES\n"
               f"📊 Trades totales: {total}\n"
               f"🏆 Winrate últimos 10: {winrate_10:.1f}%\n"
               f"⚖️ Profit Factor últimos 10: {pf:.2f}\n"
               f"💰 Balance actual: {balance:.2f} USDT\n"
               f"📈 Winrate global: {(win/(win+loss)*100) if (win+loss)>0 else 0:.1f}%")
        telegram_mensaje(msg)
        logger.info(msg)
        ULTIMO_REPORTE = total

# =================== GESTIÓN DE TRADES ACTIVOS ===================
def revisar_sl_tp_simulado(df, precio_actual):
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    if not paper_positions:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    cerrar_ids = []
    for tid, t in list(paper_positions.items()):
        tp1 = t.get('tp1', 0.0)
        tp1_ejecutado = t.get('tp1_ejecutado', False)
        qty_restante = t.get('qty_restante', 0.0)
        decision = t.get('decision', 'Hold')
        entrada = t.get('entrada', 0.0)
        sl_actual = t.get('sl_actual', 0.0)
        qty_original = t.get('qty_original', 0.0)
        razon = t.get('razon', '')
        trailing_activado = t.get('trailing_activado', False)
        best_price = t.get('best_price', entrada)
        
        # TP1
        if not tp1_ejecutado and tp1 > 0:
            if (decision == "Buy" and h >= tp1) or (decision == "Sell" and l <= tp1):
                qty_tp1 = round(qty_original * TP1_PERCENT, 3)
                if qty_tp1 >= 0.001 and qty_restante > 0:
                    pnl_parcial = (tp1 - entrada) * qty_tp1 if decision == "Buy" else (entrada - tp1) * qty_tp1
                    comision = abs(pnl_parcial) * PAPER_COMMISSION_PCT
                    pnl_parcial -= comision
                    t['pnl_parcial'] = t.get('pnl_parcial', 0.0) + pnl_parcial
                    t['qty_restante'] = round(qty_original - qty_tp1, 3)
                    t['tp1_ejecutado'] = True
                    t['trailing_activado'] = True
                    t['best_price'] = entrada
                    t['trailing_stop'] = entrada * (1 - TRAILING_PERCENT) if decision == "Buy" else entrada * (1 + TRAILING_PERCENT)
                    logger.info(f"✅ PAPER: TP1 #{tid} +{pnl_parcial:.2f} USDT, trailing activado")
                    telegram_mensaje(f"✅ PAPER TP1 #{tid}: +{pnl_parcial:.2f} USDT. Trailing stop activado.")
                    if t['qty_restante'] <= 0.0001:
                        cerrar_ids.append(tid)
                else:
                    cerrar_ids.append(tid)

        # Trailing stop
        if t.get('trailing_activado', False) and t['qty_restante'] > 0:
            if decision == 'Buy':
                if h > best_price:
                    t['best_price'] = h
                    t['trailing_stop'] = h * (1 - TRAILING_PERCENT)
                if l <= t['trailing_stop']:
                    qty_restante = t['qty_restante']
                    pnl_resto = (t['trailing_stop'] - entrada) * qty_restante
                    comision = abs(pnl_resto) * PAPER_COMMISSION_PCT
                    pnl_resto -= comision
                    pnl_total = t.get('pnl_parcial', 0.0) + pnl_resto
                    paper_balance += pnl_total
                    paper_total_trades += 1
                    if pnl_total > 0:
                        paper_win_count += 1
                    else:
                        paper_loss_count += 1
                    paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": decision, "razon": razon}))
                    cerrar_ids.append(tid)
                    msg = f"📉 PAPER CIERRE #{tid} por Trailing Stop - PnL: {pnl_total:+.2f} USDT"
                    logger.info(msg)
                    telegram_mensaje(msg)
                    reporte_estado()
                    enviar_reporte_cada_10_trades()
            else:
                if l < best_price:
                    t['best_price'] = l
                    t['trailing_stop'] = l * (1 + TRAILING_PERCENT)
                if h >= t['trailing_stop']:
                    qty_restante = t['qty_restante']
                    pnl_resto = (entrada - t['trailing_stop']) * qty_restante
                    comision = abs(pnl_resto) * PAPER_COMMISSION_PCT
                    pnl_resto -= comision
                    pnl_total = t.get('pnl_parcial', 0.0) + pnl_resto
                    paper_balance += pnl_total
                    paper_total_trades += 1
                    if pnl_total > 0:
                        paper_win_count += 1
                    else:
                        paper_loss_count += 1
                    paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": decision, "razon": razon}))
                    cerrar_ids.append(tid)
                    msg = f"📉 PAPER CIERRE #{tid} por Trailing Stop - PnL: {pnl_total:+.2f} USDT"
                    logger.info(msg)
                    telegram_mensaje(msg)
                    reporte_estado()
                    enviar_reporte_cada_10_trades()

        # Stop loss inicial
        if not t.get('tp1_ejecutado', False) and t['qty_restante'] > 0:
            cond = (decision == "Buy" and l <= sl_actual) or (decision == "Sell" and h >= sl_actual)
            if cond:
                qty_restante = t['qty_restante']
                pnl_resto = (sl_actual - entrada) * qty_restante if decision == "Buy" else (entrada - sl_actual) * qty_restante
                comision = abs(pnl_resto) * PAPER_COMMISSION_PCT
                pnl_resto -= comision
                pnl_total = t.get('pnl_parcial', 0.0) + pnl_resto
                paper_balance += pnl_total
                paper_total_trades += 1
                if pnl_total>0:
                    paper_win_count+=1
                else:
                    paper_loss_count+=1
                paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": decision, "razon": razon}))
                cerrar_ids.append(tid)
                motivo = "Stop Loss inicial"
                msg = f"🛑 PAPER CIERRE #{tid} por {motivo} - PnL: {pnl_total:+.2f} USDT"
                logger.info(msg)
                telegram_mensaje(msg)
                reporte_estado()
                enviar_reporte_cada_10_trades()

    for tid in cerrar_ids:
        del paper_positions[tid]

def real_revisar_sl_tp(df, precio_actual):
    global REAL_BALANCE, WIN_COUNT, LOSS_COUNT, TOTAL_TRADES, TRADE_HISTORY, REAL_ACTIVE_TRADES
    if not REAL_ACTIVE_TRADES:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    cerrar_ids = []
    for tid, t in list(REAL_ACTIVE_TRADES.items()):
        tp1 = t.get('tp1', 0.0)
        tp1_ejecutado = t.get('tp1_ejecutado', False)
        decision = t.get('decision', 'Hold')
        entrada = t.get('entrada', 0.0)
        sl_actual = t.get('sl_actual', 0.0)
        qty_original = t.get('qty_original', 0.0)
        qty_restante = t.get('qty_restante', 0.0)
        razon = t.get('razon', '')
        trailing_activado = t.get('trailing_activado', False)
        best_price = t.get('best_price', entrada)

        if not tp1_ejecutado and tp1 > 0:
            if (decision == "Buy" and h >= tp1) or (decision == "Sell" and l <= tp1):
                qty_tp1 = round(qty_original * TP1_PERCENT, 3)
                if qty_tp1 >= 0.001 and qty_restante > 0:
                    result = close_position_qty_confirm(qty_tp1, decision)
                    if result and result != "already_closed":
                        pnl_parcial = (tp1 - entrada) * qty_tp1 if decision == "Buy" else (entrada - tp1) * qty_tp1
                        t['pnl_parcial'] = t.get('pnl_parcial', 0.0) + pnl_parcial
                        t['qty_restante'] = round(qty_original - qty_tp1, 3)
                        t['tp1_ejecutado'] = True
                        t['trailing_activado'] = True
                        t['best_price'] = entrada
                        t['trailing_stop'] = entrada * (1 - TRAILING_PERCENT) if decision == "Buy" else entrada * (1 + TRAILING_PERCENT)
                        logger.info(f"✅ TP1 #{tid} +{pnl_parcial:.2f} USDT, trailing activado")
                        telegram_mensaje(f"✅ TP1 #{tid}: +{pnl_parcial:.2f} USDT. Trailing activado.")
                        if t['qty_restante'] <= 0.0001:
                            cerrar_ids.append(tid)
                    else:
                        logger.warning(f"TP1 no confirmado #{tid}")
                else:
                    cerrar_ids.append(tid)

        if t.get('trailing_activado', False) and t['qty_restante'] > 0:
            if decision == 'Buy':
                if h > best_price:
                    t['best_price'] = h
                    t['trailing_stop'] = h * (1 - TRAILING_PERCENT)
                if l <= t['trailing_stop']:
                    qty_restante = t['qty_restante']
                    result = close_position_qty_confirm(qty_restante, decision)
                    if result and result != "already_closed":
                        pnl_resto = (t['trailing_stop'] - entrada) * qty_restante
                        pnl_total = t.get('pnl_parcial', 0.0) + pnl_resto
                        REAL_BALANCE = get_real_balance()
                        TOTAL_TRADES += 1
                        if pnl_total > 0:
                            WIN_COUNT += 1
                        else:
                            LOSS_COUNT += 1
                        TRADE_HISTORY.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": decision, "razon": razon}))
                        cerrar_ids.append(tid)
                        msg = f"📉 CIERRE #{tid} por Trailing Stop - PnL: {pnl_total:+.2f} USDT"
                        logger.info(msg)
                        telegram_mensaje(msg)
                        reporte_estado()
                        enviar_reporte_cada_10_trades()
                    else:
                        logger.error(f"Falló cierre por trailing #{tid}")
            else:
                if l < best_price:
                    t['best_price'] = l
                    t['trailing_stop'] = l * (1 + TRAILING_PERCENT)
                if h >= t['trailing_stop']:
                    qty_restante = t['qty_restante']
                    result = close_position_qty_confirm(qty_restante, decision)
                    if result and result != "already_closed":
                        pnl_resto = (entrada - t['trailing_stop']) * qty_restante
                        pnl_total = t.get('pnl_parcial', 0.0) + pnl_resto
                        REAL_BALANCE = get_real_balance()
                        TOTAL_TRADES += 1
                        if pnl_total > 0:
                            WIN_COUNT += 1
                        else:
                            LOSS_COUNT += 1
                        TRADE_HISTORY.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": decision, "razon": razon}))
                        cerrar_ids.append(tid)
                        msg = f"📉 CIERRE #{tid} por Trailing Stop - PnL: {pnl_total:+.2f} USDT"
                        logger.info(msg)
                        telegram_mensaje(msg)
                        reporte_estado()
                        enviar_reporte_cada_10_trades()
                    else:
                        logger.error(f"Falló cierre por trailing #{tid}")

        if not t.get('tp1_ejecutado', False) and t['qty_restante'] > 0:
            cond = (decision == "Buy" and l <= sl_actual) or (decision == "Sell" and h >= sl_actual)
            if cond:
                qty_restante = t['qty_restante']
                result = close_position_qty_confirm(qty_restante, decision)
                if result and result != "already_closed":
                    pnl_resto = (sl_actual - entrada) * qty_restante if decision == "Buy" else (entrada - sl_actual) * qty_restante
                    pnl_total = t.get('pnl_parcial', 0.0) + pnl_resto
                    REAL_BALANCE = get_real_balance()
                    TOTAL_TRADES += 1
                    if pnl_total > 0:
                        WIN_COUNT += 1
                    else:
                        LOSS_COUNT += 1
                    TRADE_HISTORY.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": decision, "razon": razon}))
                    cerrar_ids.append(tid)
                    motivo = "Stop Loss inicial"
                    msg = f"🛑 CIERRE #{tid} por {motivo} - PnL: {pnl_total:+.2f} USDT"
                    logger.info(msg)
                    telegram_mensaje(msg)
                    reporte_estado()
                    enviar_reporte_cada_10_trades()
                else:
                    logger.error(f"Falló cierre por stop #{tid}")

    for tid in cerrar_ids:
        del REAL_ACTIVE_TRADES[tid]

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
    global REAL_BALANCE, TRADE_HISTORY, REAL_ACTIVE_TRADES
    global paper_balance, paper_trade_counter, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions, ultima_vela
    global ULTIMO_REPORTE

    cargar_memoria()
    set_leverage()
    sync_active_trades_with_bybit()

    if PAPER_TRADE:
        ULTIMO_REPORTE = paper_total_trades
    else:
        ULTIMO_REPORTE = TOTAL_TRADES

    telegram_mensaje("🤖 Bot iniciado - Gemini 3.1 Flash + TA-Lib (61 patrones) - Scalping 3m/30m")
    logger.info("Bot iniciado con detección de 61 patrones de velas (TA-Lib)")

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
            guardado_periodico()

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

            if es_vela_nueva and active_count < max_trades_actual and risk_management_check():
                sop_ltf, res_ltf, slope_ltf, inter_ltf, _, _ = detectar_zonas_mercado(df_ltf)
                sop_htf, res_htf, slope_htf, inter_htf, _, _ = detectar_zonas_mercado(df_htf)

                # === DETECCIÓN DE PATRONES CON TA-Lib ===
                patrones_ltf = obtener_todos_patrones_talib(df_ltf)
                patrones_htf = obtener_todos_patrones_talib(df_htf)
                patrones_ltf_texto = resumen_patrones_texto(patrones_ltf)
                patrones_htf_texto = resumen_patrones_texto(patrones_htf)
                logger.debug(f"Patrones LTF: {patrones_ltf_texto}")
                logger.debug(f"Patrones HTF: {patrones_htf_texto}")

                img_ltf = generar_grafico_para_vision(df_ltf, "BTC/USDT 3m (LTF)", sop_ltf, res_ltf, slope_ltf, inter_ltf, excluir_actual=True)
                img_htf = generar_grafico_para_vision(df_htf, "BTC/USDT 30m (HTF)", sop_htf, res_htf, slope_htf, inter_htf, excluir_actual=True)

                if img_ltf and img_htf:
                    dec, raz, explicacion, entry_ia, sl_ia, tp1_ia, conf = analizar_con_gemini(
                        img_ltf, img_htf, patrones_ltf_texto, patrones_htf_texto
                    )
                    logger.info(f"🧠 Decisión IA: {dec} - Razón: {raz} - Confianza: {conf}")

                    # Filtro por niveles (soporte/resistencia)
                    if dec != "Hold":
                        sop_ltf_actual, res_ltf_actual, _, _, _, _ = detectar_zonas_mercado(df_ltf, idx=-1)
                        sop_htf_actual, res_htf_actual, _, _, _, _ = detectar_zonas_mercado(df_htf, idx=-1)
                        umbral_distancia = 0.0025  # 0.25%
                        if dec == "Buy":
                            cerca_soporte = False
                            if sop_ltf_actual and abs(precio_actual - sop_ltf_actual) / precio_actual < umbral_distancia:
                                cerca_soporte = True
                            if sop_htf_actual and abs(precio_actual - sop_htf_actual) / precio_actual < umbral_distancia:
                                cerca_soporte = True
                            if not cerca_soporte:
                                logger.warning(f"🚫 Señal Buy rechazada: precio {precio_actual} no está cerca de soporte")
                                telegram_mensaje("🚫 Buy rechazado por falta de soporte cercano")
                                dec = "Hold"
                        elif dec == "Sell":
                            cerca_resistencia = False
                            if res_ltf_actual and abs(res_ltf_actual - precio_actual) / precio_actual < umbral_distancia:
                                cerca_resistencia = True
                            if res_htf_actual and abs(res_htf_actual - precio_actual) / precio_actual < umbral_distancia:
                                cerca_resistencia = True
                            if not cerca_resistencia:
                                logger.warning(f"🚫 Señal Sell rechazada: precio {precio_actual} no está cerca de resistencia")
                                telegram_mensaje("🚫 Sell rechazado por falta de resistencia cercana")
                                dec = "Hold"

                    # Validación extra de consistencia entre patrón y decisión (opcional)
                    if dec != "Hold":
                        # Verificar que al menos haya un patrón que coincida con la dirección
                        patrones_coinciden = False
                        if dec == "Buy":
                            if any(v > 0 for v in patrones_ltf.values()) or any(v > 0 for v in patrones_htf.values()):
                                patrones_coinciden = True
                        elif dec == "Sell":
                            if any(v < 0 for v in patrones_ltf.values()) or any(v < 0 for v in patrones_htf.values()):
                                patrones_coinciden = True
                        if not patrones_coinciden:
                            # Si no hay patrón que respalde, reducimos confianza pero no cancelamos si los niveles son buenos
                            logger.info(f"⚠️ Decisión {dec} sin patrón de confirmación. Confianza reducida.")
                            conf = max(conf - 15, 0)
                            if conf < 20:
                                logger.warning(f"🚫 Señal {dec} anulada por falta de patrón de confirmación y baja confianza.")
                                dec = "Hold"

                    if dec != "Hold":
                        if conf >= 30:
                            abrir_posicion_con_ia(dec, precio_actual, raz, explicacion, sl_ia, tp1_ia,
                                                  df_ltf, sop_ltf, res_ltf, slope_ltf, inter_ltf)
                        else:
                            logger.warning(f"🚫 Señal {dec} rechazada: confianza baja ({conf})")
                            telegram_mensaje(f"🚫 Señal {dec} rechazada: confianza baja ({conf})")
                    else:
                        logger.info(f"IA decidió HOLD. Motivo: {raz[:100]}")
                else:
                    logger.error("No se pudieron generar los gráficos")

                ultima_vela = vela_actual_time

            if PAPER_TRADE and paper_positions:
                revisar_sl_tp_simulado(df_ltf, precio_actual)
            elif not PAPER_TRADE and REAL_ACTIVE_TRADES:
                real_revisar_sl_tp(df_ltf, precio_actual)

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            logger.error(f"ERROR CRÍTICO: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == '__main__':
    run_bot()
