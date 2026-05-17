# BOT TRADING CON FILTROS NUMÉRICOS, CONFIRMACIÓN INTELIGENTE Y APRENDIZAJE CONTINUO
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

# Configurar logging para Railway (con soporte de emojis)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =================== SANITIZAR EMOJIS SOLO PARA MATPLOTLIB ===================
def sanitize_for_matplotlib(text):
    """Elimina emojis y caracteres no soportados por DejaVu Sans."""
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
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
if not SILICONFLOW_API_KEY:
    raise ValueError("Falta SILICONFLOW_API_KEY")

SILICONFLOW_BASE_URL = "https://api.siliconflow.com/v1"
client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)
MODELO_VISION = "Qwen/Qwen3-VL-32B-Instruct"

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BASE_URL = "https://api.bybit.com"

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("Faltan BYBIT_API_KEY o BYBIT_API_SECRET")

# =================== MODO PAPER TRADE ===================
PAPER_TRADE = True  # Cambiar a False para operar real

# Estado simulado para paper trading
paper_balance = 1000.0
paper_positions = {}
paper_trade_counter = 0
paper_win_count = 0
paper_loss_count = 0
paper_total_trades = 0
paper_trade_history = []

# Simular comisión (0.1% por orden)
PAPER_COMMISSION_PCT = 0.001

# =================== CONFIGURACIÓN DEL BOT ===================
SYMBOL = "BTCUSDT"
INTERVAL_LTF = "5"
INTERVAL_HTF = "60"
RISK_PER_TRADE_MAX = 3.0
LEVERAGE = 34
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 3
MIN_MARGIN_PER_TRADE = 3.0
TP1_PERCENT = 0.5
TRAILING_PERCENT = 0.005      # 0.5% trailing stop (aumentado desde 0.3%)
MIN_CIERRE_EMA_PCT = 0.001    # 0.1% por encima de EMA para considerar "cierre válido"
MIN_SL_DIST_PCT = 0.0005
MAX_SL_DIST_PCT = 0.01
ATR_MULTIPLIER_SL = 1.5       # Mínimo SL = 1.5 * ATR

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

# Señales pendientes de confirmación (solo si la IA lo solicita)
senal_pendiente = None  # dict con todos los datos de la señal

# =================== FUNCIONES BYBIT (igual que antes) ===================
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

# ====== MEMORIA PERSISTENTE ======
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
                "sl_actual": t.get("sl_actual"),
                "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
                "breakeven_activado": t.get("breakeven_activado", False),
                "trailing_activado": t.get("trailing_activado", False),
                "best_price": t.get("best_price"),
                "trailing_stop": t.get("trailing_stop")
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
                "sl_actual": t.get("sl_actual"),
                "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
                "breakeven_activado": t.get("breakeven_activado", False),
                "trailing_activado": t.get("trailing_activado", False),
                "best_price": t.get("best_price"),
                "trailing_stop": t.get("trailing_stop")
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
    partes = [texto[i:i+4000] for i in range(0, len(texto), 4000)]
    for parte in partes:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": parte}, timeout=10)
            if resp.status_code != 200:
                logger.error(f"Error Telegram: {resp.status_code} - {resp.text[:200]}")
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

# =================== INDICADORES Y GRÁFICOS ===================
def obtener_velas(interval="5", limit=150):
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
    if df.empty or len(df) < 40:
        return 0,0,0,0,"LATERAL","LATERAL"
    df_eval = df if idx == -1 else df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
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
    # Para la IA, excluir la vela actual si está en formación (solo velas cerradas)
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
    
    # Forzar colores claros
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

# =================== DETECCIÓN DE PATRONES DE VELAS ===================
def es_martillo(df, idx):
    """Martillo: cuerpo pequeño inferior, mecha inferior larga, poca o ninguna mecha superior."""
    vela = df.iloc[idx]
    cuerpo = abs(vela['close'] - vela['open'])
    sombra_inf = min(vela['close'], vela['open']) - vela['low']
    sombra_sup = vela['high'] - max(vela['close'], vela['open'])
    return sombra_inf > 2 * cuerpo and sombra_sup < cuerpo * 0.3

def es_estrella_fugaz(df, idx):
    """Estrella fugaz: cuerpo pequeño superior, mecha superior larga, poca mecha inferior."""
    vela = df.iloc[idx]
    cuerpo = abs(vela['close'] - vela['open'])
    sombra_sup = vela['high'] - max(vela['close'], vela['open'])
    sombra_inf = min(vela['close'], vela['open']) - vela['low']
    return sombra_sup > 2 * cuerpo and sombra_inf < cuerpo * 0.3

def es_tres_soldados_blancos(df, idx):
    """Tres velas alcistas consecutivas con cierres crecientes y cuerpos grandes."""
    if idx < 2:
        return False
    v1 = df.iloc[idx-2]
    v2 = df.iloc[idx-1]
    v3 = df.iloc[idx]
    return (v1['close'] > v1['open'] and v2['close'] > v2['open'] and v3['close'] > v3['open'] and
            v2['close'] > v1['close'] and v3['close'] > v2['close'] and
            (v1['close'] - v1['open']) > 0.002 * v1['close'] and
            (v2['close'] - v2['open']) > 0.002 * v2['close'] and
            (v3['close'] - v3['open']) > 0.002 * v3['close'])

def es_tres_cuervos_negros(df, idx):
    """Tres velas bajistas consecutivas con cierres decrecientes y cuerpos grandes."""
    if idx < 2:
        return False
    v1 = df.iloc[idx-2]
    v2 = df.iloc[idx-1]
    v3 = df.iloc[idx]
    return (v1['close'] < v1['open'] and v2['close'] < v2['open'] and v3['close'] < v3['open'] and
            v2['close'] < v1['close'] and v3['close'] < v2['close'] and
            (v1['open'] - v1['close']) > 0.002 * v1['close'] and
            (v2['open'] - v2['close']) > 0.002 * v2['close'] and
            (v3['open'] - v3['close']) > 0.002 * v3['close'])

def es_engulfing_alcista(df, idx):
    """Engulfing alcista: vela actual verde que engulle completamente la vela anterior roja."""
    if idx < 1:
        return False
    v_ant = df.iloc[idx-1]
    v_act = df.iloc[idx]
    return (v_ant['close'] < v_ant['open'] and v_act['close'] > v_act['open'] and
            v_act['open'] < v_ant['close'] and v_act['close'] > v_ant['open'])

def es_engulfing_bajista(df, idx):
    """Engulfing bajista: vela actual roja que engulle completamente la vela anterior verde."""
    if idx < 1:
        return False
    v_ant = df.iloc[idx-1]
    v_act = df.iloc[idx]
    return (v_ant['close'] > v_ant['open'] and v_act['close'] < v_act['open'] and
            v_act['open'] > v_ant['close'] and v_act['close'] < v_ant['open'])

# =================== PROMPT ESTRICTO CON CONFIRMACIÓN INTELIGENTE ===================
def analizar_con_qwen(img_ltf, img_htf):
    global TOKENS_ACUMULADOS
    try:
        img_ltf_b64 = pil_to_base64(img_ltf)
        img_htf_b64 = pil_to_base64(img_htf)

        prompt = f"""
Eres un trader profesional. Analiza SOLO velas CERRADAS (las últimas 5-10 velas completas del gráfico, no la que se está formando).
Mira los dos gráficos de BTC/USDT: 5 minutos (primera imagen) y 1 hora (segunda imagen).

Decide Buy, Sell o Hold siguiendo estas reglas estrictas:

- Para **Buy**: 
  * Opción A: Rechazo claro de soporte (vela larga verde después de tocar soporte) O
  * Opción B: Rompimiento de resistencia confirmado con patrón alcista (ej. martillo, tres soldados, engulfing alcista) y la vela cierra por encima de la resistencia.
  * Opción C: Retroceso a EMA20 en tendencia alcista fuerte (pendiente HTF > 0.05) con vela de rechazo (pin bar o martillo).

- Para **Sell**: 
  * Opción A: Rechazo claro de resistencia (vela larga roja después de tocar resistencia) O
  * Opción B: Rompimiento de soporte confirmado con patrón bajista (ej. estrella fugaz, tres cuervos, engulfing bajista) y la vela cierra por debajo del soporte.
  * Opción C: Retroceso a EMA20 en tendencia bajista fuerte (pendiente HTF < -0.05) con vela de rechazo.

No te fíes de simples toques a la EMA o niveles. Debe haber una vela de confirmación con cuerpo grande o patrón.

Proporciona los precios y los siguientes campos:

- "cierre_sobre_ema20_ltf": true/false
- "cierre_bajo_ema20_ltf": true/false
- "rechazo_soporte": true/false
- "rechazo_resistencia": true/false
- "tendencia_htf": "alcista"/"bajista"/"lateral"
- "confirmacion_necesaria": true/false (marca true si el contexto es dudoso)
- "tipo_setup": "rechazo" o "rompimiento" o "retroceso_ema"

Devuelve ÚNICAMENTE un JSON en una línea con esta estructura:

{{
  "decision": "Buy/Sell/Hold",
  "razon": "Frase corta pero descriptiva (max 150) ej: 'Rechazo claro en soporte 77500 con martillo + tendencia alcista'",
  "explicacion": "análisis detallado en español",
  "entry_price": 0.0,
  "sl_price": 0.0,
  "tp1_price": 0.0,
  "confianza": 0-100,
  "cierre_sobre_ema20_ltf": false,
  "cierre_bajo_ema20_ltf": false,
  "rechazo_soporte": false,
  "rechazo_resistencia": false,
  "tendencia_htf": "",
  "confirmacion_necesaria": false,
  "tipo_setup": ""
}}

Si no hay condiciones claras, decision = Hold y los precios a 0.
"""
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": img_ltf_b64}},
                {"type": "image_url", "image_url": {"url": img_htf_b64}}
            ]}],
            temperature=0.2
        )
        TOKENS_ACUMULADOS += response.usage.total_tokens if response.usage else 0
        contenido = response.choices[0].message.content
        json_match = re.search(r'\{.*\}(?=\s*$)', contenido, re.DOTALL)
        if json_match:
            datos = parse_json_seguro(json_match.group())
        else:
            datos = parse_json_seguro(contenido)
        if not datos:
            return "Hold", "Error parsing", "", None, None, None, 0, False, False, False, False, "", False, ""

        decision = datos.get("decision", "Hold")
        razon = datos.get("razon", "")
        explicacion = datos.get("explicacion", "")
        entry_price = datos.get("entry_price")
        sl_price = datos.get("sl_price")
        tp1_price = datos.get("tp1_price")
        confianza = datos.get("confianza", 0)
        cierre_sobre = datos.get("cierre_sobre_ema20_ltf", False)
        cierre_bajo = datos.get("cierre_bajo_ema20_ltf", False)
        rechazo_sop = datos.get("rechazo_soporte", False)
        rechazo_res = datos.get("rechazo_resistencia", False)
        tend_htf = datos.get("tendencia_htf", "lateral")
        confirmacion_necesaria = datos.get("confirmacion_necesaria", False)
        tipo_setup = datos.get("tipo_setup", "")
        
        return decision, razon, explicacion, entry_price, sl_price, tp1_price, confianza, cierre_sobre, cierre_bajo, rechazo_sop, rechazo_res, tend_htf, confirmacion_necesaria, tipo_setup
    except Exception as e:
        logger.error(f"Error en IA: {e}")
        return "Hold", "Error API", "", None, None, None, 0, False, False, False, False, "", False, ""

# =================== FILTRO NUMÉRICO POST-IA ===================
def validar_setup_ia(decision, razon, df_ltf, df_htf, cierre_sobre_ia, cierre_bajo_ia, rechazo_sop_ia, rechazo_res_ia, tend_htf_ia, confianza, tipo_setup):
    """
    Valida con datos reales del DataFrame si la señal de la IA es confiable.
    Retorna (True/False, mensaje_rechazo)
    """
    if len(df_ltf) < 3 or len(df_htf) < 3:
        return False, "Datos insuficientes para validar"
    
    # Última vela cerrada (índice -2 porque la última puede estar abierta)
    ultima_cerrada = df_ltf.iloc[-2]
    precio_cierre = ultima_cerrada['close']
    ema20 = ultima_cerrada['ema20']
    soporte_ltf, resistencia_ltf, _, _, _, _ = detectar_zonas_mercado(df_ltf)
    
    # Tendencia HTF real
    y_htf = df_htf['close'].values[-30:]
    slope_htf, _, _, _, _ = linregress(np.arange(len(y_htf)), y_htf)
    tendencia_htf_real = 'alcista' if slope_htf > 0.01 else 'bajista' if slope_htf < -0.01 else 'lateral'
    tendencia_fuerte = abs(slope_htf) > 0.05  # pendiente fuerte > 0.05% por vela
    
    # Verificar condiciones numéricas
    cierre_sobre_valido = (precio_cierre > ema20 * (1 + MIN_CIERRE_EMA_PCT))
    cierre_bajo_valido = (precio_cierre < ema20 * (1 - MIN_CIERRE_EMA_PCT))
    
    # Cercanía a niveles (umbral 0.5% para más flexibilidad)
    sop_cercano = abs(precio_cierre - soporte_ltf) / soporte_ltf < 0.005
    res_cercano = abs(resistencia_ltf - precio_cierre) / resistencia_ltf < 0.005
    
    # Detectar patrones reales en el DataFrame
    vela_alcista = ultima_cerrada['close'] > ultima_cerrada['open']
    vela_bajista = ultima_cerrada['close'] < ultima_cerrada['open']
    
    # Rechazo de soporte real
    rechazo_sop_valido = (sop_cercano and vela_alcista and (precio_cierre > ema20))
    # Rechazo de resistencia real
    rechazo_res_valido = (res_cercano and vela_bajista and (precio_cierre < ema20))
    
    # Rompimientos: precio cruza nivel con patrón de confirmación
    idx = -2
    rompimiento_alcista = (precio_cierre > resistencia_ltf and (es_martillo(df_ltf, idx) or es_tres_soldados_blancos(df_ltf, idx) or es_engulfing_alcista(df_ltf, idx)))
    rompimiento_bajista = (precio_cierre < soporte_ltf and (es_estrella_fugaz(df_ltf, idx) or es_tres_cuervos_negros(df_ltf, idx) or es_engulfing_bajista(df_ltf, idx)))
    
    # Retroceso a EMA con vela de rechazo (pin bar) y tendencia fuerte
    cerca_ema = abs(precio_cierre - ema20) / ema20 < 0.003
    vela_rechazo_ema = (vela_alcista and ultima_cerrada['low'] <= ema20 * 0.998) or (vela_bajista and ultima_cerrada['high'] >= ema20 * 1.002)
    
    # Evaluación según decisión y tipo de setup
    if decision == "Buy":
        cond_nivel = rechazo_sop_valido
        cond_rompimiento = rompimiento_alcista and precio_cierre > resistencia_ltf * 1.001
        cond_retroceso = (tendencia_fuerte and cerca_ema and vela_rechazo_ema)
        
        valido = cond_nivel or cond_rompimiento or cond_retroceso
        
        if not valido:
            return False, f"Buy rechazada: no hay rechazo soporte ({rechazo_sop_valido}), rompimiento ({cond_rompimiento}) ni retroceso EMA valido ({cond_retroceso})"
        if confianza < 55:
            return False, f"Confianza IA baja ({confianza})"
        return True, "Validación exitosa"
    
    elif decision == "Sell":
        cond_nivel = rechazo_res_valido
        cond_rompimiento = rompimiento_bajista and precio_cierre < soporte_ltf * 0.999
        cond_retroceso = (tendencia_fuerte and cerca_ema and vela_rechazo_ema)
        
        valido = cond_nivel or cond_rompimiento or cond_retroceso
        
        if not valido:
            return False, f"Sell rechazada: no hay rechazo resistencia ({rechazo_res_valido}), rompimiento ({cond_rompimiento}) ni retroceso EMA valido ({cond_retroceso})"
        if confianza < 55:
            return False, f"Confianza IA baja ({confianza})"
        return True, "Validación exitosa"
    
    return False, "Decisión no reconocida"

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

    if not sl_ia or sl_ia <= 0:
        logger.error("La IA no proporcionó stop loss. Cancelando.")
        return
    if not tp1_ia or tp1_ia <= 0:
        logger.warning("IA no dio TP1, cancelando.")
        return

    entrada = precio_actual
    # Calcular ATR para ajustar SL mínimo
    atr = df_ltf['atr'].iloc[-1] if 'atr' in df_ltf.columns else entrada * 0.005
    min_sl_dist = atr * ATR_MULTIPLIER_SL

    if decision == "Buy":
        distancia_original = entrada - sl_ia
        if distancia_original <= 0:
            logger.error("SL inválido (por encima del precio en Buy). Cancelando.")
            return
        min_dist = max(entrada * MIN_SL_DIST_PCT, min_sl_dist)
        max_dist = entrada * MAX_SL_DIST_PCT
        if distancia_original < min_dist:
            sl_ajustado = entrada - min_dist
        elif distancia_original > max_dist:
            sl_ajustado = entrada - max_dist
        else:
            sl_ajustado = sl_ia
        distancia_final = entrada - sl_ajustado
    else:
        distancia_original = sl_ia - entrada
        if distancia_original <= 0:
            logger.error("SL inválido (por debajo del precio en Sell). Cancelando.")
            return
        min_dist = max(entrada * MIN_SL_DIST_PCT, min_sl_dist)
        max_dist = entrada * MAX_SL_DIST_PCT
        if distancia_original < min_dist:
            sl_ajustado = entrada + min_dist
        elif distancia_original > max_dist:
            sl_ajustado = entrada + max_dist
        else:
            sl_ajustado = sl_ia
        distancia_final = sl_ajustado - entrada

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

    # Ejecutar orden market
    if is_paper:
        order_id = f"paper_{paper_trade_counter+1}"
        logger.info(f"📄 PAPER: Orden MARKET {decision} {qty_btc} BTC a {entrada:.2f}")
        comision = nocional * PAPER_COMMISSION_PCT
        paper_balance -= comision
        logger.info(f"   Comisión estimada: {comision:.2f} USDT")
        paper_trade_counter += 1
        trade_id = paper_trade_counter
        # Activar trailing stop desde el inicio
        positions[trade_id] = {
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_inicial": sl_ajustado, "sl_actual": sl_ajustado,
            "tp1": tp1_ia, "trailing_logic": "TRAILING",
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto,
            "trailing_activado": True,   # Trailing activo desde el inicio
            "best_price": entrada, "trailing_stop": entrada * (1 - TRAILING_PERCENT) if decision == "Buy" else entrada * (1 + TRAILING_PERCENT)
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
            "sl_inicial": sl_ajustado, "sl_actual": sl_ajustado,
            "tp1": tp1_ia, "trailing_logic": "TRAILING",
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto,
            "trailing_activado": True,
            "best_price": entrada, "trailing_stop": entrada * (1 - TRAILING_PERCENT) if decision == "Buy" else entrada * (1 + TRAILING_PERCENT)
        }
        modo = "💰 REAL"

    msg = (f"{modo} [#{trade_id}] {decision} MARKET en {entrada:.2f} | Qty {qty_btc} BTC (riesgo {risk_per_trade} USDT)\n"
           f"🛑 SL: {sl_ajustado:.2f} (dist {distancia_final:.1f} USD)\n"
           f"🎯 TP1: {tp1_ia:.2f} | Trailing: {TRAILING_PERCENT*100:.1f}%\n"
           f"📝 Razón: {razon}\n"
           f"💰 Margen requerido: {margen_necesario:.2f} USDT | Libre disponible: {free_margin:.2f} USDT")
    logger.info(msg)
    telegram_mensaje(msg)

    # Gráfico con niveles (excluyendo vela actual para mostrar solo cerradas)
    titulo_grafico = f"Entrada - {modo} #{trade_id}"
    img_completa = generar_grafico_para_vision(df_ltf, titulo_grafico, sop, res, slope, inter,
                                               entry_price=entrada, sl_price=sl_ajustado,
                                               tp1_price=tp1_ia, side=decision, excluir_actual=False)
    if img_completa:
        img_completa.save("/tmp/in_completo.png")
        caption = (f"{modo} #{trade_id} {decision}\n"
                   f"Entry: {entrada:.2f} | SL: {sl_ajustado:.2f} | TP1: {tp1_ia:.2f} | Trailing: {TRAILING_PERCENT*100:.1f}%")
        telegram_enviar_imagen("/tmp/in_completo.png", caption)

    guardar_memoria()

# =================== GESTIÓN DE TRADES ACTIVOS CON TRAILING STOP ===================
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

def revisar_sl_tp_simulado(df):
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions, ULTIMO_APRENDIZAJE
    if not paper_positions:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    cerrar_ids = []
    for tid, t in list(paper_positions.items()):
        # TP1: cerrar 50% si no se ha hecho y el precio alcanza TP1
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
                    # No mover SL a breakeven, mantener trailing existente
                    logger.info(f"✅ PAPER: TP1 #{tid} +{pnl_parcial:.2f} USDT")
                    telegram_mensaje(f"✅ PAPER TP1 #{tid}: +{pnl_parcial:.2f} USDT.")
                    if t['qty_restante'] <= 0.0001:
                        cerrar_ids.append(tid)
                else:
                    cerrar_ids.append(tid)

        # Trailing stop (siempre activo)
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

        # Stop loss inicial (solo si no se ha alcanzado TP1 ni trailing)
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

    if paper_total_trades - ULTIMO_APRENDIZAJE >= 10:
        aprender_de_trades()

def real_revisar_sl_tp(df):
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
                        logger.info(f"✅ TP1 #{tid} +{pnl_parcial:.2f} USDT")
                        telegram_mensaje(f"✅ TP1 #{tid}: +{pnl_parcial:.2f} USDT.")
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
            else:  # Sell
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

        # Stop loss inicial antes de TP1 (solo si no se ha alcanzado TP1)
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

        # Prompt más estructurado y en español para evitar basura
        ult_serial = convertir_serializable(ult)
        prompt = f"""Eres un trader experto. Analiza estos últimos 10 trades (formato JSON) y extrae UNA sola lección corta (máximo 200 caracteres) en español, clara y sin caracteres extraños. Enfócate en qué condiciones evitar o buscar para mejorar el winrate.

Trades: {json.dumps(ult_serial, ensure_ascii=False)}

Devuelve SOLO la lección en texto plano, sin comillas, sin saltos de línea adicionales, sin emojis."""
        resp = client.chat.completions.create(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role":"user","content":prompt}],
            timeout=10,
            temperature=0.3
        )
        leccion = resp.choices[0].message.content.strip()
        # Limpiar posibles residuos (comillas, saltos, etc.)
        leccion = re.sub(r'[\n\r"]+', ' ', leccion).strip()
        if len(leccion) > 200:
            leccion = leccion[:197] + "..."
        REGLAS_APRENDIDAS = leccion
        telegram_mensaje(f"🧠 Lección IA: {REGLAS_APRENDIDAS}")
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
    global senal_pendiente

    cargar_memoria()
    set_leverage()
    sync_active_trades_with_bybit()

    telegram_mensaje("🤖 Bot iniciado - Con filtros numéricos, confirmación inteligente y trailing stop")
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

            # ========== CONFIRMACIÓN DE SEÑAL PENDIENTE (solo si la IA lo pidió) ==========
            if senal_pendiente is not None:
                # Esperamos una vela completa (5 minutos) para confirmar
                if ultima_vela is not None and ultima_vela != senal_pendiente['vela_senal']:
                    # Evaluar confirmación
                    df_confirm = df_ltf.iloc[-2]  # Última vela cerrada (la que sigue a la señal)
                    senal = senal_pendiente
                    if senal['decision'] == 'Buy':
                        # Confirmación: la nueva vela cerró por encima del máximo de la vela señal (o por encima de EMA20 + margen)
                        vela_senal = df_ltf.loc[senal['vela_senal']]
                        confirmado = (df_confirm['close'] > vela_senal['high']) or (df_confirm['close'] > df_confirm['ema20'] * (1 + MIN_CIERRE_EMA_PCT))
                    else:
                        vela_senal = df_ltf.loc[senal['vela_senal']]
                        confirmado = (df_confirm['close'] < vela_senal['low']) or (df_confirm['close'] < df_confirm['ema20'] * (1 - MIN_CIERRE_EMA_PCT))
                    
                    if confirmado:
                        logger.info(f"✅ Señal {senal['decision']} confirmada por vela siguiente. Abriendo operación.")
                        abrir_posicion_con_ia(senal['decision'], senal['precio_actual'], senal['razon'], senal['explicacion'],
                                              senal['sl_ia'], senal['tp1_ia'], df_ltf, senal['sop'], senal['res'], senal['slope'], senal['inter'])
                    else:
                        logger.info(f"❌ Señal {senal['decision']} no confirmada por vela siguiente. Descartada.")
                        telegram_mensaje(f"❌ Señal {senal['decision']} descartada por falta de confirmación en vela siguiente.")
                    senal_pendiente = None

            # ========== NUEVA SEÑAL DE IA (solo si no hay pendiente) ==========
            if es_vela_nueva and active_count < max_trades_actual and risk_management_check() and senal_pendiente is None:
                # Generar gráficos para IA excluyendo la vela actual
                sop_ltf, res_ltf, slope_ltf, inter_ltf, _, _ = detectar_zonas_mercado(df_ltf)
                sop_htf, res_htf, slope_htf, inter_htf, _, _ = detectar_zonas_mercado(df_htf)

                # Para la IA, usamos gráficos sin la vela actual (solo velas cerradas)
                img_ltf = generar_grafico_para_vision(df_ltf, "BTC/USDT 5m (LTF)", sop_ltf, res_ltf, slope_ltf, inter_ltf, excluir_actual=True)
                img_htf = generar_grafico_para_vision(df_htf, "BTC/USDT 1h (HTF)", sop_htf, res_htf, slope_htf, inter_htf, excluir_actual=True)

                if img_ltf and img_htf:
                    dec, raz, explicacion, entry_ia, sl_ia, tp1_ia, conf, cierre_sobre, cierre_bajo, rech_sop, rech_res, tend_htf, confirm_needed, tipo_setup = analizar_con_qwen(img_ltf, img_htf)
                    logger.info(f"🧠 Decisión IA: {dec} - Razón: {raz} - Confianza: {conf} - Confirmación necesaria: {confirm_needed} - Tipo setup: {tipo_setup}")
                    if dec != "Hold":
                        # Validación numérica post-IA
                        valido, mensaje = validar_setup_ia(dec, raz, df_ltf, df_htf, cierre_sobre, cierre_bajo, rech_sop, rech_res, tend_htf, conf, tipo_setup)
                        if valido:
                            if confirm_needed:
                                # Guardar señal pendiente para confirmar en la siguiente vela
                                senal_pendiente = {
                                    'decision': dec, 'precio_actual': precio_actual, 'razon': raz, 'explicacion': explicacion,
                                    'sl_ia': sl_ia, 'tp1_ia': tp1_ia, 'sop': sop_ltf, 'res': res_ltf, 'slope': slope_ltf,
                                    'inter': inter_ltf, 'vela_senal': vela_actual_time
                                }
                                logger.info(f"⏳ Señal {dec} requiere confirmación. Pendiente de siguiente vela.")
                                telegram_mensaje(f"⏳ Señal {dec} detectada (confianza {conf}). Esperando confirmación en próxima vela.")
                            else:
                                # Abrir inmediatamente
                                abrir_posicion_con_ia(dec, precio_actual, raz, explicacion, sl_ia, tp1_ia, df_ltf, sop_ltf, res_ltf, slope_ltf, inter_ltf)
                        else:
                            logger.warning(f"🚫 Señal {dec} rechazada por filtros: {mensaje}")
                            telegram_mensaje(f"🚫 Señal {dec} rechazada: {mensaje}")
                    else:
                        logger.info(f"IA decidió HOLD. Razón: {raz[:100]}")
                else:
                    logger.error("No se pudieron generar los gráficos")

                ultima_vela = vela_actual_time

            # Gestión de trades activos (siempre)
            if PAPER_TRADE and paper_positions:
                revisar_sl_tp_simulado(df_ltf)
            elif not PAPER_TRADE and REAL_ACTIVE_TRADES:
                real_revisar_sl_tp(df_ltf)

            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            logger.error(f"ERROR CRÍTICO: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(30)

if __name__ == '__main__':
    run_bot()
