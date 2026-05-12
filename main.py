# BOT TRADING CON ANÁLISIS ESTRUCTURADO Y GRÁFICOS - VERSIÓN SIMPLIFICADA (SOLO MARKET)
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

# Configurar logging para Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =================== SANITIZAR EMOJIS PARA MATPLOTLIB ===================
def sanitize_for_matplotlib(text):
    """Elimina emojis y caracteres no soportados por DejaVu Sans."""
    if not isinstance(text, str):
        return text
    # Patrón amplio de emojis Unicode
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map
        u"\U0001F700-\U0001F77F"  # alchemical
        u"\U0001F780-\U0001F7FF"  # Geometric Shapes
        u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA00-\U0001FA6F"  # Extended-A
        u"\U0001FA70-\U0001FAFF"  # Extended-B
        u"\U00002702-\U000027B0"  # Dingbats
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
        logger.info("Paper trade: apalancamiento simulado 34x")
        return
    try:
        body = {"category": "linear", "symbol": "BTCUSDT", "buyLeverage": "34", "sellLeverage": "34"}
        result = bybit_request("/v5/position/set-leverage", method="POST", body=body)
        ret_code = result.get('retCode')
        if ret_code == 0 or ret_code == 110043:
            logger.info("Apalancamiento 34x configurado")
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
            margin_used += (t['qty_original'] * t['entrada']) / 34  # LEVERAGE fijo 34
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
        logger.info(f"PAPER: Orden {side} {qty} BTC simulada")
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
        logger.info(f"PAPER: Cierre simulado de {qty} BTC lado {side_to_close}")
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
                "tp2_ejecutado": t.get("tp2_ejecutado", False),
                "sl_actual": t.get("sl_actual"), "trailing_logic": t.get("trailing_logic", "BREAKEVEN"),
                "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
                "breakeven_activado": t.get("breakeven_activado", False)
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
                "tp2_ejecutado": t.get("tp2_ejecutado", False),
                "sl_actual": t.get("sl_actual"), "trailing_logic": t.get("trailing_logic", "BREAKEVEN"),
                "qty_original": t.get("qty_original"), "qty_restante": t.get("qty_restante"),
                "breakeven_activado": t.get("breakeven_activado", False)
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
        logger.info("Memoria guardada")
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
        logger.info(f"Memoria cargada. Trades: {paper_total_trades if PAPER_TRADE else TOTAL_TRADES}")
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

# =================== CONFIGURACIÓN DEL BOT ===================
SYMBOL = "BTCUSDT"
INTERVAL_LTF = "5"
INTERVAL_HTF = "60"
RISK_PER_TRADE_MAX = 3.0      # Máximo riesgo por operación (USD)
LEVERAGE = 34
SLEEP_SECONDS = 60
GRAFICO_VELAS_LIMIT = 120
MAX_CONCURRENT_TRADES = 3
MIN_MARGIN_PER_TRADE = 3.0
TP1_PERCENT = 0.5

MIN_SL_DIST_PCT = 0.0005   # 0.05% mínimo
MAX_SL_DIST_PCT = 0.01     # 1% máximo

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
            logger.info("Imagen enviada a Telegram")
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
    modo = "PAPER" if PAPER_TRADE else "REAL"
    mensaje = (
        f"{modo} ESTADO BTC\n"
        f"Balance: {balance:.2f} USDT\n"
        f"PnL día: {pnl_global:+.2f} USDT\n"
        f"Winrate: {winrate:.1f}%\n"
        f"Activos: {active_trades}/{max_din}\n"
        f"PF (10t): {ULTIMO_PROFIT_FACTOR:.2f}"
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
                                entry_price=None, sl_price=None, tp1_price=None, tp2_price=None, side=None):
    if df.empty:
        return None
    df_plot = df.tail(GRAFICO_VELAS_LIMIT).copy()
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
    if tp2_price is not None:
        ax.axhline(tp2_price, color='deepskyblue', linestyle='--', linewidth=2, label=f'TP2 {tp2_price:.0f}')
    # Sanitizar título para eliminar emojis
    titulo_limpio = sanitize_for_matplotlib(titulo)
    ax.set_title(titulo_limpio, color='white', fontsize=14)
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), framealpha=0.5, facecolor='black', edgecolor='white')
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

# =================== PROMPT SIMPLIFICADO ESTILO GEMINI ===================
def analizar_con_qwen(img_ltf, img_htf):
    global TOKENS_ACUMULADOS
    try:
        img_ltf_b64 = pil_to_base64(img_ltf)
        img_htf_b64 = pil_to_base64(img_htf)

        prompt = """
Eres un trader. Mira los dos gráficos de BTC/USDT: 5 minutos (primera imagen) y 1 hora (segunda imagen).
Analiza las últimas 5-10 velas, no solo la última. Responde si comprarías (Buy), venderías (Sell) o no harías nada (Hold).

Razona brevemente: ¿Qué patrón ves? (martillo, envolvente, doji, etc.) ¿Tendencia? ¿Soportes/resistencias?
Si ves una oportunidad clara, da precios concretos de entrada (precio actual o ligeramente diferente), stop loss y take profit 1 (y opcional TP2).
Si no ves oportunidad, di Hold.

Devuelve ÚNICAMENTE un JSON válido en una línea con esta estructura:
{
  "decision": "Buy/Sell/Hold",
  "razon": "frase corta (max 100)",
  "explicacion": "análisis detallado en español, mencionando velas, tendencia, niveles",
  "entry_price": 0.0,
  "sl_price": 0.0,
  "tp1_price": 0.0,
  "tp2_price": 0.0,
  "confianza": 0-100
}

Importante:
- Si decision es Hold, entry_price, sl_price, tp1_price pueden ser 0.
- entry_price debe ser el precio al que entrarías. Como el bot ejecuta inmediatamente al precio actual, pon el precio actual o muy cercano.
- SL y TP deben tener relación riesgo/recompensa >= 2.
- No inventes niveles si no hay setup claro.
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
        # Buscar JSON al final
        json_match = re.search(r'\{.*\}(?=\s*$)', contenido, re.DOTALL)
        if json_match:
            datos = parse_json_seguro(json_match.group())
        else:
            datos = parse_json_seguro(contenido)
        if not datos:
            return "Hold", "Error parsing", "", None, None, None, None, "BREAKEVEN", {}

        decision = datos.get("decision", "Hold")
        razon = datos.get("razon", "")
        explicacion = datos.get("explicacion", "")
        entry_price = datos.get("entry_price")
        sl_price = datos.get("sl_price")
        tp1_price = datos.get("tp1_price")
        tp2_price = datos.get("tp2_price")
        trailing = "BREAKEVEN"
        analisis = {
            "tendencia_htf": "desconocida",
            "tendencia_ltf": "desconocida",
            "rr_estimada": 0,
            "calidad_entrada": "media",
            "decision_texto": "Sí" if decision != "Hold" else "No"
        }
        return decision, razon, explicacion, entry_price, sl_price, tp1_price, tp2_price, trailing, analisis
    except Exception as e:
        logger.error(f"Error en IA: {e}")
        return "Hold", "Error API", "", None, None, None, None, "BREAKEVEN", {}

# =================== GESTIÓN DE RIESGO Y APERTURA (SOLO MARKET) ===================
def calcular_riesgo_dinamico(free_margin):
    if free_margin >= 20:
        return RISK_PER_TRADE_MAX
    elif free_margin >= 10:
        return 1.5
    else:
        return 1.0

def abrir_posicion_con_ia(decision, precio_actual, razon, contexto, sl_ia, tp1_ia, tp2_ia, analisis, df_ltf, sop, res, slope, inter):
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
    if not tp2_ia or tp2_ia <= 0:
        tp2_ia = tp1_ia

    entrada = precio_actual  # Market order

    if decision == "Buy":
        distancia = entrada - sl_ia
        if distancia <= 0:
            logger.error("SL inválido (por encima del precio en Buy). Cancelando.")
            return
        min_dist = entrada * MIN_SL_DIST_PCT
        max_dist = entrada * MAX_SL_DIST_PCT
        if distancia < min_dist:
            sl_ajustado = entrada - min_dist
        elif distancia > max_dist:
            sl_ajustado = entrada - max_dist
        else:
            sl_ajustado = sl_ia
        distancia_final = entrada - sl_ajustado
    else:
        distancia = sl_ia - entrada
        if distancia <= 0:
            logger.error("SL inválido (por debajo del precio en Sell). Cancelando.")
            return
        min_dist = entrada * MIN_SL_DIST_PCT
        max_dist = entrada * MAX_SL_DIST_PCT
        if distancia < min_dist:
            sl_ajustado = entrada + min_dist
        elif distancia > max_dist:
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
        logger.info(f"PAPER: Orden MARKET {decision} {qty_btc} BTC a {entrada:.2f}")
        paper_trade_counter += 1
        trade_id = paper_trade_counter
        positions[trade_id] = {
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_inicial": sl_ajustado, "sl_actual": sl_ajustado,
            "tp1": tp1_ia, "tp2": tp2_ia, "trailing_logic": "BREAKEVEN",
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "tp2_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto
        }
        modo = "PAPER"
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
            "tp1": tp1_ia, "tp2": tp2_ia, "trailing_logic": "BREAKEVEN",
            "qty_original": qty_btc, "qty_restante": round(qty_btc - qty_btc * TP1_PERCENT, 3),
            "tp1_ejecutado": False, "tp2_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto
        }
        modo = "REAL"

    msg = (f"{modo} [#{trade_id}] {decision} MARKET en {entrada:.2f} | Qty {qty_btc} BTC (riesgo {risk_per_trade} USDT)\n"
           f"SL: {sl_ajustado:.2f} (dist {distancia_final:.1f} USD)\n"
           f"TP1: {tp1_ia:.2f} | TP2: {tp2_ia:.2f}\n"
           f"Razon: {razon}\n"
           f"Margen usado: {margen_necesario:.2f} / {free_margin:.2f} USDT")
    logger.info(msg)
    telegram_mensaje(msg)

    # Gráfico con niveles
    titulo_grafico = f"Entrada - {modo} #{trade_id}"
    img_completa = generar_grafico_para_vision(df_ltf, titulo_grafico, sop, res, slope, inter,
                                               entry_price=entrada, sl_price=sl_ajustado,
                                               tp1_price=tp1_ia, tp2_price=tp2_ia, side=decision)
    if img_completa:
        img_completa.save("/tmp/in_completo.png")
        caption = (f"{modo} #{trade_id} {decision}\n"
                   f"Entry: {entrada:.2f} | SL: {sl_ajustado:.2f} | TP1: {tp1_ia:.2f} | TP2: {tp2_ia:.2f}")
        telegram_enviar_imagen("/tmp/in_completo.png", caption)

    guardar_memoria()

# =================== GESTIÓN DE TRADES ACTIVOS ===================
def sync_active_trades_with_bybit():
    if PAPER_TRADE:
        return
    global REAL_ACTIVE_TRADES
    real_size = get_real_position_size()
    if real_size == 0.0 and REAL_ACTIVE_TRADES:
        logger.info("Sincronización: No hay posición real. Limpiando trades fantasmas.")
        REAL_ACTIVE_TRADES.clear()
        guardar_memoria()
    elif real_size > 0.0 and not REAL_ACTIVE_TRADES:
        logger.warning("Hay posición real pero el bot no la registra.")
    else:
        mem_size = sum(t['qty_restante'] for t in REAL_ACTIVE_TRADES.values())
        if abs(mem_size - real_size) > 0.002:
            logger.warning(f"Discrepancia de tamaño: memoria {mem_size:.3f} BTC, real {real_size:.3f} BTC.")
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
        if not t['tp1_ejecutado'] and t['tp1'] is not None and t['tp1'] > 0:
            if (t['decision']=="Buy" and h >= t['tp1']) or (t['decision']=="Sell" and l <= t['tp1']):
                qty_tp1 = round(t['qty_original'] * TP1_PERCENT, 3)
                if qty_tp1 >= 0.001 and t['qty_restante'] > 0:
                    pnl_parcial = (t['tp1'] - t['entrada']) * qty_tp1 if t['decision']=="Buy" else (t['entrada'] - t['tp1']) * qty_tp1
                    t['pnl_parcial'] += pnl_parcial
                    t['qty_restante'] = round(t['qty_original'] - qty_tp1, 3)
                    t['tp1_ejecutado'] = True
                    t['breakeven_activado'] = True
                    offset = 2.0
                    t['sl_actual'] = t['entrada'] - offset if t['decision']=="Buy" else t['entrada'] + offset
                    logger.info(f"PAPER: TP1 #{tid} +{pnl_parcial:.2f} USDT")
                    telegram_mensaje(f"PAPER TP1 #{tid}: +{pnl_parcial:.2f} USDT")
                    if t['qty_restante'] <= 0.0001:
                        cerrar_ids.append(tid)
                else:
                    cerrar_ids.append(tid)
        if t['tp1_ejecutado'] and not t['tp2_ejecutado'] and t['tp2'] is not None and t['tp2'] > 0 and t['qty_restante'] > 0:
            if (t['decision']=="Buy" and h >= t['tp2']) or (t['decision']=="Sell" and l <= t['tp2']):
                qty_restante = t['qty_restante']
                if qty_restante >= 0.001:
                    pnl_resto = (t['tp2'] - t['entrada']) * qty_restante if t['decision']=="Buy" else (t['entrada'] - t['tp2']) * qty_restante
                    pnl_total = t['pnl_parcial'] + pnl_resto
                    paper_balance += pnl_total
                    paper_total_trades += 1
                    if pnl_total > 0:
                        paper_win_count += 1
                    else:
                        paper_loss_count += 1
                    paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                    cerrar_ids.append(tid)
                    msg = f"PAPER CIERRE #{tid} TP2 - PnL: {pnl_total:+.2f} USDT"
                    logger.info(msg)
                    telegram_mensaje(msg)
                    reporte_estado()
                else:
                    cerrar_ids.append(tid)
        if t['qty_restante'] > 0.001:
            cond = (t['decision']=="Buy" and l <= t['sl_actual']) or (t['decision']=="Sell" and h >= t['sl_actual'])
            if cond:
                qty_restante = t['qty_restante']
                pnl_resto = (t['sl_actual'] - t['entrada'])*qty_restante if t['decision']=="Buy" else (t['entrada'] - t['sl_actual'])*qty_restante
                pnl_total = t['pnl_parcial'] + pnl_resto
                paper_balance += pnl_total
                paper_total_trades += 1
                if pnl_total>0:
                    paper_win_count+=1
                else:
                    paper_loss_count+=1
                paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                cerrar_ids.append(tid)
                motivo = "Stop Loss inicial" if not t.get('breakeven_activado') else "Breakeven"
                msg = f"PAPER CIERRE #{tid} por {motivo} - PnL: {pnl_total:+.2f} USDT"
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
                        t['breakeven_activado']=True
                        offset=2.0
                        t['sl_actual']=t['entrada']-offset if t['decision']=="Buy" else t['entrada']+offset
                        logger.info(f"TP1 #{tid} +{pnl_parcial:.2f} USDT")
                        telegram_mensaje(f"TP1 #{tid}: +{pnl_parcial:.2f} USDT")
                        if t['qty_restante']<=0.0001:
                            cerrar_ids.append(tid)
                    else:
                        logger.warning(f"TP1 no confirmado #{tid}")
                else:
                    cerrar_ids.append(tid)
        if t['tp1_ejecutado'] and not t['tp2_ejecutado'] and t['tp2']>0 and t['qty_restante']>0:
            if (t['decision']=="Buy" and h>=t['tp2']) or (t['decision']=="Sell" and l<=t['tp2']):
                qty_restante = t['qty_restante']
                if qty_restante>=0.001:
                    result = close_position_qty_confirm(qty_restante, t['decision'])
                    if result and result!="already_closed":
                        pnl_resto = (t['tp2']-t['entrada'])*qty_restante if t['decision']=="Buy" else (t['entrada']-t['tp2'])*qty_restante
                        pnl_total = t['pnl_parcial']+pnl_resto
                        REAL_BALANCE = get_real_balance()
                        TOTAL_TRADES+=1
                        if pnl_total>0:
                            WIN_COUNT+=1
                        else:
                            LOSS_COUNT+=1
                        TRADE_HISTORY.append(convertir_serializable({"pnl":pnl_total, "resultado_win":pnl_total>0, "decision":t['decision'], "razon":t['razon']}))
                        cerrar_ids.append(tid)
                        msg = f"CIERRE COMPLETO #{tid} TP2 - PnL: {pnl_total:+.2f} USDT"
                        logger.info(msg)
                        telegram_mensaje(msg)
                        reporte_estado()
                    else:
                        logger.error(f"Falló cierre TP2 #{tid}")
                else:
                    cerrar_ids.append(tid)
        if t['qty_restante']>0.001:
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
                    motivo = "Stop Loss inicial" if not t.get('breakeven_activado') else "Breakeven"
                    msg = f"CIERRE #{tid} por {motivo} - PnL: {pnl_total:+.2f} USDT"
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
            resumen = f"APRENDIZAJE PAPER #{paper_total_trades}\nWinrate: {winrate:.1f}% | PF: {ULTIMO_PROFIT_FACTOR:.2f}"
        else:
            ult = TRADE_HISTORY[-10:] if len(TRADE_HISTORY)>=10 else TRADE_HISTORY
            gan = sum(t['pnl'] for t in ult if t['pnl']>0)
            per = abs(sum(t['pnl'] for t in ult if t['pnl']<0))
            ULTIMO_PROFIT_FACTOR = gan/per if per>0 else 1.0
            winrate = (WIN_COUNT/TOTAL_TRADES*100) if TOTAL_TRADES>0 else 0
            resumen = f"APRENDIZAJE #{TOTAL_TRADES}\nWinrate: {winrate:.1f}% | PF: {ULTIMO_PROFIT_FACTOR:.2f}"
        telegram_mensaje(resumen)
        try:
            ult_serial = convertir_serializable(ult)
            prompt = f"Analiza estos 10 trades y da una lección corta (max 200 chars): {json.dumps(ult_serial)}"
            resp = client.chat.completions.create(model=MODELO_VISION, messages=[{"role":"user","content":prompt}], timeout=10)
            REGLAS_APRENDIDAS = resp.choices[0].message.content
            telegram_mensaje(f"Lección IA: {REGLAS_APRENDIDAS}")
        except:
            pass
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
        logger.info(f"Nuevo día: {hoy}. Balance inicial: {balance:.2f}")
    balance_actual = paper_balance if PAPER_TRADE else REAL_BALANCE
    if balance_actual is not None and DAILY_START_BALANCE is not None:
        drawdown = (balance_actual - DAILY_START_BALANCE) / DAILY_START_BALANCE
        if drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
            STOPPED_TODAY = True
            logger.warning("Drawdown diario superado. Operaciones detenidas.")
    return not STOPPED_TODAY

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global REAL_BALANCE, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, TRADE_HISTORY, REAL_ACTIVE_TRADES
    global paper_balance, paper_trade_counter, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    cargar_memoria()
    set_leverage()

    telegram_mensaje("Bot iniciado - Modo simplificado (solo market, sin órdenes pendientes)")
    logger.info("Bot iniciado")

    if PAPER_TRADE:
        logger.info(f"PAPER TRADE - Saldo: {paper_balance:.2f} USDT")
        telegram_mensaje(f"Bot Paper Trade Online - Saldo simulado: {paper_balance:.2f} USDT")
    else:
        REAL_BALANCE = get_real_balance()
        if REAL_BALANCE is None:
            logger.error("No se pudo obtener saldo real. Abortando.")
            return
        logger.info(f"BOT REAL - Balance: {REAL_BALANCE:.2f} USDT")
        telegram_mensaje(f"Bot Real Online - Balance: {REAL_BALANCE:.2f} USDT")

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

            sop_ltf, res_ltf, slope_ltf, inter_ltf, _, _ = detectar_zonas_mercado(df_ltf)
            sop_htf, res_htf, slope_htf, inter_htf, _, _ = detectar_zonas_mercado(df_htf)

            img_ltf = generar_grafico_para_vision(df_ltf, "BTC/USDT 5m (LTF)", sop_ltf, res_ltf, slope_ltf, inter_ltf)
            img_htf = generar_grafico_para_vision(df_htf, "BTC/USDT 1h (HTF)", sop_htf, res_htf, slope_htf, inter_htf)

            if img_ltf and img_htf:
                dec, raz, explicacion, entry_ia, sl_ia, tp1_ia, tp2_ia, trailing, analisis = analizar_con_qwen(img_ltf, img_htf)
                logger.info(f"Decisión IA: {dec} - Razón: {raz}")
                if dec != "Hold":
                    logger.info(f"Explicación: {explicacion[:200]}")
                active_count = len(paper_positions) if PAPER_TRADE else len(REAL_ACTIVE_TRADES)
                vela_c = df_ltf.index[-2]
                if ultima_vela is None:
                    ultima_vela = vela_c

                if dec in ["Buy","Sell"] and active_count < max_trades_actual and risk_management_check() and ultima_vela != vela_c:
                    abrir_posicion_con_ia(dec, precio_actual, raz, explicacion, sl_ia, tp1_ia, tp2_ia, analisis,
                                          df_ltf, sop_ltf, res_ltf, slope_ltf, inter_ltf)
                    ultima_vela = vela_c
                else:
                    if dec in ["Buy","Sell"]:
                        if active_count >= max_trades_actual:
                            logger.info(f"No se abre {dec}: límite de trades alcanzado ({active_count}/{max_trades_actual})")
                        elif not risk_management_check():
                            logger.info("No se abre: drawdown diario superado")
                        elif ultima_vela == vela_c:
                            logger.info("No se abre: ya se analizó esta vela")
                    else:
                        logger.info(f"IA decidió HOLD. Motivo: {raz[:100]}")
            else:
                logger.error("No se pudieron generar los gráficos")

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
