# BOT TRADING CON ANÁLISIS ESTRUCTURADO PASO A PASO (Bybit + Qwen3-VL-32B-Instruct)
# Versión 3.0 - Análisis crítico, sin sesgos, con contexto_pensamiento, condiciones_espera, zonas_liquidez
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
        print("📄 Paper trade: apalancamiento simulado 34x")
        return
    try:
        body = {"category": "linear", "symbol": "BTCUSDT", "buyLeverage": "34", "sellLeverage": "34"}
        result = bybit_request("/v5/position/set-leverage", method="POST", body=body)
        ret_code = result.get('retCode')
        if ret_code == 0 or ret_code == 110043:
            print("✅ Apalancamiento 34x configurado")
        else:
            print(f"⚠️ Error configurando apalancamiento: {result}")
    except Exception as e:
        print(f"❌ Excepción configurando apalancamiento: {e}")

def get_real_balance():
    if PAPER_TRADE:
        return paper_balance
    try:
        params = {"accountType": "UNIFIED", "coin": "USDT"}
        result = bybit_request("/v5/account/wallet-balance", method="GET", params=params)
        return float(result['result']['list'][0]['coin'][0]['walletBalance'])
    except Exception as e:
        print(f"❌ Error obteniendo saldo: {e}")
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
        print(f"❌ Error obteniendo margen libre: {e}")
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
        print(f"❌ Error get_real_position_size: {e}")
        return 0.0

def place_market_order(side, qty):
    if PAPER_TRADE:
        print(f"📄 PAPER: Orden {side} {qty} BTC simulada")
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
            print(f"❌ Error orden market: {result}")
            return None
    except Exception as e:
        print(f"❌ Excepción place_market_order: {e}")
        return None

def close_position_qty(qty, side_to_close):
    if PAPER_TRADE:
        print(f"📄 PAPER: Cierre simulado de {qty} BTC lado {side_to_close}")
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
            print(f"✅ Orden de cierre enviada: {qty_to_close} BTC")
            return result['result']['orderId']
        else:
            print(f"❌ Error close_position_qty: {result}")
            return None
    except Exception as e:
        print(f"❌ Excepción close_position_qty: {e}")
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
            print(f"✅ Confirmada reducción: {size_before:.4f} -> {size_after:.4f}")
            return order_id
    print(f"❌ No se confirmó reducción tras {max_wait}s")
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
        print("💾 Memoria guardada")
    except Exception as e: print(f"Error guardando memoria: {e}")

def cargar_memoria():
    global TRADE_HISTORY, REGLAS_APRENDIDAS, REAL_BALANCE, WIN_COUNT, LOSS_COUNT
    global TOTAL_TRADES, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, REAL_ACTIVE_TRADES
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    if not os.path.exists(MEMORY_FILE): return
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
        print(f"🧠 Memoria cargada. Trades: {paper_total_trades if PAPER_TRADE else TOTAL_TRADES}")
    except Exception as e: print(f"Error cargando memoria: {e}")

def parse_json_seguro(raw):
    if not raw or raw.strip() == "": return None
    try:
        repaired = json_repair.repair_json(raw)
        return json.loads(repaired)
    except: return None

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

# Límites de seguridad para SL (solo para evitar valores extremos)
MIN_SL_DIST_PCT = 0.001   # 0.1% mínimo
MAX_SL_DIST_PCT = 0.01    # 1% máximo

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
REGLAS_APRENDIDAS = "Aún no hay lecciones. Busca confluencia."
TOKENS_ACUMULADOS = 0

# =================== TELEGRAM (con logs) ===================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram no configurado")
        return
    try:
        if len(texto) > 4000:
            texto = texto[:4000]
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
        if resp.status_code != 200:
            print(f"❌ Error Telegram: {resp.status_code} - {resp.text[:200]}")
        else:
            print("✅ Mensaje enviado a Telegram")
    except Exception as e:
        print(f"❌ Excepción en telegram_mensaje: {e}")

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram no configurado para imagen")
        return
    try:
        if not os.path.exists(ruta_imagen):
            print(f"⚠️ Imagen no encontrada: {ruta_imagen}")
            return
        with open(ruta_imagen, 'rb') as foto:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
        if resp.status_code != 200:
            print(f"❌ Error imagen Telegram: {resp.status_code} - {resp.text[:200]}")
        else:
            print("✅ Imagen enviada a Telegram")
    except Exception as e:
        print(f"❌ Excepción en telegram_enviar_imagen: {e}")

def reporte_estado():
    if PAPER_TRADE:
        balance = paper_balance
        win_count = paper_win_count
        loss_count = paper_loss_count
        total_trades = paper_total_trades
        active_trades = len(paper_positions)
    else:
        if REAL_BALANCE is None: return
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
        f"{modo} **ESTADO BTC**\n"
        f"💰 Balance: {balance:.2f} USDT\n"
        f"📈 PnL día: {pnl_global:+.2f} USDT\n"
        f"🎯 Winrate: {winrate:.1f}%\n"
        f"⚡ Activos: {active_trades}/{max_din}\n"
        f"📐 PF (10t): {ULTIMO_PROFIT_FACTOR:.2f}"
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
        print(f"Error obteniendo velas {interval}: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
    if df.empty: return df
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['tr'] = np.maximum(df['high'] - df['low'],
                          np.maximum(abs(df['high'] - df['close'].shift(1)),
                                     abs(df['low'] - df['close'].shift(1))))
    df['atr'] = df['tr'].rolling(14).mean()
    return df.dropna()

def detectar_zonas_mercado(df, idx=-2):
    if df.empty or len(df) < 40: return 0,0,0,0,"LATERAL","LATERAL"
    df_eval = df if idx == -1 else df.iloc[:idx+1]
    soporte = df_eval['low'].rolling(40).min().iloc[-1]
    resistencia = df_eval['high'].rolling(40).max().iloc[-1]
    y = df_eval['close'].values[-120:]
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    micro_slope, _, _, _, _ = linregress(np.arange(8), df_eval['close'].values[-8:])
    tend = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    micro = 'SUBIENDO' if micro_slope > 0.2 else 'CAYENDO' if micro_slope < -0.2 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tend, micro

def generar_grafico_para_vision(df, titulo, soporte=None, resistencia=None, slope=None, intercept=None):
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
        ax.plot(x_trend, y_trend, color='white', linestyle='-.', linewidth=2, label='Tendencia', alpha=0.7)
    ax.set_title(titulo, color='white', fontsize=14)
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
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

# =================== PROMPT DEFINITIVO (sin restricciones, con todos los campos) ===================
def analizar_con_qwen(img_ltf, img_htf):
    global TOKENS_ACUMULADOS
    try:
        img_ltf_b64 = pil_to_base64(img_ltf)
        img_htf_b64 = pil_to_base64(img_htf)
        
        prompt = """
Eres un trader profesional, objetivo y crítico. No tienes sesgo de confirmación.

**ACTIVO:** BTC/USDT
**MARCO DE TIEMPO:** LTF = 5 minutos, HTF = 1 hora
**PRECIO ACTUAL:** (obsérvalo en el gráfico LTF)
**CONTEXTO DEL ACTIVO:** Bot automatizado con apalancamiento 34x, riesgo máximo 3 USDT por trade. El mercado de criptomonedas es muy volátil. Este análisis es para una operación de corto plazo (minutos/horas).

---

### ESTRUCTURA DEL ANÁLISIS (sigue cada paso)

#### 1. Análisis de tendencia
- **HTF (1h):** ¿alcista, bajista o lateral? Usa la línea de tendencia blanca y la EMA20. ¿Hay máximos/mínimos crecientes o decrecientes?
- **LTF (5m):** misma pregunta. ¿Pendiente de la línea de regresión y relación con EMA20?
- **Alineación:** ¿HTF y LTF coinciden? Si no, ¿cuál es el conflicto?

#### 2. Niveles clave
- **Soporte/resistencia:** líneas cyan y magenta en ambos gráficos.
- **EMA20:** ¿actúa como soporte o resistencia dinámica?
- **Liquidez:** identifica máximos/mínimos iguales recientes o zonas de congestión (stop hunts).
- **Contexto de ruptura o rango:** ¿El precio está en rango (entre soporte y resistencia) o está rompiendo un nivel? Describe si es una ruptura limpia o un fakeout.

#### 3. Evaluación del setup del trade
- **Dirección sugerida:** Long / Short / Neutral.
- **Calidad de entrada:** temprana (recién tocando nivel), óptima (confirmación con vela), tardía (lejos del nivel).
- **Tipo de operación:** ¿momentum (continuación) o reversión a la media?

#### 4. Riesgo / Beneficio (sobre el gráfico, sin reglas fijas)
Define niveles concretos que veas en el gráfico:
- **Entrada:** precio exacto (puede ser el actual o un nivel límite).
- **Stop loss:** precio donde la tesis se invalida (justo debajo de soporte en Long, encima de resistencia en Short). Calcula la **distancia en %**.
- **Take profit 1:** primer objetivo (un nivel visible, puede ser una resistencia previa, un soporte, etc.).
- **Take profit 2:** segundo objetivo (más lejano, donde esperarías una reacción fuerte).
- **Relación R aproximada:** (TP1 - entrada) / (entrada - SL) para Long, o inversa para Short. No la fuerces, calcula la que realmente se obtiene de los niveles que elegiste.
- **¿El trade es asimétrico?** (la ganancia potencial es mucho mayor que la pérdida, o es simétrico, o incluso negativo). Sé honesto.

#### 5. Decisión final
- **¿Debería tomarlo?** (Sí / No / Solo con condiciones)
- **Si sí:** ¿qué tengo que esperar para confirmar la entrada? (ej. que el precio cierre por encima de la EMA, que se respete un soporte, etc.)
- **Si no:** ¿por qué? ¿qué falta?

Responde ÚNICAMENTE con un JSON válido en una línea. La estructura es:
{
  "decision": "Buy/Sell/Hold",
  "razon": "resumen breve de la decisión",
  "contexto_pensamiento": "aquí describes tu razonamiento paso a paso, qué estás considerando, qué te preocupa, etc.",
  "condiciones_espera": "si la decisión es Sí, qué esperar para entrar; si es No o Condiciones, qué falta o qué monitorear",
  "entry_price": 0.0,
  "sl_price": 0.0,
  "tp1_price": 0.0,
  "tp2_price": 0.0,
  "trailing_logic": "BREAKEVEN",
  "analisis": {
    "tendencia_htf": "alcista/bajista/lateral",
    "tendencia_ltf": "alcista/bajista/lateral",
    "alineacion": "alineada/conflicto",
    "zonas_liquidez": "descripción de las zonas de liquidez detectadas",
    "contexto_ruptura_rango": "texto corto",
    "calidad_entrada": "temprana/optima/tardia",
    "tipo_operacion": "momentum/reversion",
    "rr_estimada": 2.5,
    "asimétrico": true/false,
    "decision_texto": "Sí/No/Condiciones"
  }
}
"""
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": img_ltf_b64}},
                    {"type": "image_url", "image_url": {"url": img_htf_b64}}
                ]}
            ],
            temperature=0.2
        )
        TOKENS_ACUMULADOS += response.usage.total_tokens if response.usage else 0
        datos = parse_json_seguro(response.choices[0].message.content)
        if not datos:
            return "Hold", "Error parsing", "", "", None, None, None, None, "BREAKEVEN", {}
        
        decision = datos.get("decision", "Hold")
        razon = datos.get("razon", "")
        contexto_pensamiento = datos.get("contexto_pensamiento", "")
        condiciones_espera = datos.get("condiciones_espera", "")
        entry_price = datos.get("entry_price")
        sl_price = datos.get("sl_price")
        tp1_price = datos.get("tp1_price")
        tp2_price = datos.get("tp2_price")
        trailing = datos.get("trailing_logic", "BREAKEVEN")
        analisis = datos.get("analisis", {})
        
        return decision, razon, contexto_pensamiento, condiciones_espera, entry_price, sl_price, tp1_price, tp2_price, trailing, analisis
    except Exception as e:
        print(f"❌ Error en IA: {e}")
        return "Hold", "Error API", "", "", None, None, None, None, "BREAKEVEN", {}

# =================== GESTIÓN DE RIESGO Y APERTURA (respetando niveles de IA) ===================
def calcular_riesgo_dinamico(free_margin):
    if free_margin >= 20:
        return RISK_PER_TRADE_MAX
    elif free_margin >= 10:
        return 1.5
    else:
        return 1.0

def abrir_posicion_con_ia(decision, precio_actual, razon, contexto_pensamiento, condiciones_espera, sl_ia, tp1_ia, tp2_ia, analisis, df_ltf, sop, res, slope, inter):
    global paper_balance, paper_positions, paper_trade_counter, REAL_BALANCE, TRADE_COUNTER, REAL_ACTIVE_TRADES
    
    if PAPER_TRADE:
        balance = paper_balance
        positions = paper_positions
        counter_ref = paper_trade_counter
        is_paper = True
    else:
        if REAL_BALANCE is None:
            REAL_BALANCE = get_real_balance()
            if REAL_BALANCE is None:
                print("❌ No se pudo obtener balance real")
                return
        balance = REAL_BALANCE
        positions = REAL_ACTIVE_TRADES
        counter_ref = TRADE_COUNTER
        is_paper = False
    
    max_trades = get_dynamic_max_trades()
    if len(positions) >= max_trades:
        print(f"⚠️ Máximo dinámico de trades ({max_trades}) alcanzado.")
        return
    
    free_margin = get_free_margin()
    if free_margin <= 0:
        print("❌ Margen libre insuficiente.")
        return
    
    risk_per_trade = calcular_riesgo_dinamico(free_margin)
    print(f"💰 Riesgo fijado en {risk_per_trade} USDT (margen libre: {free_margin:.2f})")
    
    entrada = precio_actual  # usamos precio actual (se puede cambiar por entry_price de IA si se implementan órdenes límite)
    
    # Validar SL
    if not sl_ia or sl_ia <= 0:
        print("❌ La IA no proporcionó un stop loss válido. Cancelando.")
        return
    # Validar TP
    if not tp1_ia or tp1_ia <= 0:
        print("⚠️ La IA no dio TP1, se usará un TP por defecto (podría no ejecutarse).")
        if tp2_ia and tp2_ia > 0:
            tp1_ia = tp2_ia
        else:
            tp1_ia = entrada * (1.005 if decision=="Buy" else 0.995)  # TP muy conservador
    if not tp2_ia or tp2_ia <= 0:
        tp2_ia = tp1_ia * (1.003 if decision=="Buy" else 0.997) if decision=="Buy" else tp1_ia * (0.997 if decision=="Buy" else 1.003)
    
    # Aplicar límites de seguridad al SL (distancia entre 0.1% y 1%)
    if decision == "Buy":
        distancia_propuesta = entrada - sl_ia
        if distancia_propuesta <= 0:
            print("❌ SL inválido (por encima de entrada en Buy). Cancelando.")
            return
        min_dist = entrada * MIN_SL_DIST_PCT
        max_dist = entrada * MAX_SL_DIST_PCT
        if distancia_propuesta < min_dist:
            print(f"⚠️ SL demasiado cerca ({distancia_propuesta:.1f} USD). Ajustando a mínimo {min_dist:.1f} USD")
            sl_ajustado = entrada - min_dist
        elif distancia_propuesta > max_dist:
            print(f"⚠️ SL demasiado lejos ({distancia_propuesta:.1f} USD). Ajustando a máximo {max_dist:.1f} USD")
            sl_ajustado = entrada - max_dist
        else:
            sl_ajustado = sl_ia
        distancia_final = entrada - sl_ajustado
    else:  # Sell
        distancia_propuesta = sl_ia - entrada
        if distancia_propuesta <= 0:
            print("❌ SL inválido (por debajo de entrada en Sell). Cancelando.")
            return
        min_dist = entrada * MIN_SL_DIST_PCT
        max_dist = entrada * MAX_SL_DIST_PCT
        if distancia_propuesta < min_dist:
            print(f"⚠️ SL demasiado cerca ({distancia_propuesta:.1f} USD). Ajustando a mínimo {min_dist:.1f} USD")
            sl_ajustado = entrada + min_dist
        elif distancia_propuesta > max_dist:
            print(f"⚠️ SL demasiado lejos ({distancia_propuesta:.1f} USD). Ajustando a máximo {max_dist:.1f} USD")
            sl_ajustado = entrada + max_dist
        else:
            sl_ajustado = sl_ia
        distancia_final = sl_ajustado - entrada
    
    # Calcular cantidad
    qty_btc = risk_per_trade / distancia_final
    max_qty = (balance * LEVERAGE) / entrada
    qty_btc = min(qty_btc, max_qty)
    if qty_btc < 0.001:
        print(f"⚠️ Cantidad muy pequeña ({qty_btc:.4f} BTC). No se abre.")
        return
    
    # Margen necesario
    margen_necesario = (qty_btc * entrada) / LEVERAGE
    if margen_necesario > free_margin * 0.98:
        print(f"❌ Margen insuficiente: necesario {margen_necesario:.2f} USDT > libre {free_margin:.2f} USDT.")
        return
    
    # Redondear cantidad y mínimo nocional
    qty_btc = round(qty_btc, 3)
    nocional = qty_btc * entrada
    if nocional < 100.0:
        qty_btc = round(100.0 / entrada, 3)
        print(f"⚠️ Ajustado a nocional mínimo: {qty_btc} BTC (nominal ~{qty_btc*entrada:.2f} USDT)")
        margen_necesario = (qty_btc * entrada) / LEVERAGE
        if margen_necesario > free_margin:
            print(f"❌ Tras ajuste por nocional, margen excedido. Cancelado.")
            return
    
    # Abrir orden
    if is_paper:
        order_id = f"paper_{paper_trade_counter+1}"
        print(f"📄 PAPER: Orden {decision} {qty_btc} BTC simulada")
    else:
        order_id = place_market_order(decision, qty_btc)
        if not order_id:
            print("❌ No se pudo abrir la orden real.")
            return
    
    # Registrar trade
    qty_tp1 = round(qty_btc * TP1_PERCENT, 3)
    qty_restante = round(qty_btc - qty_tp1, 3)
    if is_paper:
        paper_trade_counter += 1
        trade_id = paper_trade_counter
        positions[trade_id] = {
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_inicial": sl_ajustado, "sl_actual": sl_ajustado,
            "tp1": tp1_ia, "tp2": tp2_ia, "trailing_logic": "BREAKEVEN",
            "qty_original": qty_btc, "qty_restante": qty_restante,
            "tp1_ejecutado": False, "tp2_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto_pensamiento, "condiciones_espera": condiciones_espera
        }
        modo = "📄 PAPER"
    else:
        TRADE_COUNTER += 1
        trade_id = TRADE_COUNTER
        positions[trade_id] = {
            "id": trade_id, "decision": decision, "entrada": entrada,
            "sl_inicial": sl_ajustado, "sl_actual": sl_ajustado,
            "tp1": tp1_ia, "tp2": tp2_ia, "trailing_logic": "BREAKEVEN",
            "qty_original": qty_btc, "qty_restante": qty_restante,
            "tp1_ejecutado": False, "tp2_ejecutado": False, "pnl_parcial": 0.0,
            "razon": razon, "order_id": order_id, "breakeven_activado": False,
            "contexto_pensamiento": contexto_pensamiento, "condiciones_espera": condiciones_espera
        }
        modo = "🚀 REAL"
    
    # Construir mensaje detallado para Telegram
    msg = (f"{modo} [#{trade_id}] {decision} en {entrada:.2f} | Qty {qty_btc} BTC (riesgo {risk_per_trade} USDT)\n"
           f"SL: {sl_ajustado:.2f} (dist {distancia_final:.1f} USD)\n"
           f"TP1: {tp1_ia:.2f} | TP2: {tp2_ia:.2f}\n"
           f"Razón: {razon}\n"
           f"🧠 Contexto pensamiento: {contexto_pensamiento[:200]}\n"
           f"⏳ Condiciones espera: {condiciones_espera[:150]}\n"
           f"📊 Análisis: HTF={analisis.get('tendencia_htf','?')} LTF={analisis.get('tendencia_ltf','?')} | {analisis.get('alineacion','')}\n"
           f"🔍 Liquidez: {analisis.get('zonas_liquidez','')}\n"
           f"📐 Contexto: {analisis.get('contexto_ruptura_rango','')}\n"
           f"🎯 Calidad: {analisis.get('calidad_entrada','')} | Tipo: {analisis.get('tipo_operacion','')}\n"
           f"📏 R:R≈{analisis.get('rr_estimada','?')} | Asimétrico: {analisis.get('asimétrico',False)} | Decisión: {analisis.get('decision_texto','')}\n"
           f"💰 Margen usado: {margen_necesario:.2f} / {free_margin:.2f} USDT")
    print(msg)
    telegram_mensaje(msg)
    
    # Enviar gráfico de entrada
    img_completa = generar_grafico_para_vision(df_ltf, "Entrada - 5m", sop, res, slope, inter)
    if img_completa:
        img_completa.save("/tmp/in_completo.png")
        telegram_enviar_imagen("/tmp/in_completo.png", msg)
    
    guardar_memoria()

# =================== GESTIÓN DE TRADES ACTIVOS (sin cambios relevantes) ===================
def sync_active_trades_with_bybit():
    if PAPER_TRADE:
        return
    global REAL_ACTIVE_TRADES
    real_size = get_real_position_size()
    if real_size == 0.0 and REAL_ACTIVE_TRADES:
        print("🧹 Sincronización: No hay posición real. Limpiando trades fantasmas.")
        REAL_ACTIVE_TRADES.clear()
        guardar_memoria()
    elif real_size > 0.0 and not REAL_ACTIVE_TRADES:
        print("⚠️ Hay posición real pero el bot no la registra.")
    else:
        mem_size = sum(t['qty_restante'] for t in REAL_ACTIVE_TRADES.values())
        if abs(mem_size - real_size) > 0.002:
            print(f"⚠️ Discrepancia de tamaño: memoria {mem_size:.3f} BTC, real {real_size:.3f} BTC. Reconstruyendo...")
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
    global paper_balance, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    if not paper_positions:
        return
    h = df['high'].iloc[-1]
    l = df['low'].iloc[-1]
    cerrar_ids = []
    for tid, t in list(paper_positions.items()):
        # TP1
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
                    print(f"📄 PAPER: TP1 #{tid} +{pnl_parcial:.2f} USDT, resto SL breakeven")
                    telegram_mensaje(f"📄 PAPER TP1 #{tid}: +{pnl_parcial:.2f} USDT (cerrado {qty_tp1} BTC)")
                    if t['qty_restante'] <= 0.0001:
                        cerrar_ids.append(tid)
                else:
                    cerrar_ids.append(tid)
        # TP2
        if t['tp1_ejecutado'] and not t['tp2_ejecutado'] and t['tp2'] is not None and t['tp2'] > 0 and t['qty_restante'] > 0:
            if (t['decision']=="Buy" and h >= t['tp2']) or (t['decision']=="Sell" and l <= t['tp2']):
                qty_restante = t['qty_restante']
                if qty_restante >= 0.001:
                    pnl_resto = (t['tp2'] - t['entrada']) * qty_restante if t['decision']=="Buy" else (t['entrada'] - t['tp2']) * qty_restante
                    pnl_total = t['pnl_parcial'] + pnl_resto
                    paper_balance += pnl_total
                    paper_total_trades += 1
                    if pnl_total > 0: paper_win_count += 1
                    else: paper_loss_count += 1
                    paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                    cerrar_ids.append(tid)
                    msg = f"📄 PAPER CIERRE #{tid} TP2 - PnL: {pnl_total:+.2f} USDT"
                    print(msg); telegram_mensaje(msg); reporte_estado()
                else: cerrar_ids.append(tid)
        # Stop Loss
        if t['qty_restante'] > 0.001:
            cond = (t['decision']=="Buy" and l <= t['sl_actual']) or (t['decision']=="Sell" and h >= t['sl_actual'])
            if cond:
                qty_restante = t['qty_restante']
                pnl_resto = (t['sl_actual'] - t['entrada'])*qty_restante if t['decision']=="Buy" else (t['entrada'] - t['sl_actual'])*qty_restante
                pnl_total = t['pnl_parcial'] + pnl_resto
                paper_balance += pnl_total
                paper_total_trades += 1
                if pnl_total>0: paper_win_count+=1
                else: paper_loss_count+=1
                paper_trade_history.append(convertir_serializable({"pnl": pnl_total, "resultado_win": pnl_total>0, "decision": t['decision'], "razon": t['razon']}))
                cerrar_ids.append(tid)
                msg = f"📄 PAPER CIERRE #{tid} por SL - PnL: {pnl_total:+.2f} USDT"
                print(msg); telegram_mensaje(msg); reporte_estado()
    for tid in cerrar_ids:
        del paper_positions[tid]
    if paper_total_trades>0 and paper_total_trades%10==0 and paper_total_trades != ULTIMO_APRENDIZAJE:
        aprender_de_trades()

def real_revisar_sl_tp(df):
    global REAL_BALANCE, WIN_COUNT, LOSS_COUNT, TOTAL_TRADES, TRADE_HISTORY, REAL_ACTIVE_TRADES
    if not REAL_ACTIVE_TRADES: return
    h = df['high'].iloc[-1]; l = df['low'].iloc[-1]
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
                        t['breakeven_activado']=True
                        offset=2.0
                        t['sl_actual']=t['entrada']-offset if t['decision']=="Buy" else t['entrada']+offset
                        print(f"🎯 TP1 #{tid} +{pnl_parcial:.2f} USDT")
                        telegram_mensaje(f"🎯 TP1 #{tid}: +{pnl_parcial:.2f} USDT")
                        if t['qty_restante']<=0.0001: cerrar_ids.append(tid)
                    else:
                        print(f"⚠️ TP1 no confirmado #{tid}")
                else:
                    cerrar_ids.append(tid)
        # TP2
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
                        if pnl_total>0: WIN_COUNT+=1
                        else: LOSS_COUNT+=1
                        TRADE_HISTORY.append(convertir_serializable({"pnl":pnl_total, "resultado_win":pnl_total>0, "decision":t['decision'], "razon":t['razon']}))
                        cerrar_ids.append(tid)
                        msg = f"✅ CIERRE COMPLETO #{tid} TP2 - PnL: {pnl_total:+.2f} USDT"
                        print(msg); telegram_mensaje(msg); reporte_estado()
                    else:
                        print(f"❌ Falló cierre TP2 #{tid}")
                else:
                    cerrar_ids.append(tid)
        # Stop Loss
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
                    if pnl_total>0: WIN_COUNT+=1
                    else: LOSS_COUNT+=1
                    TRADE_HISTORY.append(convertir_serializable({"pnl":pnl_total, "resultado_win":pnl_total>0, "decision":t['decision'], "razon":t['razon']}))
                    cerrar_ids.append(tid)
                    motivo = "Stop Loss inicial" if not t.get('breakeven_activado') else "Breakeven"
                    msg = f"❌ CIERRE #{tid} por {motivo} - PnL: {pnl_total:+.2f} USDT"
                    print(msg); telegram_mensaje(msg); reporte_estado()
                else:
                    print(f"❌ Falló cierre por stop #{tid}")
    for tid in cerrar_ids:
        del REAL_ACTIVE_TRADES[tid]
    if TOTAL_TRADES>0 and TOTAL_TRADES%10==0 and TOTAL_TRADES!=ULTIMO_APRENDIZAJE:
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
            resumen = f"📊 APRENDIZAJE PAPER #{paper_total_trades}\nWinrate: {winrate:.1f}% PF: {ULTIMO_PROFIT_FACTOR:.2f}"
        else:
            ult = TRADE_HISTORY[-10:] if len(TRADE_HISTORY)>=10 else TRADE_HISTORY
            gan = sum(t['pnl'] for t in ult if t['pnl']>0)
            per = abs(sum(t['pnl'] for t in ult if t['pnl']<0))
            ULTIMO_PROFIT_FACTOR = gan/per if per>0 else 1.0
            winrate = (WIN_COUNT/TOTAL_TRADES*100) if TOTAL_TRADES>0 else 0
            resumen = f"📊 APRENDIZAJE #{TOTAL_TRADES}\nWinrate: {winrate:.1f}% PF: {ULTIMO_PROFIT_FACTOR:.2f}"
        telegram_mensaje(resumen)
        try:
            ult_serial = convertir_serializable(ult)
            prompt = f"Analiza estos 10 trades y da una lección corta (max 200 chars): {json.dumps(ult_serial)}"
            resp = client.chat.completions.create(model=MODELO_VISION, messages=[{"role":"user","content":prompt}], timeout=10)
            REGLAS_APRENDIDAS = resp.choices[0].message.content
            telegram_mensaje(f"🧠 Lección IA: {REGLAS_APRENDIDAS}")
        except: pass
    except Exception as e: print(f"Error aprendizaje: {e}")
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
        print(f"📅 Nuevo día: {hoy}. Balance inicial: {balance:.2f}")
    balance_actual = paper_balance if PAPER_TRADE else REAL_BALANCE
    if balance_actual is not None and DAILY_START_BALANCE is not None:
        drawdown = (balance_actual - DAILY_START_BALANCE) / DAILY_START_BALANCE
        if drawdown <= -MAX_DAILY_DRAWDOWN_PCT:
            STOPPED_TODAY = True
            print(f"🚨 Drawdown diario superado. Operaciones detenidas.")
    return not STOPPED_TODAY

# =================== LOOP PRINCIPAL ===================
def run_bot():
    global REAL_BALANCE, ULTIMO_APRENDIZAJE, TOKENS_ACUMULADOS, ULTIMO_PROFIT_FACTOR, TRADE_HISTORY, REAL_ACTIVE_TRADES
    global paper_balance, paper_trade_counter, paper_win_count, paper_loss_count, paper_total_trades, paper_trade_history, paper_positions
    cargar_memoria()
    set_leverage()
    
    telegram_mensaje("🤖 Bot iniciando - Modo análisis estructurado paso a paso sin restricciones forzadas")
    
    if PAPER_TRADE:
        print(f"📄 Iniciando PAPER TRADE - Saldo: {paper_balance:.2f} USDT")
        telegram_mensaje(f"📄 Bot Paper Trade Online - Saldo simulado: {paper_balance:.2f} USDT - Riesgo máx 3 USDT")
    else:
        REAL_BALANCE = get_real_balance()
        if REAL_BALANCE is None:
            print("❌ No se pudo obtener saldo real. Abortando.")
            return
        print(f"🤖 BOT REAL - Balance: {REAL_BALANCE:.2f} USDT")
        telegram_mensaje(f"🤖 Bot Real Online - Balance: {REAL_BALANCE:.2f} USDT")
    
    ultima_vela = None
    iteracion = 0
    while True:
        try:
            iteracion += 1
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
            vela_c = df_ltf.index[-2]
            if ultima_vela is None:
                ultima_vela = vela_c
            
            active_count = len(paper_positions) if PAPER_TRADE else len(REAL_ACTIVE_TRADES)
            if active_count < max_trades_actual and ultima_vela != vela_c:
                if risk_management_check():
                    sop_ltf, res_ltf, slope_ltf, inter_ltf, _, _ = detectar_zonas_mercado(df_ltf)
                    sop_htf, res_htf, slope_htf, inter_htf, _, _ = detectar_zonas_mercado(df_htf)
                    
                    img_ltf = generar_grafico_para_vision(df_ltf, "BTC/USDT 5m (LTF)", sop_ltf, res_ltf, slope_ltf, inter_ltf)
                    img_htf = generar_grafico_para_vision(df_htf, "BTC/USDT 1h (HTF)", sop_htf, res_htf, slope_htf, inter_htf)
                    
                    if img_ltf and img_htf:
                        dec, raz, ctx_pens, cond_esp, entry_ia, sl_ia, tp1_ia, tp2_ia, trailing, analisis = analizar_con_qwen(img_ltf, img_htf)
                        print(f"🤖 Decisión IA: {dec} - Razón: {raz}")
                        print(f"🧠 Contexto pensamiento: {ctx_pens[:200]}")
                        print(f"⏳ Condiciones espera: {cond_esp[:150]}")
                        print(f"📊 Análisis detallado: {analisis}")
                        
                        if dec in ["Buy","Sell"]:
                            abrir_posicion_con_ia(dec, precio_actual, raz, ctx_pens, cond_esp, sl_ia, tp1_ia, tp2_ia, analisis, df_ltf, sop_ltf, res_ltf, slope_ltf, inter_ltf)
                        else:
                            print(f"⏸️ IA decidió HOLD. Motivo: {raz[:100]}")
                ultima_vela = vela_c
            else:
                if ultima_vela == vela_c:
                    print("⏳ Misma vela, no se repite análisis.")
                else:
                    print(f"⏸️ Límite de trades alcanzado ({active_count}/{max_trades_actual})")
            
            if PAPER_TRADE and paper_positions:
                revisar_sl_tp_simulado(df_ltf)
            elif not PAPER_TRADE and REAL_ACTIVE_TRADES:
                real_revisar_sl_tp(df_ltf)
            
            if iteracion % 10 == 0:
                reporte_estado()
                if PAPER_TRADE:
                    winrate = (paper_win_count/paper_total_trades*100) if paper_total_trades>0 else 0
                    print(f"📈 RESUMEN PAPER: Balance={paper_balance:.2f} | Trades={paper_total_trades} | Winrate={winrate:.1f}% | PF={ULTIMO_PROFIT_FACTOR:.2f}")
                else:
                    winrate = (WIN_COUNT/TOTAL_TRADES*100) if TOTAL_TRADES>0 else 0
                    print(f"📈 RESUMEN REAL: Balance={REAL_BALANCE:.2f} | Trades={TOTAL_TRADES} | Winrate={winrate:.1f}% | PF={ULTIMO_PROFIT_FACTOR:.2f}")
            
            time.sleep(SLEEP_SECONDS)
        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {e}")
            import traceback; traceback.print_exc()
            time.sleep(30)

if __name__ == '__main__':
    run_bot()
