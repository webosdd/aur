import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from scipy.stats import linregress
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io
import base64
import re
import schedule
import threading
import hashlib
import hmac
from openai import OpenAI
from pybit.unified_trading import HTTP
import websocket

# ================= CONFIGURACIÓN LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot_duplo.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ================= API KEYS =================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Falta OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("Faltan BYBIT_API_KEY o BYBIT_API_SECRET")

# ================= CONFIGURACIÓN GENERAL =================
MODELO_VISION = "google/gemini-3.1-flash-image-preview"
SYMBOLS = ["BTCUSDT", "SOLUSDT"]          # Pares a analizar (diario)
MAX_INVEST_PERCENT = 0.5                  # Invertir máximo el 50% del saldo
MIN_INVEST_USDT = 10                      # Monto mínimo para invertir
HORA_EJECUCION = "07:00"                  # Hora UTC del análisis diario

# Variables globales
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Dual Asset Bot"})
bybit_session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# Variables para persistir la decisión del día (sin repetir IA)
decision_guardada = {}      # {"BTC": "Buy Low", "SOL": "Sell High"}
fecha_decision = None

# WebSocket de ofertas
ws_offers = {}
ws_connected = False
product_id_cache = {}       # cache: {"BTC": ["123", "456"], "SOL": [...]}
cache_timestamp = 0
CACHE_DURATION = 300        # 5 minutos

# ================= TELEGRAM =================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

def telegram_enviar_imagen(ruta, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        with open(ruta, 'rb') as foto:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto",
                          data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption},
                          files={"photo": foto}, timeout=15)
    except Exception as e:
        logger.error(f"Error enviando imagen: {e}")

# ================= OBTENER VELAS DIARIAS =================
def obtener_velas_diarias(symbol, limit=100):
    """Velas de 1 día (intervalo 'D') para análisis de tendencia."""
    try:
        resp = bybit_session.get_kline(category="spot", symbol=symbol, interval="D", limit=limit)
        if resp.get("retCode") != 0:
            return pd.DataFrame()
        lista = resp["result"]["list"][::-1]  # orden cronológico
        df = pd.DataFrame(lista, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error velas diarias {symbol}: {e}")
        return pd.DataFrame()

def calcular_indicadores_diarios(df):
    if df.empty:
        return df
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    # MACD
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    # Tendencia por regresión lineal
    y = df['close'].values[-60:]  # últimos 60 días
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    df['trend_slope'] = slope
    return df.dropna()

def detectar_soportes_resistencias_diario(df):
    if df.empty or len(df) < 20:
        return 0, 0
    soporte = df['low'].rolling(20).min().iloc[-1]
    resistencia = df['high'].rolling(20).max().iloc[-1]
    return soporte, resistencia

def generar_grafico_diario(df, symbol, soporte, resistencia, slope, intercept):
    if df.empty:
        return None
    df_plot = df.tail(120).copy()
    fig, ax = plt.subplots(figsize=(16, 8))
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
    if slope != 0:
        x_trend = np.array([0, len(df_plot)-1])
        y_trend = intercept + slope * x_trend
        ax.plot(x_trend, y_trend, color='white', linestyle='-.', lw=2, label='Tendencia', alpha=0.7)
    # RSI
    ax2 = ax.twinx()
    ax2.plot(x, df_plot['rsi'], color='orange', lw=1.5, alpha=0.7, label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI', color='orange')
    # MACD
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x, df_plot['macd'], color='blue', lw=1, label='MACD')
    ax3.plot(x, df_plot['signal'], color='red', lw=1, label='Signal')
    ax3.bar(x, df_plot['histogram'], color='gray', alpha=0.3, width=0.8)
    ax3.set_ylabel('MACD', color='blue')
    ax.set_title(f"Dual Asset - {symbol} (Análisis Diario)", color='white', fontsize=14)
    ax.set_xlabel('Días', color='white')
    ax.set_ylabel('Precio (USDT)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.set_facecolor('#121212')
    fig.patch.set_facecolor('#121212')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
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

# ================= IA (SOLO UNA VEZ AL DÍA) =================
def analizar_con_gemini(img, symbol):
    try:
        img_b64 = pil_to_base64(img)
        prompt = f"""
        Eres un analista experto en Dual Asset de Bybit.
        Revisa el gráfico DIARIO de {symbol} (velas diarias) con EMAs, RSI, MACD y niveles de soporte/resistencia.
        Tu tarea es predecir si el precio dentro de 1 día (mañana) será más alto o más bajo que el precio actual.
        - Si crees que el precio será más alto mañana, recomienda **Sell High** (porque venderás caro).
        - Si crees que el precio será más bajo mañana, recomienda **Buy Low** (porque comprarás barato).
        - Si no hay claridad, recomienda Hold.
        Reglas:
        - RSI > 70 + resistencia fuerte → probable caída → Buy Low.
        - RSI < 30 + soporte fuerte → probable subida → Sell High.
        - Tendencia alcista fuerte (EMA20 > EMA50 + pendiente positiva) → Sell High.
        - Tendencia bajista fuerte → Buy Low.
        Responde SOLO con un JSON en UNA línea: 
        {{"decision": "Buy Low"/"Sell High"/"Hold", "razon": "explicación breve", "explicacion": "detallada", "confianza": 0-100}}
        """
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_b64}}]}],
            temperature=0.2,
            timeout=60
        )
        contenido = response.choices[0].message.content
        # Extraer JSON
        inicio = contenido.find('{')
        fin = contenido.rfind('}') + 1
        if inicio != -1 and fin != 0:
            contenido = contenido[inicio:fin]
        datos = json.loads(contenido)
        return datos.get("decision"), datos.get("razon"), datos.get("explicacion"), datos.get("confianza", 0)
    except Exception as e:
        logger.error(f"Error IA {symbol}: {e}")
        return "Hold", f"Error: {e}", "", 0

# ================= OBTENER LISTA DE PRODUCTOS POR ACTIVO (API REST) =================
def obtener_lista_productos_por_activo(symbol_base):
    """Obtiene los productId de DoubleWin para un activo base usando el endpoint REST."""
    global product_id_cache, cache_timestamp
    ahora = time.time()
    
    # Si el cache no ha expirado y ya tenemos la lista para este símbolo, la devolvemos
    if ahora - cache_timestamp < CACHE_DURATION:
        if symbol_base in product_id_cache:
            return product_id_cache[symbol_base]
    
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = "category=DoubleWin"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        url = f"https://api.bybit.com/v5/earn/advance/product-info?{query}"
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        
        if data.get("retCode") != 0:
            logger.error(f"Error obteniendo productos: {data}")
            return []
        
        # Filtrar productos que correspondan al símbolo base
        productos = []
        for prod in data.get("result", {}).get("list", []):
            # El campo "symbol" viene como "BTCUSDT" o "SOLUSDT"
            if prod.get("symbol") == f"{symbol_base}USDT":
                productos.append(prod.get("productId"))
        
        # Actualizar cache
        product_id_cache[symbol_base] = productos
        cache_timestamp = ahora
        
        logger.info(f"IDs de producto para {symbol_base}: {productos}")
        return productos
    except Exception as e:
        logger.error(f"Excepción obteniendo productos: {e}")
        return []

# ================= WEBSOCKET DE OFERTAS (DOUBLEWIN) =================
def on_message(ws, message):
    global ws_offers
    try:
        data = json.loads(message)
        if "data" in data:
            for product in data["data"]:
                ws_offers[str(product["p"])] = product  # Aseguramos que la clave sea string
    except Exception as e:
        logger.error(f"Error procesando mensaje WS: {e}")

def on_error(ws, error):
    logger.error(f"WS error: {error}")

def on_close(ws, close_status_code, close_msg):
    global ws_connected
    logger.warning("WS cerrado, reconectando...")
    ws_connected = False
    time.sleep(5)
    start_websocket()

def on_open(ws):
    global ws_connected
    # Suscribirse al topic correcto para DoubleWin
    ws.send(json.dumps({"op": "subscribe", "args": ["earn.doublewin.offers"]}))
    ws_connected = True
    logger.info("WebSocket DoubleWin conectado y suscrito")

def start_websocket():
    websocket.enableTrace(False)
    ws_url = "wss://stream.bybit.com/v5/public/fp"
    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()
    return ws

# ================= OBTENER MEJOR OFERTA =================
def obtener_mejor_oferta(symbol_base, decision):
    """Busca en ws_offers la oferta con mayor APY para la dirección indicada."""
    target_key = "b" if decision == "Buy Low" else "s"
    
    # Obtener lista de productId válidos para este activo
    product_ids_validos = obtener_lista_productos_por_activo(symbol_base)
    if not product_ids_validos:
        logger.warning(f"No se encontraron IDs de producto para {symbol_base}")
        return None

    mejores = []
    for pid, product in ws_offers.items():
        # Verificar que el productId esté en la lista de válidos
        if pid not in product_ids_validos:
            continue
        
        if target_key in product:
            for quote in product[target_key]:
                apy = int(quote.get("a", 0)) / 1e8
                max_amount = float(quote.get("m", 0))
                select_price = float(quote.get("s", 0))
                expire_time = int(quote.get("x", 0))
                if expire_time > int(time.time()) * 1000:  # no expirada
                    mejores.append({
                        "productId": pid,
                        "apy": apy,
                        "maxAmount": max_amount,
                        "selectPrice": select_price,
                        "expireTime": expire_time
                    })
    
    if not mejores:
        return None
    return max(mejores, key=lambda x: x["apy"])

# ================= OBTENER QUOTE FIJO =================
def obtener_quote_fijo(product_id):
    """GET /v5/earn/advance/product-extra-info?category=DoubleWin&productId=..."""
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = f"category=DoubleWin&productId={product_id}"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        url = f"https://api.bybit.com/v5/earn/advance/product-extra-info?{query}"
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            return {
                "leverage": data["result"]["leverage"],
                "maxInvestmentAmount": float(data["result"]["maxInvestmentAmount"]),
                "currentPrice": float(data["result"]["currentPrice"])
            }
        else:
            logger.error(f"Error en quote: {data}")
            return None
    except Exception as e:
        logger.error(f"Excepción quote: {e}")
        return None

# ================= SUSCRIPCIÓN A DUAL ASSET =================
def suscribir_dual_asset(product_id, amount_usdt, decision, quote_info, order_link_id=None):
    if order_link_id is None:
        order_link_id = f"duplo_{int(time.time())}"
    # La moneda base para la suscripción:
    # - Buy Low: se invierte USDT
    # - Sell High: se invierte la moneda base (BTC, SOL)
    coin = "USDT" if decision == "Buy Low" else "BTC"  # Ajustar según producto real
    
    body = {
        "category": "DoubleWin",
        "productId": product_id,
        "orderType": "Stake",
        "amount": str(amount_usdt),
        "accountType": "UNIFIED",
        "coin": coin,
        "orderLinkId": order_link_id,
        "doubleWinStakeExtra": {
            "leverage": quote_info["leverage"],
            "initialPrice": str(quote_info["currentPrice"])
        }
    }
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        body_json = json.dumps(body)
        payload = timestamp + BYBIT_API_KEY + recv_window + body_json
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }
        url = "https://api.bybit.com/v5/earn/advance/place-order"
        resp = requests.post(url, headers=headers, data=body_json, timeout=15)
        result = resp.json()
        if result.get("retCode") == 0:
            logger.info(f"Suscripción exitosa: {product_id} {amount_usdt} USDT")
            telegram_mensaje(f"✅ Suscripción Dual Asset: {product_id} | {decision} | {amount_usdt} USDT | APY {quote_info.get('apy', '?')}%")
            return True
        else:
            logger.error(f"Suscripción falló: {result}")
            telegram_mensaje(f"❌ Error suscripción: {result.get('retMsg')}")
            return False
    except Exception as e:
        logger.error(f"Excepción suscripción: {e}")
        return False

# ================= SALDO Y LIQUIDACIONES =================
def obtener_saldo_usdt():
    try:
        resp = bybit_session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        for coin in resp["result"]["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0
    except Exception as e:
        logger.error(f"Error saldo: {e}")
        return 0.0

def verificar_liquidaciones():
    """Revisa posiciones vencidas y convierte a USDT si es necesario."""
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = "category=DoubleWin"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        url = "https://api.bybit.com/v5/earn/advance/position"
        resp = requests.get(url, headers=headers, params={"category": "DoubleWin"}, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            for pos in data["result"]["list"]:
                if pos.get("status") == "Settled" and pos.get("coin") != "USDT":
                    amount = float(pos.get("amount", 0))
                    coin = pos.get("coin")
                    if amount > 0 and coin:
                        convertir_a_usdt(coin, amount)
                        telegram_mensaje(f"🔄 Liquidación automática: {amount} {coin} convertido a USDT")
    except Exception as e:
        logger.error(f"Error liquidaciones: {e}")

def convertir_a_usdt(coin, amount):
    try:
        symbol = f"{coin}USDT"
        ticker = bybit_session.get_ticker(category="spot", symbol=symbol)
        price = float(ticker["result"]["list"][0]["lastPrice"])
        order = bybit_session.place_order(category="spot", symbol=symbol, side="Sell", orderType="Market", qty=str(amount))
        if order["retCode"] == 0:
            usdt_received = amount * price
            logger.info(f"Convertido {amount} {coin} -> {usdt_received:.2f} USDT")
            telegram_mensaje(f"💱 Conversión: {amount} {coin} -> {usdt_received:.2f} USDT")
    except Exception as e:
        logger.error(f"Error conversión: {e}")

# ================= ESTRATEGIA DIARIA (UNA SOLA IA) =================
def ejecutar_estrategia():
    global decision_guardada, fecha_decision
    hoy = datetime.now().date()
    # Si ya se ejecutó hoy, no repetir IA
    if fecha_decision == hoy and decision_guardada:
        logger.info("Ya se ejecutó la IA hoy. Usando decisión guardada.")
        # Usar la decisión guardada para intentar suscribir (sin IA)
        for symbol in SYMBOLS:
            base = symbol.replace("USDT", "")
            if base in decision_guardada:
                decision = decision_guardada[base]
                logger.info(f"Reintento suscripción para {base} con decisión {decision} (sin IA)")
                saldo = obtener_saldo_usdt()
                if saldo < MIN_INVEST_USDT:
                    continue
                monto = saldo * MAX_INVEST_PERCENT
                if monto < MIN_INVEST_USDT:
                    monto = MIN_INVEST_USDT
                oferta = obtener_mejor_oferta(base, decision)
                if oferta:
                    quote = obtener_quote_fijo(oferta["productId"])
                    if quote:
                        monto_final = min(monto, oferta["maxAmount"], quote["maxInvestmentAmount"])
                        if monto_final >= MIN_INVEST_USDT:
                            suscribir_dual_asset(oferta["productId"], monto_final, decision, {**quote, "apy": oferta["apy"]})
        return

    logger.info("=== ANÁLISIS DIARIO CON IA ===")
    saldo_usdt = obtener_saldo_usdt()
    if saldo_usdt < MIN_INVEST_USDT:
        telegram_mensaje(f"Saldo insuficiente: {saldo_usdt:.2f} USDT. No se invierte hoy.")
        return

    for symbol in SYMBOLS:
        base = symbol.replace("USDT", "")
        df = obtener_velas_diarias(symbol, limit=100)
        if df.empty:
            continue
        df = calcular_indicadores_diarios(df)
        soporte, resistencia = detectar_soportes_resistencias_diario(df)
        slope = df['trend_slope'].iloc[-1] if 'trend_slope' in df else 0
        intercept = 0  # no necesario para el gráfico
        img = generar_grafico_diario(df, symbol, soporte, resistencia, slope, intercept)
        if img:
            img_path = f"/tmp/duplo_{symbol}.png"
            img.save(img_path)
            telegram_enviar_imagen(img_path, caption=f"📊 Análisis Diario {symbol}")
            decision, razon, explicacion, confianza = analizar_con_gemini(img, symbol)
        else:
            decision, razon, confianza = "Hold", "Error gráfico", 0

        logger.info(f"{symbol}: IA -> {decision} (confianza {confianza}) - {razon}")
        telegram_mensaje(f"🤖 {symbol}: {decision} (confianza {confianza}%)\n📝 {razon}")

        if decision != "Hold" and confianza >= 60:
            # Guardar decisión para el resto del día
            decision_guardada[base] = decision
            # Intentar suscribir ahora
            monto = saldo_usdt * MAX_INVEST_PERCENT
            if monto < MIN_INVEST_USDT:
                monto = MIN_INVEST_USDT
            oferta = obtener_mejor_oferta(base, decision)
            if oferta:
                quote = obtener_quote_fijo(oferta["productId"])
                if quote:
                    monto_final = min(monto, oferta["maxAmount"], quote["maxInvestmentAmount"])
                    if monto_final >= MIN_INVEST_USDT:
                        suscribir_dual_asset(oferta["productId"], monto_final, decision, {**quote, "apy": oferta["apy"]})
                    else:
                        logger.warning(f"Monto {monto_final} muy pequeño para {base}")
                else:
                    logger.warning(f"No se obtuvo quote para {oferta['productId']}")
            else:
                logger.warning(f"No hay oferta para {base} con dirección {decision}")
        else:
            telegram_mensaje(f"⚠️ {symbol}: No se invierte (Hold o confianza baja)")

    fecha_decision = hoy
    verificar_liquidaciones()

# ================= KEEP ALIVE =================
def keep_alive():
    def ping():
        try:
            requests.get("https://railway.app/health", timeout=5)
        except:
            pass
        threading.Timer(600, ping).start()
    threading.Timer(600, ping).start()

# ================= MAIN =================
def main():
    logger.info("🚀 Bot Dual Asset (Análisis Diario) iniciado")
    telegram_mensaje("🚀 Bot Dual Asset activo - Estrategia diaria con IA única")
    start_websocket()
    keep_alive()
    # Esperar a que WebSocket tenga datos
    time.sleep(5)
    # Ejecutar diariamente a la hora configurada
    schedule.every().day.at(HORA_EJECUCION).do(ejecutar_estrategia)
    # También verificar liquidaciones cada 2 horas
    schedule.every(2).hours.do(verificar_liquidaciones)
    # Ejecutar una vez al inicio
    ejecutar_estrategia()
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
