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
from openai import OpenAI
from pybit.unified_trading import HTTP
import websocket
import threading
import hashlib
import hmac

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
SYMBOLS = ["BTCUSDT", "SOLUSDT"]          # Monedas a analizar
MAX_INVEST_PERCENT = 0.5                  # Invertir máximo 50% del saldo
DAILY_ANALYSIS_HOUR = 7                   # Hora UTC para análisis diario (7 AM)
DUAL_ASSET_DURATION = 1                   # 1 día

# Inicializar clientes
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Dual Asset Bot"})
bybit_session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# Variables globales
ws_offers = {}        # Almacena ofertas recibidas por WebSocket
last_ia_decision = {}  # Guarda la última decisión de IA para cada símbolo (para reintentos sin IA)
subscription_active = False  # Evita múltiples suscripciones el mismo día

# ================= TELEGRAM =================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram no configurado")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e:
        logger.error(f"Error enviando mensaje a Telegram: {e}")

def telegram_enviar_imagen(ruta_imagen, caption=""):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        with open(ruta_imagen, 'rb') as foto:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
        logger.info("Imagen enviada a Telegram")
    except Exception as e:
        logger.error(f"Error enviando imagen: {e}")

# ================= FUNCIONES DE MERCADO (Velas diarias) =================
def obtener_velas_diarias(symbol="BTCUSDT", limit=200):
    """Obtiene velas diarias para análisis de tendencia."""
    try:
        resp = bybit_session.get_kline(category="spot", symbol=symbol, interval="D", limit=limit)
        if resp.get("retCode") != 0:
            return pd.DataFrame()
        lista = resp["result"]["list"][::-1]  # Orden cronológico
        df = pd.DataFrame(lista, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error obteniendo velas diarias de {symbol}: {e}")
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
    # Soporte/resistencia (máximos/mínimos de 20 días)
    df['soporte'] = df['low'].rolling(20).min()
    df['resistencia'] = df['high'].rolling(20).max()
    return df.dropna()

def detectar_zonas_mercado_diario(df):
    if df.empty or len(df) < 20:
        return 0, 0, 0, 0, "LATERAL", "LATERAL"
    soporte = df['low'].rolling(20).min().iloc[-1]
    resistencia = df['high'].rolling(20).max().iloc[-1]
    y = df['close'].values[-120:]
    slope, intercept, _, _, _ = linregress(np.arange(len(y)), y)
    micro_slope, _, _, _, _ = linregress(np.arange(8), df['close'].values[-8:])
    tend = 'ALCISTA' if slope > 0.01 else 'BAJISTA' if slope < -0.01 else 'LATERAL'
    micro = 'SUBIENDO' if micro_slope > 0.2 else 'CAYENDO' if micro_slope < -0.2 else 'LATERAL'
    return soporte, resistencia, slope, intercept, tend, micro

def generar_grafico_diario(df, titulo, soporte, resistencia, slope, intercept):
    if df.empty:
        return None
    df_plot = df.tail(120).copy()
    if len(df_plot) < 3:
        return None
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(16, 8))
    x = np.arange(len(df_plot))
    # Velas
    for i in range(len(df_plot)):
        o, h, l, c = df_plot['open'].iloc[i], df_plot['high'].iloc[i], df_plot['low'].iloc[i], df_plot['close'].iloc[i]
        color = '#00ff00' if c >= o else '#ff0000'
        ax.vlines(x[i], l, h, color=color, linewidth=1.5)
        ax.add_patch(plt.Rectangle((x[i]-0.35, min(o,c)), 0.7, max(abs(c-o), 0.1), color=color, alpha=0.9))
    if soporte:
        ax.axhline(soporte, color='cyan', ls='--', lw=2, label='Soporte (20d)')
    if resistencia:
        ax.axhline(resistencia, color='magenta', ls='--', lw=2, label='Resistencia (20d)')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
    if slope != 0:
        x_trend = np.array([0, len(df_plot)-1])
        y_trend = intercept + slope * x_trend
        ax.plot(x_trend, y_trend, color='white', linestyle='-.', lw=2, label='Tendencia', alpha=0.7)
    # RSI en eje secundario
    ax2 = ax.twinx()
    ax2.plot(x, df_plot['rsi'], color='orange', lw=1.5, alpha=0.7, label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI', color='orange')
    # MACD en tercer eje
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x, df_plot['macd'], color='blue', lw=1, label='MACD')
    ax3.plot(x, df_plot['signal'], color='red', lw=1, label='Signal')
    ax3.bar(x, df_plot['histogram'], color='gray', alpha=0.3, width=0.8)
    ax3.set_ylabel('MACD', color='blue')
    ax.set_title(titulo, color='white', fontsize=14)
    ax.set_xlabel('Días', color='white')
    ax.set_ylabel('Precio (USDT)', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
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

# ================= ANÁLISIS CON IA (solo 1 vez al día) =================
def analizar_con_gemini_diario(img, symbol):
    try:
        img_b64 = pil_to_base64(img)
        prompt = f"""
        Eres un analista experto en Dual Asset de Bybit.
        Analiza el gráfico DIARIO de {symbol} con velas diarias, EMAs, RSI, MACD y niveles de soporte/resistencia.
        El Dual Asset de 1 día consiste en elegir entre:
        - **Buy Low**: apostar a que el precio bajará en las próximas 24h, para comprar barato.
        - **Sell High**: apostar a que el precio subirá en las próximas 24h, para vender caro.
        Basándote en la tendencia (alcista/bajista/lateral), la posición del RSI (sobrecompra/sobreventa), el MACD y los niveles clave, decide cuál es la mejor opción para las próximas 24h.
        Si el mercado está muy lateral o sin señales claras, recomienda Hold.
        Responde SOLO con un JSON en UNA línea:
        {{"decision": "Buy Low"/"Sell High"/"Hold", "razon": "explicación breve", "explicacion": "análisis detallado", "confianza": 0-100}}
        """
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_b64}}]}],
            temperature=0.3,
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
        logger.error(f"Error en IA para {symbol}: {e}")
        return "Hold", f"Error: {e}", "", 0

# ================= WEBSOCKET DE OFERTAS DUAL ASSET =================
def on_message(ws, message):
    global ws_offers
    try:
        data = json.loads(message)
        if "data" in data:
            for product in data["data"]:
                pid = product.get("p")
                if pid:
                    ws_offers[pid] = product
    except Exception as e:
        logger.error(f"WebSocket message error: {e}")

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.warning("WebSocket closed, reconectando...")
    time.sleep(5)
    start_websocket()

def on_open(ws):
    logger.info("WebSocket conectado")
    ws.send(json.dumps({"op": "subscribe", "args": ["earn.dualassets.offers"]}))

def start_websocket():
    websocket.enableTrace(False)
    ws_url = "wss://stream.bybit.com/v5/public/fp"
    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()
    return ws

def obtener_mejor_oferta_dual_asset(symbol_base, decision):
    """Busca en ws_offers la mejor oferta (mayor APY) para el símbolo y dirección (Buy Low o Sell High)."""
    global ws_offers
    target_key = "b" if decision == "Buy Low" else "s"
    mejores = []
    for pid, product in ws_offers.items():
        # Verificar si el producto corresponde al símbolo base (ej. "BTC" o "SOL")
        if symbol_base.upper() not in pid.upper():
            continue
        if target_key in product:
            for quote in product[target_key]:
                apy = int(quote.get("a", 0)) / 1e8
                max_amount = float(quote.get("m", 0))
                select_price = float(quote.get("s", 0))
                expire_time = int(quote.get("x", 0))
                # Verificar que el producto sea de 1 día (24h)
                # Según la estructura, podría haber un campo "d" de duración
                duration = product.get("d", 0)
                if duration == 86400 or duration == 1:  # 1 día en segundos o número
                    mejores.append({
                        "productId": pid,
                        "apy": apy,
                        "maxAmount": max_amount,
                        "selectPrice": select_price,
                        "expireTime": expire_time
                    })
    if not mejores:
        return None
    mejor = max(mejores, key=lambda x: x["apy"])
    return mejor

# ================= OBTENER QUOTE FIJO POR API =================
def obtener_quote_fijo(product_id):
    """Obtiene cotización fija (leverage, precio actual, monto máximo) para un producto."""
    try:
        # Endpoint real: GET /v5/earn/advance/product-extra-info?category=DoubleWin&productId=xxx
        timestamp = int(time.time() * 1000)
        recv_window = "5000"
        query_string = f"category=DoubleWin&productId={product_id}"
        payload = str(timestamp) + BYBIT_API_KEY + recv_window + query_string
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        url = f"https://api.bybit.com/v5/earn/advance/product-extra-info?{query_string}"
        response = requests.get(url, headers=headers, timeout=15)
        result = response.json()
        if result.get("retCode") == 0:
            quote = result["result"]
            return {
                "leverage": quote.get("leverage", 1),
                "maxInvestmentAmount": float(quote.get("maxInvestmentAmount", 0)),
                "currentPrice": float(quote.get("currentPrice", 0))
            }
        else:
            logger.error(f"Error obteniendo quote: {result}")
            return None
    except Exception as e:
        logger.error(f"Excepción obteniendo quote: {e}")
        return None

# ================= SUSCRIPCIÓN DUAL ASSET =================
def suscribir_dual_asset(product_id, amount_usdt, decision, quote_info, order_link_id=None):
    if order_link_id is None:
        order_link_id = f"duplo_{int(time.time())}"
    # Determinar moneda base: para Buy Low se usa USDT, para Sell High la moneda del activo (ej. BTC)
    coin = "USDT" if decision == "Buy Low" else "BTC"  # Ajustar según símbolo real (podríamos extraer de product_id)
    body = {
        "category": "DoubleWin",
        "productId": product_id,
        "orderType": "Stake",
        "amount": str(amount_usdt),
        "accountType": "UNIFIED",  # o "FUND"
        "coin": coin,
        "orderLinkId": order_link_id,
        "doubleWinStakeExtra": {
            "leverage": quote_info["leverage"],
            "initialPrice": str(quote_info["currentPrice"])
        }
    }
    try:
        timestamp = int(time.time() * 1000)
        recv_window = "5000"
        body_json = json.dumps(body)
        payload = str(timestamp) + BYBIT_API_KEY + recv_window + body_json
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }
        url = "https://api.bybit.com/v5/earn/advance/place-order"
        response = requests.post(url, headers=headers, data=body_json, timeout=15)
        result = response.json()
        if result.get("retCode") == 0:
            logger.info(f"Suscripción exitosa a {product_id} por {amount_usdt} USDT")
            telegram_mensaje(f"✅ Suscripción Dual Asset: {product_id}, Monto: {amount_usdt} USDT, Decisión: {decision}")
            return True
        else:
            logger.error(f"Error en suscripción: {result}")
            telegram_mensaje(f"❌ Error suscripción: {result.get('retMsg')}")
            return False
    except Exception as e:
        logger.error(f"Excepción suscribiendo: {e}")
        telegram_mensaje(f"❌ Excepción suscribiendo: {e}")
        return False

# ================= SALDO Y POSICIONES =================
def obtener_saldo_usdt():
    try:
        resp = bybit_session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        for coin in resp["result"]["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0
    except Exception as e:
        logger.error(f"Error obteniendo saldo USDT: {e}")
        return 0.0

def obtener_posiciones_activas():
    """Devuelve lista de posiciones Dual Asset activas (no liquidadas)."""
    try:
        timestamp = int(time.time() * 1000)
        recv_window = "5000"
        query_string = "category=DoubleWin"
        payload = str(timestamp) + BYBIT_API_KEY + recv_window + query_string
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": str(timestamp),
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        url = f"https://api.bybit.com/v5/earn/advance/position?{query_string}"
        response = requests.get(url, headers=headers, timeout=15)
        result = response.json()
        if result.get("retCode") == 0:
            return result["result"]["list"]
        else:
            return []
    except Exception as e:
        logger.error(f"Error obteniendo posiciones: {e}")
        return []

def convertir_a_usdt(coin, amount):
    """Convierte una criptomoneda a USDT en spot."""
    try:
        symbol = f"{coin}USDT"
        ticker = bybit_session.get_ticker(category="spot", symbol=symbol)
        price = float(ticker["result"]["list"][0]["lastPrice"])
        # Orden de mercado para vender
        order = bybit_session.place_order(category="spot", symbol=symbol, side="Sell", orderType="Market", qty=str(amount))
        if order["retCode"] == 0:
            usdt_received = amount * price
            logger.info(f"Convertidos {amount} {coin} a {usdt_received:.2f} USDT")
            telegram_mensaje(f"💱 Conversión automática: {amount} {coin} -> {usdt_received:.2f} USDT")
        else:
            logger.error(f"Error convirtiendo {coin} a USDT: {order}")
    except Exception as e:
        logger.error(f"Excepción convirtiendo {coin} a USDT: {e}")

def verificar_y_liquidar_posiciones():
    """Revisa posiciones vencidas y convierte a USDT si es necesario."""
    posiciones = obtener_posiciones_activas()
    for pos in posiciones:
        # Si la posición está liquidada (status Settled) y no es USDT, convertir
        if pos.get("status") == "Settled" and pos.get("coin") != "USDT":
            amount = float(pos.get("amount", 0))
            coin = pos.get("coin")
            if amount > 0:
                convertir_a_usdt(coin, amount)
        # También podemos registrar cambios de saldo
    # Opcional: notificar saldo actual
    saldo = obtener_saldo_usdt()
    telegram_mensaje(f"💰 Saldo USDT actual: {saldo:.2f}")

# ================= ESTRATEGIA DIARIA (solo 1 vez con IA) =================
def estrategia_diaria_con_ia():
    global subscription_active, last_ia_decision
    if subscription_active:
        logger.info("Ya hay una suscripción activa hoy. No se ejecuta nueva estrategia.")
        return

    logger.info("=== Análisis diario con IA para Dual Asset ===")
    saldo_usdt = obtener_saldo_usdt()
    if saldo_usdt < 50:
        telegram_mensaje("⚠️ Saldo insuficiente (<50 USDT). No se invertirá hoy.")
        return

    monto_invertir = saldo_usdt * MAX_INVEST_PERCENT
    logger.info(f"Saldo USDT: {saldo_usdt:.2f}. Monto máximo a invertir: {monto_invertir:.2f}")

    mejores_opciones = []
    for symbol in SYMBOLS:
        base = symbol.replace("USDT", "")
        df = obtener_velas_diarias(symbol, limit=200)
        if df.empty:
            continue
        df = calcular_indicadores_diarios(df)
        sop, res, slope, intercept, tend, micro = detectar_zonas_mercado_diario(df)
        img = generar_grafico_diario(df, f"Dual Asset - {symbol} (Diario)", sop, res, slope, intercept)
        if img:
            img.save(f"/tmp/duplo_{symbol}.png")
            telegram_enviar_imagen(f"/tmp/duplo_{symbol}.png", caption=f"📊 Análisis diario {symbol}")
            decision, razon, explicacion, confianza = analizar_con_gemini_diario(img, symbol)
        else:
            decision, razon, explicacion, confianza = "Hold", "Sin gráfico", "", 0

        logger.info(f"{symbol}: IA recomienda {decision} (confianza {confianza}) - {razon}")
        telegram_mensaje(f"🤖 Análisis {symbol}: {decision} (confianza {confianza}%)\n📝 {razon}")
        last_ia_decision[symbol] = decision  # Guardar para reintentos

        if decision != "Hold" and confianza >= 60:
            # Esperar a que WebSocket tenga datos (ya debería estar conectado)
            time.sleep(2)
            oferta = obtener_mejor_oferta_dual_asset(base, decision)
            if oferta:
                quote = obtener_quote_fijo(oferta["productId"])
                if quote:
                    monto_final = min(monto_invertir, oferta["maxAmount"], quote["maxInvestmentAmount"])
                    if monto_final >= 10:
                        mejores_opciones.append({
                            "symbol": symbol,
                            "decision": decision,
                            "apy": oferta["apy"],
                            "productId": oferta["productId"],
                            "targetPrice": oferta["selectPrice"],
                            "quote": quote,
                            "amount": monto_final
                        })
                    else:
                        logger.warning(f"Monto {monto_final} muy pequeño para {oferta['productId']}")
                else:
                    logger.warning(f"No se obtuvo quote para {oferta['productId']}")
            else:
                logger.warning(f"No hay oferta para {base} con decisión {decision}")

    if mejores_opciones:
        # Elegir la de mayor APY
        mejor = max(mejores_opciones, key=lambda x: x["apy"])
        if mejor["apy"] > 0:
            exito = suscribir_dual_asset(mejor["productId"], mejor["amount"], mejor["decision"], mejor["quote"])
            if exito:
                subscription_active = True
                # Programar desactivación después de 24h (para poder reinvertir al día siguiente)
                threading.Timer(86400, lambda: globals().update(subscription_active=False)).start()
            else:
                # Si falla, reintentar más tarde sin IA (usando la misma decisión guardada)
                logger.info("La suscripción falló, se reintentará más tarde con la misma decisión.")
        else:
            telegram_mensaje("APY 0, no se invierte.")
    else:
        telegram_mensaje("No se encontraron oportunidades claras hoy.")

# ================= REINTENTOS SIN IA (usando última decisión) =================
def reintentar_suscripcion_sin_ia():
    global subscription_active
    if subscription_active:
        return
    # Verificar si hay una decisión guardada de hoy
    if not last_ia_decision:
        return
    saldo_usdt = obtener_saldo_usdt()
    if saldo_usdt < 50:
        return
    monto_invertir = saldo_usdt * MAX_INVEST_PERCENT
    for symbol, decision in last_ia_decision.items():
        if decision == "Hold":
            continue
        base = symbol.replace("USDT", "")
        oferta = obtener_mejor_oferta_dual_asset(base, decision)
        if oferta:
            quote = obtener_quote_fijo(oferta["productId"])
            if quote:
                monto_final = min(monto_invertir, oferta["maxAmount"], quote["maxInvestmentAmount"])
                if monto_final >= 10:
                    exito = suscribir_dual_asset(oferta["productId"], monto_final, decision, quote)
                    if exito:
                        subscription_active = True
                        threading.Timer(86400, lambda: globals().update(subscription_active=False)).start()
                        break  # Solo una suscripción por día
    # Si no se pudo, se esperará al próximo ciclo diario con IA

# ================= KEEP ALIVE PARA RAILWAY =================
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
    logger.info("🚀 Bot Dual Asset iniciado (análisis diario con velas diarias)")
    telegram_mensaje("🚀 Bot Dual Asset en funcionamiento - Estrategia diaria con IA")
    start_websocket()
    keep_alive()
    # Programar análisis diario a las 7 AM UTC
    schedule.every().day.at(f"{DAILY_ANALYSIS_HOUR:02d}:00").do(estrategia_diaria_con_ia)
    # Reintentos cada 2 horas (sin IA) por si falló la suscripción
    schedule.every(2).hours.do(reintentar_suscripcion_sin_ia)
    # Verificar liquidaciones cada 6 horas
    schedule.every(6).hours.do(verificar_y_liquidar_posiciones)
    # Ejecutar una vez al inicio
    estrategia_diaria_con_ia()
    verificar_y_liquidar_posiciones()
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
