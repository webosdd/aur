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
import traceback

# ================= CONFIGURACIÓN =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot_duplo.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Falta OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("Faltan BYBIT_API_KEY o BYBIT_API_SECRET")

MODELO_VISION = "google/gemini-3.1-flash-image-preview"
SYMBOLS = ["BTCUSDT", "SOLUSDT"]
MIN_INVEST_USDT = 20
HORA_ANALISIS = "07:00"   # UTC

PRODUCT_IDS = {
    "BTC": "134685",
    "SOL": "134711"
}

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Dual Asset Bot"})
bybit_session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# Estado en memoria
decision_diaria = {}        # {"BTC": "Buy Low", "SOL": "Sell High"}
ultimo_resumen = 0
ws_offers = {}
ws_connected = False

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

# ================= SALDO Y CONVERSIONES =================
def obtener_saldo_usdt_unificado():
    try:
        resp = bybit_session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if resp.get("retCode") != 0:
            logger.error(f"Error obteniendo saldo USDT: {resp}")
            return 0.0
        for coin in resp["result"]["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0
    except Exception as e:
        logger.error(f"Excepción obteniendo saldo USDT: {e}")
        traceback.print_exc()
        return 0.0

def convertir_activo_a_usdt(coin, amount):
    try:
        ticker = bybit_session.get_ticker(category="spot", symbol=f"{coin}USDT")
        if ticker.get("retCode") != 0:
            logger.error(f"Error obteniendo precio {coin}: {ticker}")
            return 0.0
        price = float(ticker["result"]["list"][0]["lastPrice"])
        order = bybit_session.place_order(
            category="spot",
            symbol=f"{coin}USDT",
            side="Sell",
            orderType="Market",
            qty=str(amount)
        )
        if order["retCode"] == 0:
            usdt_received = amount * price
            logger.info(f"Convertido {amount} {coin} -> {usdt_received:.2f} USDT")
            telegram_mensaje(f"💱 Conversión automática: {amount} {coin} -> {usdt_received:.2f} USDT")
            return usdt_received
        else:
            logger.error(f"Error vendiendo {coin}: {order}")
            return 0.0
    except Exception as e:
        logger.error(f"Excepción convirtiendo {coin}: {e}")
        traceback.print_exc()
        return 0.0

def obtener_saldo_total_disponible():
    total = obtener_saldo_usdt_unificado()
    logger.info(f"Saldo USDT inicial: {total:.2f}")
    try:
        resp = bybit_session.get_wallet_balance(accountType="UNIFIED")
        if resp.get("retCode") != 0:
            logger.error(f"Error obteniendo todos los activos: {resp}")
            return total
        for coin_data in resp["result"]["list"][0]["coin"]:
            coin = coin_data["coin"]
            if coin != "USDT":
                monto = float(coin_data["walletBalance"])
                if monto > 0.001:
                    logger.info(f"Detectado {monto} {coin}, convirtiendo...")
                    total += convertir_activo_a_usdt(coin, monto)
                    time.sleep(1)
    except Exception as e:
        logger.error(f"Error obteniendo otros activos: {e}")
        traceback.print_exc()
    logger.info(f"Saldo total disponible: {total:.2f} USDT")
    return total

# ================= VELAS DIARIAS Y GRÁFICOS =================
def obtener_velas_diarias(symbol, limit=100):
    try:
        resp = bybit_session.get_kline(category="spot", symbol=symbol, interval="D", limit=limit)
        if resp.get("retCode") != 0:
            return pd.DataFrame()
        lista = resp["result"]["list"][::-1]
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
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    exp12 = df['close'].ewm(span=12, adjust=False).mean()
    exp26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['histogram'] = df['macd'] - df['signal']
    y = df['close'].values[-60:]
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
    ax2 = ax.twinx()
    ax2.plot(x, df_plot['rsi'], color='orange', lw=1.5, alpha=0.7, label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI', color='orange')
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

# ================= IA (ANÁLISIS DIARIO) =================
def analizar_con_gemini(img, symbol):
    try:
        img_b64 = pil_to_base64(img)
        prompt = f"""
        Eres un analista experto en Dual Asset de Bybit.
        Revisa el gráfico DIARIO de {symbol} (velas diarias) con EMAs, RSI, MACD y niveles de soporte/resistencia.
        Tu tarea es predecir si el precio dentro de 1 día (mañana) será más alto o más bajo que el precio actual.
        - Si crees que el precio será más alto mañana, recomienda **Sell High**.
        - Si crees que el precio será más bajo mañana, recomienda **Buy Low**.
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
        inicio = contenido.find('{')
        fin = contenido.rfind('}') + 1
        if inicio != -1 and fin != 0:
            contenido = contenido[inicio:fin]
        datos = json.loads(contenido)
        return datos.get("decision"), datos.get("razon"), datos.get("explicacion"), datos.get("confianza", 0)
    except Exception as e:
        logger.error(f"Error IA {symbol}: {e}")
        return "Hold", f"Error: {e}", "", 0

def ejecutar_analisis_diario():
    global decision_diaria
    logger.info("=== ANÁLISIS DIARIO CON IA ===")
    telegram_mensaje("📈 Iniciando análisis diario con IA...")
    for symbol in SYMBOLS:
        base = symbol.replace("USDT", "")
        df = obtener_velas_diarias(symbol, limit=100)
        if df.empty:
            continue
        df = calcular_indicadores_diarios(df)
        soporte, resistencia = detectar_soportes_resistencias_diario(df)
        slope = df['trend_slope'].iloc[-1] if 'trend_slope' in df else 0
        img = generar_grafico_diario(df, symbol, soporte, resistencia, slope, 0)
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
            decision_diaria[base] = decision
        else:
            decision_diaria[base] = None
    telegram_mensaje(f"📅 Decisión del día: BTC={decision_diaria.get('BTC')}, SOL={decision_diaria.get('SOL')}")

# ================= WEBSOCKET (ofertas) =================
def on_message(ws, message):
    global ws_offers
    try:
        data = json.loads(message)
        if "data" in data:
            for product in data["data"]:
                pid = str(product.get("p"))
                if pid in [PRODUCT_IDS["BTC"], PRODUCT_IDS["SOL"]]:
                    ws_offers[pid] = product
                    logger.info(f"Oferta actualizada para producto {pid}")
    except Exception as e:
        logger.error(f"Error en WS message: {e}")

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
    ws.send(json.dumps({"op": "subscribe", "args": ["earn.dualassets.offers"]}))
    ws_connected = True
    logger.info("WebSocket Dual Asset conectado y suscrito.")

def start_websocket():
    websocket.enableTrace(False)
    ws_url = "wss://stream.bybit.com/v5/public/fp"
    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    wst = threading.Thread(target=ws.run_forever, daemon=True)
    wst.start()
    return ws

def obtener_mejor_oferta(symbol_base, decision):
    target_key = "b" if decision == "Buy Low" else "s"
    product_id = PRODUCT_IDS.get(symbol_base)
    if not product_id:
        return None
    product = ws_offers.get(product_id)
    if not product or target_key not in product:
        logger.warning(f"No hay oferta para {symbol_base} con decisión {decision}")
        return None
    mejor = None
    mejor_apy = -1
    for quote in product[target_key]:
        apy = int(quote.get("a", 0)) / 1e8
        if apy > mejor_apy:
            mejor_apy = apy
            mejor = {
                "productId": product_id,
                "apy": apy,
                "maxAmount": float(quote.get("m", 0)),
                "selectPrice": float(quote.get("s", 0)),
                "expireTime": int(quote.get("x", 0))
            }
    return mejor

def obtener_quote_fijo(product_id):
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
            logger.error(f"Error en quote para {product_id}: {data}")
            return None
    except Exception as e:
        logger.error(f"Excepción quote: {e}")
        return None

def suscribir_dual_asset(product_id, amount_usdt, decision, quote_info):
    order_link_id = f"duplo_{int(time.time())}"
    coin = "USDT" if decision == "Buy Low" else "BTC" if "BTC" in product_id else "SOL"
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
            return True, order_link_id
        else:
            logger.error(f"Suscripción falló: {result}")
            telegram_mensaje(f"❌ Error suscripción {product_id}: {result.get('retMsg')}")
            return False, None
    except Exception as e:
        logger.error(f"Excepción suscripción: {e}")
        telegram_mensaje(f"❌ Excepción suscripción: {e}")
        return False, None

# ================= POSICIONES ACTIVAS (desde API) =================
def obtener_posiciones_activas():
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
        if data.get("retCode") != 0:
            return []
        activas = []
        for pos in data["result"]["list"]:
            if pos.get("status") == "Active":
                activas.append({
                    "productId": pos.get("productId"),
                    "amount": float(pos.get("amount", 0)),
                    "coin": pos.get("coin"),
                    "createTime": int(pos.get("createTime", 0)),
                    "expireTime": int(pos.get("expireTime", 0))
                })
        return activas
    except Exception as e:
        logger.error(f"Error obteniendo posiciones: {e}")
        return []

def reporte_posiciones():
    global ultimo_resumen
    ahora = time.time()
    if ahora - ultimo_resumen >= 10800:  # 3 horas
        posiciones = obtener_posiciones_activas()
        if not posiciones:
            telegram_mensaje("📊 No hay posiciones activas en este momento.")
        else:
            texto = f"📊 RESUMEN DE POSICIONES ACTIVAS ({len(posiciones)})\n"
            for pos in posiciones:
                venc = datetime.fromtimestamp(pos["expireTime"] / 1000).strftime("%H:%M UTC")
                texto += f"🔹 {pos['productId']} | {pos['amount']:.0f} {pos['coin']} | vence {venc}\n"
                tiempo_restante = (pos["expireTime"] / 1000) - ahora
                if 0 < tiempo_restante <= 3600:
                    telegram_mensaje(f"⚠️ POSICIÓN PRÓXIMA A VENCER\nProducto: {pos['productId']}\nVence en {tiempo_restante/60:.0f} minutos.")
            telegram_mensaje(texto)
        ultimo_resumen = ahora

# ================= CICLO HORARIO (CADA HORA) =================
def ciclo_horario():
    try:
        logger.info("🕒 Iniciando ciclo horario")
        telegram_mensaje("🔄 Ciclo horario: revisando saldo y ofertas...")
        
        # 1. Reporte de posiciones cada 3 horas
        reporte_posiciones()
        
        # 2. Obtener saldo total disponible
        saldo = obtener_saldo_total_disponible()
        telegram_mensaje(f"💰 Saldo total disponible: {saldo:.2f} USDT")
        
        if saldo < MIN_INVEST_USDT:
            logger.info(f"Saldo insuficiente ({saldo:.2f} USDT). No se invierte ahora.")
            return
        
        # 3. Intentar invertir para cada símbolo según decisión del día
        for symbol in SYMBOLS:
            base = symbol.replace("USDT", "")
            decision = decision_diaria.get(base)
            if not decision or decision == "Hold":
                logger.info(f"{base}: Sin decisión válida (Hold o None)")
                continue
            
            logger.info(f"Buscando oferta para {base} con decisión {decision}")
            oferta = obtener_mejor_oferta(base, decision)
            if not oferta:
                logger.warning(f"No hay oferta para {base} con dirección {decision}")
                continue
            
            quote = obtener_quote_fijo(oferta["productId"])
            if not quote:
                continue
            
            monto = min(saldo, oferta["maxAmount"], quote["maxInvestmentAmount"])
            if monto < MIN_INVEST_USDT:
                logger.info(f"Monto {monto} menor que mínimo {MIN_INVEST_USDT}")
                continue
            
            logger.info(f"Intentando suscribir {base} con {monto} USDT, APY {oferta['apy']}%")
            exito, order_id = suscribir_dual_asset(oferta["productId"], monto, decision, {**quote, "apy": oferta["apy"]})
            if exito:
                telegram_mensaje(
                    f"🟢 NUEVA INVERSIÓN DUAL ASSET\n"
                    f"Producto: {oferta['productId']}\n"
                    f"Dirección: {decision}\n"
                    f"Monto: {monto:.2f} USDT\n"
                    f"APY: {oferta['apy']}%\n"
                    f"Vence en 24 horas."
                )
                saldo -= monto
                if saldo < MIN_INVEST_USDT:
                    break
            else:
                logger.error(f"Fallo al suscribir {base}")
            time.sleep(2)
    except Exception as e:
        error_msg = f"Error en ciclo horario: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        telegram_mensaje(f"❌ Error en ciclo horario: {str(e)}")

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
    logger.info("🚀 Bot Dual Asset Avanzado (con logs y reportes cada hora) iniciado")
    telegram_mensaje("🚀 Bot Dual Asset activo - Revisará saldo cada hora y reportará posiciones.")
    
    start_websocket()
    keep_alive()
    time.sleep(5)  # esperar WebSocket
    
    # Programar análisis diario a las 07:00 UTC
    schedule.every().day.at(HORA_ANALISIS).do(ejecutar_analisis_diario)
    # Programar ciclo horario cada 60 minutos
    schedule.every(1).hours.do(ciclo_horario)
    
    # Ejecutar análisis inmediatamente si no hay decisión
    if not decision_diaria:
        ejecutar_analisis_diario()
    else:
        telegram_mensaje(f"📌 Decisión del día ya guardada: BTC={decision_diaria.get('BTC')}, SOL={decision_diaria.get('SOL')}")
    
    # Ejecutar ciclo horario inmediatamente después del análisis (para que actúe ya)
    ciclo_horario()
    
    # Bucle principal
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
