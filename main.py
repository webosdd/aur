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
SYMBOLS = ["BTC", "SOL"]   # Solo monedas base
MIN_INVEST_USDT = 20
MIN_APY_PERCENT = 180.0          # APY mínimo para invertir
HORA_ANALISIS = "07:00"          # UTC

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Dual Asset Bot"})
bybit_session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# Estado en memoria
decision_diaria = {}        # {"BTC": "Buy Low", "SOL": "Sell High"}
ultimo_resumen = 0
ultima_inversion_time = 0

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
            logger.error(f"Error saldo USDT: {resp}")
            return 0.0
        for coin in resp["result"]["list"][0]["coin"]:
            if coin["coin"] == "USDT":
                return float(coin["walletBalance"])
        return 0.0
    except Exception as e:
        logger.error(f"Excepción saldo USDT: {e}")
        return 0.0

def convertir_activo_a_usdt(coin, amount):
    try:
        symbol = f"{coin}USDT"
        tickers = bybit_session.get_tickers(category="spot", symbol=symbol)
        if tickers.get("retCode") != 0 or not tickers["result"]["list"]:
            logger.warning(f"No se puede convertir {coin}: par {symbol} no encontrado")
            return 0.0
        price = float(tickers["result"]["list"][0]["lastPrice"])
        order = bybit_session.place_order(
            category="spot",
            symbol=symbol,
            side="Sell",
            orderType="Market",
            qty=str(amount)
        )
        if order.get("retCode") == 0:
            usdt_received = amount * price
            logger.info(f"Convertido {amount} {coin} -> {usdt_received:.2f} USDT")
            telegram_mensaje(f"💱 Conversión: {amount} {coin} -> {usdt_received:.2f} USDT")
            return usdt_received
        else:
            logger.error(f"Error vendiendo {coin}: {order}")
            return 0.0
    except Exception as e:
        logger.error(f"Excepción convirtiendo {coin}: {e}")
        return 0.0

def obtener_saldo_total_disponible():
    total = obtener_saldo_usdt_unificado()
    logger.info(f"Saldo USDT inicial: {total:.2f}")
    activos_convertibles = ["BTC", "SOL", "ETH", "BNB", "XRP", "DOGE", "ADA", "TRX", "LINK", "MATIC", "AVAX", "UNI"]
    try:
        resp = bybit_session.get_wallet_balance(accountType="UNIFIED")
        if resp.get("retCode") != 0:
            logger.error(f"Error obteniendo otros activos: {resp}")
            return total
        for coin_data in resp["result"]["list"][0]["coin"]:
            coin = coin_data["coin"]
            if coin == "USDT":
                continue
            monto = float(coin_data["walletBalance"])
            if monto < 0.001:
                continue
            if coin in activos_convertibles:
                logger.info(f"Detectado {monto} {coin}, convirtiendo...")
                total += convertir_activo_a_usdt(coin, monto)
                time.sleep(1)
            else:
                logger.info(f"Ignorando {coin} (no convertible)")
    except Exception as e:
        logger.error(f"Error procesando activos: {e}")
    logger.info(f"Saldo total disponible: {total:.2f} USDT")
    return total

# ================= VELAS DIARIAS Y GRÁFICOS =================
def obtener_velas_diarias(symbol, limit=100):
    try:
        resp = bybit_session.get_kline(category="spot", symbol=f"{symbol}USDT", interval="D", limit=limit)
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
    telegram_mensaje("📈 Iniciando análisis diario...")
    for base in SYMBOLS:
        df = obtener_velas_diarias(base, limit=100)
        if df.empty:
            continue
        df = calcular_indicadores_diarios(df)
        soporte, resistencia = detectar_soportes_resistencias_diario(df)
        slope = df['trend_slope'].iloc[-1] if 'trend_slope' in df else 0
        img = generar_grafico_diario(df, base, soporte, resistencia, slope, 0)
        if img:
            img_path = f"/tmp/duplo_{base}.png"
            img.save(img_path)
            telegram_enviar_imagen(img_path, caption=f"📊 Análisis Diario {base}")
            decision, razon, explicacion, confianza = analizar_con_gemini(img, base)
        else:
            decision, razon, confianza = "Hold", "Error gráfico", 0

        logger.info(f"{base}: IA -> {decision} (confianza {confianza}) - {razon}")
        telegram_mensaje(f"🤖 {base}: {decision} (confianza {confianza}%)\n📝 {razon}")

        if decision != "Hold" and confianza >= 60:
            decision_diaria[base] = decision
        else:
            decision_diaria[base] = None
    telegram_mensaje(f"📅 Decisión del día: BTC={decision_diaria.get('BTC')}, SOL={decision_diaria.get('SOL')}")

# ================= OBTENER PRODUCTOS DUAL ASSET DESDE REST =================
def obtener_productos_dual_asset(coin=None):
    """
    Obtiene todos los productos activos de Dual Asset Mining.
    Si coin es 'BTC' o 'SOL', filtra por esa moneda.
    Retorna lista de productos con sus APY y direcciones.
    """
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        # Endpoint para listar productos de Dual Asset (Mining)
        # Según documentación, es GET /v5/earn/dual-asset/product
        query = "status=active"
        if coin:
            query += f"&coin={coin}"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        url = f"https://api.bybit.com/v5/earn/dual-asset/product?{query}"
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("retCode") != 0:
            logger.error(f"Error obteniendo productos: {data}")
            return []
        productos = []
        for prod in data["result"]["list"]:
            # Extraer datos relevantes
            product_id = prod.get("productId")
            coin_name = prod.get("coin")  # Ej: "BTC"
            direction = prod.get("direction")  # "Buy Low" o "Sell High"
            apy_e8 = prod.get("apyE8", "0")
            apy = int(apy_e8) / 1e8
            min_amount = float(prod.get("minAmount", 0))
            max_amount = float(prod.get("maxAmount", 0))
            if apy >= MIN_APY_PERCENT and product_id and direction:
                productos.append({
                    "productId": product_id,
                    "coin": coin_name,
                    "direction": direction,
                    "apy": apy,
                    "minAmount": min_amount,
                    "maxAmount": max_amount
                })
        logger.info(f"Productos Dual Asset encontrados: {len(productos)}")
        return productos
    except Exception as e:
        logger.error(f"Excepción obteniendo productos: {e}")
        return []

def obtener_mejor_producto(coin, decision):
    """Devuelve el producto con mayor APY para la moneda y dirección dadas."""
    productos = obtener_productos_dual_asset(coin)
    filtrados = [p for p in productos if p["coin"] == coin and p["direction"] == decision and p["apy"] >= MIN_APY_PERCENT]
    if not filtrados:
        logger.warning(f"No hay productos para {coin} con dirección {decision} y APY >= {MIN_APY_PERCENT}%")
        return None
    mejor = max(filtrados, key=lambda x: x["apy"])
    logger.info(f"Mejor producto para {coin} {decision}: ID {mejor['productId']} APY {mejor['apy']}%")
    return mejor

# ================= OBTENER QUOTE FIJO =================
def obtener_quote_fijo_dual_asset(product_id):
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = f"productId={product_id}"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        url = f"https://api.bybit.com/v5/earn/dual-asset/product-extra-info?{query}"
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
    side = "Buy" if decision == "Buy Low" else "Sell"
    body = {
        "productId": product_id,
        "side": side,
        "orderType": "Stake",
        "amount": str(amount_usdt),
        "accountType": "UNIFIED",
        "orderLinkId": order_link_id,
        "dualAssetStakeExtra": {
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
        url = "https://api.bybit.com/v5/earn/dual-asset/place-order"
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
        return False, None

# ================= POSICIONES ACTIVAS Y REPORTES =================
def obtener_posiciones_activas():
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = ""  # No se necesitan parámetros adicionales
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        url = "https://api.bybit.com/v5/earn/dual-asset/position"
        resp = requests.get(url, headers=headers, timeout=10)
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
    if ahora - ultimo_resumen >= 10800:
        posiciones = obtener_posiciones_activas()
        if not posiciones:
            telegram_mensaje("📊 No hay posiciones activas en este momento.")
        else:
            texto = f"📊 POSICIONES ACTIVAS ({len(posiciones)})\n"
            for pos in posiciones:
                venc = datetime.fromtimestamp(pos["expireTime"] / 1000).strftime("%H:%M UTC")
                texto += f"🔹 {pos['productId']} | {pos['amount']:.0f} {pos['coin']} | vence {venc}\n"
                tiempo_restante = (pos["expireTime"] / 1000) - ahora
                if 0 < tiempo_restante <= 3600:
                    telegram_mensaje(f"⚠️ Vence pronto: {pos['productId']} en {tiempo_restante/60:.0f} min.")
            telegram_mensaje(texto)
        ultimo_resumen = ahora

# ================= CICLO HORARIO =================
def ciclo_horario():
    try:
        logger.info("🕒 EJECUTANDO CICLO HORARIO")
        telegram_mensaje("🔄 Ciclo horario: revisando saldo y ofertas...")
        
        reporte_posiciones()
        saldo = obtener_saldo_total_disponible()
        telegram_mensaje(f"💰 Saldo disponible: {saldo:.2f} USDT")
        
        if saldo < MIN_INVEST_USDT:
            logger.info(f"Saldo insuficiente ({saldo:.2f} < {MIN_INVEST_USDT})")
            telegram_mensaje(f"⚠️ Saldo insuficiente. Mínimo: {MIN_INVEST_USDT} USDT")
            return
        
        for base in SYMBOLS:
            decision = decision_diaria.get(base)
            if not decision or decision == "Hold":
                logger.info(f"{base}: decisión Hold, no se invierte")
                continue
            
            # Obtener el mejor producto para esta moneda y dirección
            mejor_producto = obtener_mejor_producto(base, decision)
            if not mejor_producto:
                logger.warning(f"No hay producto adecuado para {base} {decision}")
                continue
            
            product_id = mejor_producto["productId"]
            apy = mejor_producto["apy"]
            max_amount = mejor_producto["maxAmount"]
            
            # Obtener cotización fija
            quote = obtener_quote_fijo_dual_asset(product_id)
            if not quote:
                logger.error(f"No se pudo obtener quote para {product_id}")
                continue
            
            monto = min(saldo, max_amount, quote["maxInvestmentAmount"])
            if monto < MIN_INVEST_USDT:
                logger.info(f"Monto {monto} menor al mínimo")
                continue
            
            logger.info(f"INVIRTIENDO: {base} - {decision} - {monto} USDT - APY {apy}%")
            exito, order_id = suscribir_dual_asset(product_id, monto, decision, {**quote, "apy": apy})
            if exito:
                telegram_mensaje(
                    f"🟢 NUEVA INVERSIÓN\n"
                    f"Producto: {product_id}\n"
                    f"Dirección: {decision}\n"
                    f"Monto: {monto:.2f} USDT\n"
                    f"APY: {apy}%\n"
                    f"Vence en 24h."
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
        telegram_mensaje(f"❌ Error en ciclo: {str(e)}")

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
    logger.info("🚀 Bot Dual Asset Mining (REST) iniciado")
    telegram_mensaje("🚀 Bot Dual Asset Mining activo - APY mínimo 180%, revisión cada hora")
    
    keep_alive()
    time.sleep(2)
    
    # Programar análisis diario a las 07:00 UTC
    schedule.every().day.at(HORA_ANALISIS).do(ejecutar_analisis_diario)
    # Programar ciclo horario cada 60 minutos
    schedule.every(1).hours.do(ciclo_horario)
    
    # Ejecutar análisis inmediato si no hay decisión
    if not decision_diaria:
        ejecutar_analisis_diario()
    else:
        telegram_mensaje(f"📌 Decisión actual: BTC={decision_diaria.get('BTC')}, SOL={decision_diaria.get('SOL')}")
    
    # Ejecutar ciclo horario inmediatamente
    ciclo_horario()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
