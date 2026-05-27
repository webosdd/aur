import os
import time
import asyncio
import json
import numpy as np
import pandas as pd
from scipy.stats import linregress
from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io
import base64
import re
import logging
import requests
import schedule
from openai import OpenAI
import hashlib
import hmac
from pybit.unified_trading import HTTP
import websocket
import threading

# ================= CONFIGURACIÓN LOGGING =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot_duplo.log"),
        logging.StreamHandler()
    ]
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
SYMBOLS = ["BTCUSDT", "SOLUSDT"]          # Pares que vamos a analizar
DURATION_DAYS = 1                         # Solo operaciones de 1 día
MAX_INVEST_PERCENT = 0.3                  # Invertir máximo el 30% del saldo
CONVERT_TO_USDT = True                    # Convertir automáticamente a USDT al liquidar

# Inicializar cliente OpenAI y Bybit
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Dual Asset Bot"})
bybit_session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# ================= FUNCIONES TELEGRAM =================
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
        logger.warning("Telegram no configurado para imágenes")
        return
    try:
        with open(ruta_imagen, 'rb') as foto:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            resp = requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption}, files={"photo": foto}, timeout=15)
        if resp.status_code != 200:
            logger.error(f"Error enviando imagen a Telegram: {resp.text[:200]}")
        else:
            logger.info("Imagen enviada a Telegram")
    except Exception as e:
        logger.error(f"Error enviando imagen: {e}")

# ================= FUNCIONES DE MERCADO =================
def obtener_velas(symbol="BTCUSDT", intervalo="30", limit=200):
    """Obtiene velas de 30 minutos para análisis de tendencia diaria."""
    try:
        resp = bybit_session.get_kline(category="spot", symbol=symbol, interval=intervalo, limit=limit)
        if resp.get("retCode") != 0:
            return pd.DataFrame()
        lista = resp["result"]["list"]
        df = pd.DataFrame(lista, columns=['time','open','high','low','close','volume','turnover'])
        for col in ['open','high','low','close','volume']:
            df[col] = df[col].astype(float)
        df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
        df.set_index('time', inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error obteniendo velas de {symbol}: {e}")
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
    return df.dropna()

def detectar_zonas_mercado(df):
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

def generar_grafico_vision(df, titulo, soporte, resistencia, slope, intercept):
    if df.empty:
        return None
    df_plot = df.tail(120).copy()
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
    if slope != 0:
        x_trend = np.array([0, len(df_plot)-1])
        y_trend = intercept + slope * x_trend
        ax.plot(x_trend, y_trend, color='white', linestyle='-.', lw=2, label='Tendencia', alpha=0.7)
    # Graficar RSI
    ax2 = ax.twinx()
    ax2.plot(x, df_plot['rsi'], color='orange', lw=1.5, alpha=0.7, label='RSI')
    ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
    ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('RSI', color='orange')
    # Graficar MACD
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(x, df_plot['macd'], color='blue', lw=1, label='MACD')
    ax3.plot(x, df_plot['signal'], color='red', lw=1, label='Signal')
    ax3.bar(x, df_plot['histogram'], color='gray', alpha=0.3, width=0.8)
    ax3.set_ylabel('MACD', color='blue')
    ax.set_title(sanitize_for_matplotlib(titulo), color='white', fontsize=14)
    ax.set_xlabel('Tiempo (velas de 30m)', color='white')
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

def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}"

# ================= ANÁLISIS CON IA =================
def analizar_con_gemini(img, symbol):
    try:
        img_b64 = pil_to_base64(img)
        prompt = f"""
        Eres un analista experto en Dual Asset de Bybit. Revisa el gráfico de {symbol} (velas de 30m)
        con EMAs, RSI, MACD y niveles de soporte/resistencia.
        Decide si la mejor opción es **Buy Low** (apostar a que el precio baje para comprar barato) o **Sell High** (apostar a que suba para vender caro).
        **Reglas estrictas:**
        - **Buy Low**: Se recomienda cuando el RSI está cerca de sobreventa (30-40), hay soportes claros y velas muestran rebote.
        - **Sell High**: Se recomienda cuando el RSI está cerca de sobrecompra (60-70), hay resistencias claras y velas muestran rechazo.
        - Si está lateral y sin señales claras, recomienda no invertir.
        Responde SOLO con un JSON en UNA línea: {{"decision": "Buy Low"/"Sell High"/"Hold", "razon": "explicación breve", "explicacion": "análisis detallado", "confianza": 0-100}}
        """
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_b64}}]}],
            temperature=0.3,
            timeout=60
        )
        contenido = response.choices[0].message.content
        try:
            datos = json.loads(contenido)
        except:
            import json_repair
            datos = json_repair.loads(contenido)
        return datos.get("decision"), datos.get("razon"), datos.get("explicacion"), datos.get("confianza", 0)
    except Exception as e:
        logger.error(f"Error en IA para {symbol}: {e}")
        return "Hold", f"Error: {e}", "", 0

# ================= DUAL ASSET: OFERTAS Y SUSCRIPCIÓN =================
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

def obtener_ofertas_dual_asset(symbol_base):
    """Obtiene las ofertas activas de Dual Asset para un símbolo mediante WebSocket (simulado por REST en este ejemplo)."""
    try:
        # Nota: Para obtener ofertas en tiempo real, el método recomendado es WebSocket.
        # Como simplificación, haremos una llamada REST simulada.
        # Reemplazar con la llamada real a /v5/earn/advance/product-quote
        # y suscribirse a earn.dualassets.offers vía WS.
        resp = bybit_session.get_dual_asset_offers()  # Método hipotético si existiera
        if resp.get("retCode") != 0:
            return []
        ofertas = []
        for product in resp["result"]["list"]:
            if product["symbol"] == symbol_base:
                ofertas.append({
                    "productId": product["productId"],
                    "targetPrice": product["targetPrice"],
                    "apy": product["apy"],
                    "expireTime": product["expireTime"]
                })
        return ofertas
    except Exception as e:
        logger.error(f"Error obteniendo ofertas Dual Asset para {symbol_base}: {e}")
        return []

def suscribir_dual_asset(product_id, amount_usdt, decision):
    """Suscribe a un producto Dual Asset (Buy Low o Sell High)."""
    try:
        # Determinar coin a invertir según decisión
        coin = "USDT" if decision == "Buy Low" else "BTC"  # Según corresponda
        order_link_id = f"duplo_{int(time.time())}"
        # Obtener el leverage y initialPrice de la cotización del producto
        quote = bybit_session.get_fixed_product_quote(productId=product_id)  # Método hipotético
        initial_price = quote["result"]["currentPrice"]
        leverage = quote["result"]["leverage"]
        body = {
            "category": "DoubleWin",
            "productId": product_id,
            "orderType": "Stake",
            "amount": str(amount_usdt),
            "accountType": "FUND",
            "coin": coin,
            "orderLinkId": order_link_id,
            "doubleWinStakeExtra": {
                "leverage": leverage,
                "initialPrice": initial_price
            }
        }
        resp = bybit_session.place_order_advance_earn(body)  # Endpoint POST /v5/earn/advance/place-order
        if resp.get("retCode") == 0:
            logger.info(f"Suscripción exitosa a {product_id} por {amount_usdt} USDT")
            telegram_mensaje(f"✅ Suscripción Dual Asset: Producto {product_id}, Monto: {amount_usdt} USDT, Decisión: {decision}")
            return True
        else:
            logger.error(f"Error en suscripción: {resp}")
            telegram_mensaje(f"❌ Error en suscripción Dual Asset: {resp.get('retMsg')}")
            return False
    except Exception as e:
        logger.error(f"Excepción suscribiendo Dual Asset: {e}")
        return False

# ================= LIQUIDACIÓN Y CONVERSIÓN =================
def verificar_liquidaciones():
    """Revisa productos vencidos, recoge fondos y convierte a USDT si se necesita."""
    try:
        resp = bybit_session.get_dual_asset_positions()  # Endpoint hipotético /v5/earn/advance/position
        if resp.get("retCode") != 0:
            return
        for pos in resp["result"]["list"]:
            if pos["status"] == "Settled":  # Liquidado
                if pos["coin"] != "USDT" and CONVERT_TO_USDT:
                    convertir_a_usdt(pos["coin"], float(pos["amount"]))
                telegram_mensaje(f"🔄 Producto Dual Asset liquidado: {pos['productId']}. Monto recibido: {pos['amount']} {pos['coin']}")
    except Exception as e:
        logger.error(f"Error verificando liquidaciones: {e}")

def convertir_a_usdt(coin, amount):
    """Convierte una criptomoneda a USDT en el mercado spot."""
    try:
        symbol = f"{coin}USDT"
        # Obtener precio actual
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

# ================= ESTRATEGIA PRINCIPAL =================
def estrategia_diaria():
    logger.info("Iniciando análisis diario para Dual Asset")
    saldo_usdt = obtener_saldo_usdt()
    if saldo_usdt < 50:
        telegram_mensaje("⚠️ Saldo insuficiente para Dual Asset (<50 USDT). No se invertirá hoy.")
        return
    monto_invertir = saldo_usdt * MAX_INVEST_PERCENT
    mejores_opciones = []
    for symbol in SYMBOLS:
        df = obtener_velas(symbol, intervalo="30", limit=200)
        if df.empty:
            continue
        df = calcular_indicadores(df)
        sop, res, slope, intercept, tend, micro = detectar_zonas_mercado(df)
        img = generar_grafico_vision(df, f"Dual Asset - {symbol}", sop, res, slope, intercept)
        if img:
            img.save(f"/tmp/duplo_{symbol}.png")
            telegram_enviar_imagen(f"/tmp/duplo_{symbol}.png", caption=f"📊 Análisis {symbol}")
            decision, razon, explicacion, confianza = analizar_con_gemini(img, symbol)
        else:
            decision, razon, explicacion, confianza = "Hold", "Sin gráfico", "", 0
        logger.info(f"{symbol}: IA recomienda {decision} (confianza {confianza}) - {razon}")
        telegram_mensaje(f"🤖 Análisis {symbol}: {decision} (confianza {confianza}%)\n📝 {razon}")
        if decision != "Hold" and confianza >= 60:
            ofertas = obtener_ofertas_dual_asset(symbol.split('USDT')[0])
            if ofertas:
                mejor_oferta = max(ofertas, key=lambda x: x["apy"])
                mejores_opciones.append({
                    "symbol": symbol,
                    "decision": decision,
                    "apy": mejor_oferta["apy"],
                    "productId": mejor_oferta["productId"],
                    "targetPrice": mejor_oferta["targetPrice"]
                })
    if mejores_opciones:
        mejor = max(mejores_opciones, key=lambda x: x["apy"])
        if mejor["apy"] > 0:
            suscribir_dual_asset(mejor["productId"], monto_invertir, mejor["decision"])
        else:
            telegram_mensaje("No se encontraron productos Dual Asset con APY > 0 para invertir.")
    else:
        telegram_mensaje("No se encontraron oportunidades claras de inversión Dual Asset hoy.")

# ================= KEEP ALIVE PARA RAILWAY =================
def keep_alive():
    """Mantiene el bot activo en Railway haciendo una petición cada 10 minutos."""
    from threading import Timer
    def ping():
        try:
            requests.get("https://railway.app/health", timeout=5)
        except:
            pass
        Timer(600, ping).start()
    Timer(600, ping).start()

# ================= EJECUCIÓN PRINCIPAL =================
def main():
    logger.info("🚀 Bot Dual Asset iniciado")
    telegram_mensaje("🚀 Bot Dual Asset en funcionamiento (Bybit Earn Duplo)")
    keep_alive()
    # Programación diaria: análisis a las 7 AM UTC (hora del settlement)
    schedule.every().day.at("07:00").do(estrategia_diaria)
    # Verificar liquidaciones cada 3 horas
    schedule.every(3).hours.do(verificar_liquidaciones)
    # Ejecutar una vez al inicio
    estrategia_diaria()
    verificar_liquidaciones()
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
