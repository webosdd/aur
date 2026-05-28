import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import schedule
import threading
import hmac
import hashlib
import traceback
from openai import OpenAI
from pybit.unified_trading import HTTP

# ================= CONFIGURACIÓN =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("bot_grid.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Variables de entorno
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Falta OPENROUTER_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise ValueError("Faltan BYBIT_API_KEY o BYBIT_API_SECRET")

SYMBOL = "BTCUSDT"
FIXED_INVEST_USDT = 10          # Monto fijo para el grid
GRID_COUNT = 20                 # Niveles del grid
KILL_SWITCH_MARGIN = 0.02       # 2% fuera del rango

# Cliente OpenRouter (IA)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Grid Bot"})

# Cliente Bybit
bybit = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# Estado en memoria (sin archivos)
grid_activo = {
    "orderId": None,
    "soporte": None,
    "resistencia": None,
    "createdAt": None
}

# ================= TELEGRAM =================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                      data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ================= OBTENER VELAS DIARIAS E INDICADORES =================
def obtener_velas_diarias(symbol, limit=100):
    try:
        resp = bybit.get_kline(category="spot", symbol=symbol, interval="D", limit=limit)
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
        logger.error(f"Error obteniendo velas: {e}")
        return pd.DataFrame()

def calcular_indicadores(df):
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
    return df.dropna()

def soporte_resistencia_historicos(df):
    """Soporte = mínimo de últimos 20 días, resistencia = máximo de últimos 20 días"""
    if len(df) < 20:
        return 0, 0
    soporte = df['low'].rolling(20).min().iloc[-1]
    resistencia = df['high'].rolling(20).max().iloc[-1]
    return soporte, resistencia

# ================= IA: OBTENER SOPORTE Y RESISTENCIA =================
def analizar_con_ia(df, soporte_hist, resistencia_hist):
    """Llama a Gemini y devuelve (soporte, resistencia, tendencia, confianza)"""
    # Preparar resumen de indicadores para la IA
    ultimo = df.iloc[-1]
    resumen = f"""
    Precio actual: {ultimo['close']:.2f}
    EMA20: {ultimo['ema20']:.2f}
    EMA50: {ultimo['ema50']:.2f}
    RSI: {ultimo['rsi']:.2f}
    MACD: {ultimo['macd']:.5f}
    Señal MACD: {ultimo['signal']:.5f}
    Soporte histórico (20d): {soporte_hist:.2f}
    Resistencia histórica (20d): {resistencia_hist:.2f}
    """
    prompt = f"""
    Eres un analista técnico experto. Basado en los siguientes datos del par BTCUSDT en gráfico diario, determina niveles de soporte y resistencia para las próximas 5 horas.
    Además indica la tendencia (Bullish/Bearish/Neutral) y una confianza del 0 al 100.

    Datos:
    {resumen}

    Reglas:
    - Soporte debe ser un precio por debajo del actual donde haya alta probabilidad de rebote (cerca del histórico o zona de RSI sobrecomprado/sobrevendido).
    - Resistencia debe ser un precio por encima del actual.
    - Si la tendencia es alcista, la resistencia puede estar más lejos; si bajista, el soporte más profundo.
    - Responde ÚNICAMENTE con un JSON válido en UNA línea como este:
    {{"soporte": 45000.0, "resistencia": 52000.0, "tendencia": "Bullish", "confianza": 85}}
    """
    try:
        response = client.chat.completions.create(
            model="google/gemini-3.1-flash-image-preview",  # modelo de visión, pero usamos texto
            messages=[{"role": "user", "content": prompt}],
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
        soporte = float(datos.get("soporte", soporte_hist))
        resistencia = float(datos.get("resistencia", resistencia_hist))
        tendencia = datos.get("tendencia", "Neutral")
        confianza = int(datos.get("confianza", 50))
        # Validar que soporte < resistencia y ambos positivos
        if soporte >= resistencia or soporte <= 0 or resistencia <= 0:
            soporte, resistencia = soporte_hist, resistencia_hist
        return soporte, resistencia, tendencia, confianza
    except Exception as e:
        logger.error(f"Error en IA: {e}")
        return soporte_hist, resistencia_hist, "Neutral", 0

# ================= GESTIÓN DE GRID (API V5) =================
def crear_grid_spot(soporte, resistencia):
    """Crea un grid neutral con 10 USDT y 20 niveles"""
    try:
        params = {
            "symbol": SYMBOL,
            "gridType": 1,          # Neutral grid
            "lowerPrice": str(soporte),
            "upperPrice": str(resistencia),
            "gridCount": GRID_COUNT,
            "totalInvestment": str(FIXED_INVEST_USDT)
        }
        # Usar el método genérico de pybit (si existe) o requests firmado
        # pybit no tiene método directo, usamos requests con firma
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = "&".join([f"{k}={v}" for k, v in params.items()])
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }
        # Nota: en la práctica el body debe ser JSON y la firma sobre el body string, no query.
        # Simplificamos: usamos POST con body JSON
        body = params
        body_str = json.dumps(body)
        payload = timestamp + BYBIT_API_KEY + recv_window + body_str
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers["X-BAPI-SIGN"] = signature
        url = "https://api.bybit.com/v5/spot-execution/grid-order"
        resp = requests.post(url, headers=headers, data=body_str, timeout=15)
        data = resp.json()
        if data.get("retCode") == 0:
            order_id = data["result"]["orderId"]
            logger.info(f"Grid creado exitosamente. OrderId: {order_id}")
            telegram_mensaje(f"🟢 Grid Neutral creado para {SYMBOL}\n"
                             f"Rango: {soporte} - {resistencia}\n"
                             f"Inversión: {FIXED_INVEST_USDT} USDT\n"
                             f"Niveles: {GRID_COUNT}\n"
                             f"OrderId: {order_id}")
            return order_id
        else:
            logger.error(f"Error creando grid: {data}")
            telegram_mensaje(f"❌ Error al crear grid: {data.get('retMsg')}")
            return None
    except Exception as e:
        logger.error(f"Excepción creando grid: {e}")
        telegram_mensaje(f"❌ Excepción al crear grid: {str(e)}")
        return None

def cancelar_grid(order_id):
    """Cancela un grid activo por su orderId"""
    try:
        body = {"orderId": order_id}
        body_str = json.dumps(body)
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        payload = timestamp + BYBIT_API_KEY + recv_window + body_str
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }
        url = "https://api.bybit.com/v5/spot-execution/cancel-grid-order"
        resp = requests.post(url, headers=headers, data=body_str, timeout=15)
        data = resp.json()
        if data.get("retCode") == 0:
            logger.info(f"Grid {order_id} cancelado exitosamente.")
            telegram_mensaje(f"⚠️ Grid {order_id} cancelado (kill-switch activado).")
            return True
        else:
            logger.error(f"Error cancelando grid: {data}")
            return False
    except Exception as e:
        logger.error(f"Excepción cancelando grid: {e}")
        return False

def obtener_detalle_grid(order_id):
    """Obtiene estado, PnL, niveles del grid"""
    try:
        url = f"https://api.bybit.com/v5/spot-execution/grid-order-details?orderId={order_id}"
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = f"orderId={order_id}"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            return data["result"]
        else:
            logger.error(f"Error obteniendo detalle grid: {data}")
            return None
    except Exception as e:
        logger.error(f"Excepción detalle grid: {e}")
        return None

def listar_grids_activos():
    """Consulta grids activos para BTCUSDT. Retorna el primer orderId encontrado o None"""
    try:
        url = f"https://api.bybit.com/v5/spot-execution/grid-orders?symbol={SYMBOL}&limit=10&orderStatus=active"
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = f"symbol={SYMBOL}&limit=10&orderStatus=active"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0 and data["result"]["list"]:
            # Tomamos el primero activo
            grid_info = data["result"]["list"][0]
            return grid_info["orderId"], float(grid_info["lowerPrice"]), float(grid_info["upperPrice"])
        else:
            return None, None, None
    except Exception as e:
        logger.error(f"Error listando grids: {e}")
        return None, None, None

def obtener_precio_actual():
    """Obtiene el precio de BTCUSDT desde ticker"""
    try:
        ticker = bybit.get_tickers(category="spot", symbol=SYMBOL)
        if ticker.get("retCode") == 0:
            return float(ticker["result"]["list"][0]["lastPrice"])
        return None
    except Exception as e:
        logger.error(f"Error obteniendo precio: {e}")
        return None

def obtener_saldo_usdt():
    try:
        bal = bybit.get_wallet_balance(accountType="UNIFIED", coin="USDT")
        if bal.get("retCode") == 0:
            for coin in bal["result"]["list"][0]["coin"]:
                if coin["coin"] == "USDT":
                    return float(coin["walletBalance"])
        return 0.0
    except Exception as e:
        logger.error(f"Error saldo USDT: {e}")
        return 0.0

def vender_todo_btc():
    """Vende todo el saldo de BTC a USDT a mercado"""
    try:
        # Obtener saldo BTC
        bal = bybit.get_wallet_balance(accountType="UNIFIED", coin="BTC")
        if bal.get("retCode") != 0:
            return
        btc_amount = 0.0
        for coin in bal["result"]["list"][0]["coin"]:
            if coin["coin"] == "BTC":
                btc_amount = float(coin["walletBalance"])
                break
        if btc_amount <= 0:
            return
        order = bybit.place_order(
            category="spot",
            symbol=SYMBOL,
            side="Sell",
            orderType="Market",
            qty=str(btc_amount)
        )
        if order.get("retCode") == 0:
            logger.info(f"Vendidos {btc_amount} BTC a mercado")
            telegram_mensaje(f"💰 Venta de emergencia: {btc_amount} BTC convertidos a USDT.")
        else:
            logger.error(f"Error vendiendo BTC: {order}")
    except Exception as e:
        logger.error(f"Excepción vendiendo BTC: {e}")

# ================= LÓGICA PRINCIPAL =================
def ejecutar_analisis_y_crear_grid():
    global grid_activo
    logger.info("=== ANÁLISIS CADA 5 HORAS ===")
    telegram_mensaje("📊 Iniciando análisis técnico para grid bot...")

    # Verificar si ya hay un grid activo (por si acaso)
    order_id, lower, upper = listar_grids_activos()
    if order_id:
        logger.info(f"Ya existe grid activo {order_id}, no se crea uno nuevo.")
        grid_activo["orderId"] = order_id
        grid_activo["soporte"] = lower
        grid_activo["resistencia"] = upper
        grid_activo["createdAt"] = datetime.now()
        return

    # Obtener datos históricos
    df = obtener_velas_diarias(SYMBOL, limit=100)
    if df.empty:
        logger.error("No se pudieron obtener velas")
        return
    df = calcular_indicadores(df)
    sop_hist, res_hist = soporte_resistencia_historicos(df)
    if sop_hist == 0 or res_hist == 0:
        logger.error("No se pudo calcular soporte/resistencia históricos")
        return

    # Llamar a IA
    soporte, resistencia, tendencia, confianza = analizar_con_ia(df, sop_hist, res_hist)
    logger.info(f"IA => Soporte: {soporte}, Resistencia: {resistencia}, Tendencia: {tendencia}, Confianza: {confianza}%")
    telegram_mensaje(f"🤖 IA: Soporte {soporte:.0f} / Resistencia {resistencia:.0f}\nTendencia {tendencia} (confianza {confianza}%)")

    # Crear grid
    if soporte < resistencia and soporte > 0 and resistencia > 0:
        nuevo_id = crear_grid_spot(soporte, resistencia)
        if nuevo_id:
            grid_activo["orderId"] = nuevo_id
            grid_activo["soporte"] = soporte
            grid_activo["resistencia"] = resistencia
            grid_activo["createdAt"] = datetime.now()
        else:
            logger.error("Fallo al crear grid")
    else:
        logger.error("Niveles inválidos, no se crea grid")

def monitorear_y_kill_switch():
    global grid_activo
    logger.info("=== MONITOREO HORARIO ===")
    if not grid_activo["orderId"]:
        # Intentar recuperar grid activo desde API
        oid, low, high = listar_grids_activos()
        if oid:
            grid_activo["orderId"] = oid
            grid_activo["soporte"] = low
            grid_activo["resistencia"] = high
            logger.info(f"Grid activo recuperado: {oid}")
        else:
            logger.info("No hay grid activo, omitiendo monitoreo.")
            return

    # Obtener estado del grid
    detalle = obtener_detalle_grid(grid_activo["orderId"])
    if not detalle:
        logger.warning("No se pudo obtener detalle del grid, puede que ya no exista.")
        # Si no existe, limpiar estado
        grid_activo["orderId"] = None
        return

    status = detalle.get("gridStatus")  # 1=activo, 2=cerrado, etc.
    pnl = float(detalle.get("currentPnL", 0))
    precio_actual = obtener_precio_actual()
    saldo_usdt = obtener_saldo_usdt()

    # Construir reporte
    msg = (
        f"📊 ESTADO GRID {SYMBOL}\n"
        f"Status: {'Activo' if status == 1 else 'Inactivo'}\n"
        f"Precio actual: {precio_actual:.2f} USDT\n"
        f"Rango: {grid_activo['soporte']:.0f} - {grid_activo['resistencia']:.0f}\n"
        f"PnL: {pnl:.2f} USDT\n"
        f"Saldo USDT: {saldo_usdt:.2f}\n"
    )
    telegram_mensaje(msg)

    # Kill-switch: si precio fuera del rango ±2%
    if precio_actual:
        lower_band = grid_activo["soporte"] * (1 - KILL_SWITCH_MARGIN)
        upper_band = grid_activo["resistencia"] * (1 + KILL_SWITCH_MARGIN)
        if precio_actual < lower_band or precio_actual > upper_band:
            logger.warning(f"KILL-SWITCH activado! Precio {precio_actual} fuera de banda [{lower_band:.0f}, {upper_band:.0f}]")
            telegram_mensaje(f"🚨 KILL-SWITCH: Precio fuera de rango tolerado. Cancelando grid y vendiendo BTC...")
            cancelar_grid(grid_activo["orderId"])
            vender_todo_btc()
            grid_activo["orderId"] = None
            # Opcional: forzar nuevo análisis en la próxima ejecución

# ================= KEEP ALIVE Y MAIN =================
def keep_alive():
    def ping():
        try:
            requests.get("https://railway.app/health", timeout=5)
        except:
            pass
        threading.Timer(600, ping).start()
    threading.Timer(600, ping).start()

def main():
    logger.info("🚀 Bot Grid Spot Neutral iniciado")
    telegram_mensaje("🚀 Bot Grid Neutral activo - BTCUSDT - 10 USDT fijos, análisis cada 5h")

    keep_alive()

    # Programar análisis cada 5 horas
    schedule.every(5).hours.do(ejecutar_analisis_y_crear_grid)
    # Programar monitoreo cada hora
    schedule.every(1).hours.do(monitorear_y_kill_switch)

    # Al inicio: verificar si hay grid activo
    oid, low, high = listar_grids_activos()
    if oid:
        grid_activo["orderId"] = oid
        grid_activo["soporte"] = low
        grid_activo["resistencia"] = high
        logger.info(f"Grid activo encontrado al inicio: {oid}")
        telegram_mensaje(f"🔄 Bot retomó grid existente: {oid}")
    else:
        # Forzar análisis y creación inmediata
        ejecutar_analisis_y_crear_grid()

    # Bucle principal
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
