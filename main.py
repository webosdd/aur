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
FIXED_INVEST_USDT = 10          # Monto fijo en USDT para el grid
GRID_COUNT = 20                 # Niveles del grid
KILL_SWITCH_MARGIN = 0.02       # 2% fuera del rango

# Cliente OpenRouter (IA)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Grid Bot"}
)

# Cliente Bybit (para funciones que no requieren firmas manuales)
bybit = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# Estado en memoria (solo para evitar múltiples consultas)
grid_activo = {
    "grid_id": None,
    "soporte": None,
    "resistencia": None,
    "createdAt": None
}

# ================= TELEGRAM =================
def telegram_mensaje(texto):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": texto},
            timeout=10
        )
    except Exception as e:
        logger.error(f"Telegram error: {e}")

# ================= VELAS E INDICADORES =================
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
        logger.error(f"Error velas: {e}")
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
    return df.dropna()

def soporte_resistencia_historicos(df):
    if len(df) < 20:
        return 0, 0
    soporte = df['low'].rolling(20).min().iloc[-1]
    resistencia = df['high'].rolling(20).max().iloc[-1]
    return soporte, resistencia

# ================= IA (OpenRouter) =================
def analizar_con_ia(df, soporte_hist, resistencia_hist):
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
Eres un analista técnico experto en criptomonedas. Basado en los datos diarios de BTCUSDT, determina niveles de soporte y resistencia para las próximas 5 horas.
También indica la tendencia (Bullish/Bearish/Neutral) y una confianza del 0 al 100.

Datos:
{resumen}

Reglas:
- Soporte debe ser un precio por debajo del actual donde haya alta probabilidad de rebote.
- Resistencia debe ser un precio por encima del actual.
- Si la tendencia es alcista, la resistencia puede estar más lejos; si bajista, el soporte más profundo.
- Responde ÚNICAMENTE con un JSON en UNA línea:
{{"soporte": 45000.0, "resistencia": 52000.0, "tendencia": "Bullish", "confianza": 85}}
"""
    try:
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-lite",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            timeout=60
        )
        contenido = response.choices[0].message.content
        inicio = contenido.find('{')
        fin = contenido.rfind('}') + 1
        if inicio != -1 and fin != 0:
            contenido = contenido[inicio:fin]
        datos = json.loads(contenido)
        soporte = float(datos.get("soporte", soporte_hist))
        resistencia = float(datos.get("resistencia", resistencia_hist))
        tendencia = datos.get("tendencia", "Neutral")
        confianza = int(datos.get("confianza", 50))
        if soporte >= resistencia or soporte <= 0 or resistencia <= 0:
            soporte, resistencia = soporte_hist, resistencia_hist
        return soporte, resistencia, tendencia, confianza
    except Exception as e:
        logger.error(f"Error IA: {e}")
        return soporte_hist, resistencia_hist, "Neutral", 0

# ================= GRID BOT (END CORRECTOS V5) =================
def validar_grid_params(soporte, resistencia):
    """Validación previa usando /v5/grid/validate-input"""
    try:
        url = "https://api.bybit.com/v5/grid/validate-input"
        body = {
            "symbol": SYMBOL,
            "min_price": str(soporte),
            "max_price": str(resistencia),
            "cell_number": GRID_COUNT,
            "quote_investment": str(FIXED_INVEST_USDT),
            "invest_mode": 0
        }
        resp = requests.post(url, json=body, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            check_code = data["result"].get("check_code")
            if check_code == "SPOT_CHECK_CODE_SUCCESS_UNSPECIFIED":
                logger.info("Validación correcta")
                return True
            else:
                logger.warning(f"Validación falló: {check_code}")
                return False
        else:
            logger.error(f"Error validación: {data}")
            return False
    except Exception as e:
        logger.error(f"Excepción validación: {e}")
        return False

def crear_grid_spot(soporte, resistencia):
    """Crea grid neutral con endpoint /v5/grid/create-grid"""
    try:
        if not validar_grid_params(soporte, resistencia):
            telegram_mensaje("⚠️ Parámetros de grid inválidos según Bybit")
            return None

        params = {
            "symbol": SYMBOL,
            "min_price": str(soporte),
            "max_price": str(resistencia),
            "cell_number": GRID_COUNT,
            "quote_investment": str(FIXED_INVEST_USDT),
            "invest_mode": 0,
            "enable_trailing": False
        }
        url = "https://api.bybit.com/v5/grid/create-grid"
        body_str = json.dumps(params)
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        payload = timestamp + BYBIT_API_KEY + recv_window + body_str
        signature = hmac.new(
            BYBIT_API_SECRET.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }
        resp = requests.post(url, headers=headers, data=body_str, timeout=15)
        if not resp.text:
            logger.error("Respuesta vacía de la API")
            telegram_mensaje("❌ Respuesta vacía de Bybit")
            return None
        data = resp.json()
        logger.info(f"Respuesta create-grid: {data}")
        if data.get("retCode") == 0 and data["result"].get("status_code") == 200:
            grid_id = data["result"]["grid_id"]
            logger.info(f"Grid creado con ID: {grid_id}")
            telegram_mensaje(
                f"🟢 Grid Neutral activo\n"
                f"Rango: {soporte:.0f} - {resistencia:.0f} USDT\n"
                f"Inversión: {FIXED_INVEST_USDT} USDT\n"
                f"Niveles: {GRID_COUNT}\n"
                f"ID: {grid_id}"
            )
            return grid_id
        else:
            logger.error(f"Error creando grid: {data}")
            telegram_mensaje(f"❌ Error creando grid: {data.get('retMsg', 'desconocido')}")
            return None
    except json.JSONDecodeError as e:
        logger.error(f"JSON inválido: {e} - Respuesta: {resp.text[:200]}")
        return None
    except Exception as e:
        logger.error(f"Excepción creando grid: {e}")
        return None

def cancelar_grid(grid_id):
    """Cancela grid usando /v5/grid/close-grid"""
    try:
        url = "https://api.bybit.com/v5/grid/close-grid"
        body = {"grid_id": str(grid_id)}
        body_str = json.dumps(body)
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        payload = timestamp + BYBIT_API_KEY + recv_window + body_str
        signature = hmac.new(
            BYBIT_API_SECRET.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }
        resp = requests.post(url, headers=headers, data=body_str, timeout=15)
        data = resp.json()
        if data.get("retCode") == 0:
            logger.info(f"Grid {grid_id} cancelado")
            telegram_mensaje(f"⚠️ Grid {grid_id} cancelado (kill-switch)")
            return True
        else:
            logger.error(f"Error cancelando: {data}")
            return False
    except Exception as e:
        logger.error(f"Excepción cancelando: {e}")
        return False

def obtener_detalle_grid(grid_id):
    """Detalle del grid mediante /v5/grid/get-detail"""
    try:
        url = f"https://api.bybit.com/v5/grid/get-detail?grid_id={grid_id}"
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = f"grid_id={grid_id}"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(
            BYBIT_API_SECRET.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
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
            logger.error(f"Error detalle: {data}")
            return None
    except Exception as e:
        logger.error(f"Excepción detalle: {e}")
        return None

def listar_grids_activos():
    """Lista grids activos de BTCUSDT -> devuelve (grid_id, min_price, max_price)"""
    try:
        url = f"https://api.bybit.com/v5/grid/list?symbol={SYMBOL}&grid_status=active&limit=10"
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = f"symbol={SYMBOL}&grid_status=active&limit=10"
        payload = timestamp + BYBIT_API_KEY + recv_window + query
        signature = hmac.new(
            BYBIT_API_SECRET.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        headers = {
            "X-BAPI-API-KEY": BYBIT_API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN": signature,
        }
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0 and data["result"]["list"]:
            grid = data["result"]["list"][0]
            return grid["grid_id"], float(grid["min_price"]), float(grid["max_price"])
        return None, None, None
    except Exception as e:
        logger.error(f"Error listando grids: {e}")
        return None, None, None

# ================= MONITOREO Y KILL-SWITCH =================
def obtener_precio_actual():
    try:
        ticker = bybit.get_tickers(category="spot", symbol=SYMBOL)
        if ticker.get("retCode") == 0:
            return float(ticker["result"]["list"][0]["lastPrice"])
        return None
    except Exception as e:
        logger.error(f"Error precio: {e}")
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
    """Vende todo el BTC disponible en la wallet unificada a mercado"""
    try:
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
            logger.info(f"Vendidos {btc_amount} BTC")
            telegram_mensaje(f"💰 Venta de emergencia: {btc_amount} BTC → USDT")
        else:
            logger.error(f"Error venta: {order}")
    except Exception as e:
        logger.error(f"Excepción venta BTC: {e}")

def monitorear_y_kill_switch():
    global grid_activo
    logger.info("=== MONITOREO HORARIO ===")
    # Si no tenemos grid_id en memoria, intentar recuperar desde API
    if not grid_activo["grid_id"]:
        gid, low, high = listar_grids_activos()
        if gid:
            grid_activo["grid_id"] = gid
            grid_activo["soporte"] = low
            grid_activo["resistencia"] = high
            logger.info(f"Grid recuperado: {gid}")
        else:
            logger.info("No hay grid activo")
            return

    detalle = obtener_detalle_grid(grid_activo["grid_id"])
    if not detalle:
        logger.warning("No se pudo obtener detalle, limpiando estado")
        grid_activo["grid_id"] = None
        return

    status = detalle.get("grid_status")  # 1 activo, 2 detenido, 3 finalizado
    pnl = float(detalle.get("total_pnl", 0))
    precio_actual = obtener_precio_actual()
    saldo_usdt = obtener_saldo_usdt()

    msg = (
        f"📊 ESTADO GRID {SYMBOL}\n"
        f"Estado: {'🟢 Activo' if status == 1 else '⭕ Inactivo'}\n"
        f"Precio actual: {precio_actual:.2f} USDT\n"
        f"Rango: {grid_activo['soporte']:.0f} - {grid_activo['resistencia']:.0f}\n"
        f"PnL total: {pnl:.2f} USDT\n"
        f"Saldo USDT: {saldo_usdt:.2f}\n"
    )
    telegram_mensaje(msg)

    # Kill-switch
    if precio_actual:
        lower_band = grid_activo["soporte"] * (1 - KILL_SWITCH_MARGIN)
        upper_band = grid_activo["resistencia"] * (1 + KILL_SWITCH_MARGIN)
        if precio_actual < lower_band or precio_actual > upper_band:
            telegram_mensaje(
                f"🚨 KILL-SWITCH activado\n"
                f"Precio {precio_actual:.0f} fuera de banda [{lower_band:.0f}, {upper_band:.0f}]"
            )
            cancelar_grid(grid_activo["grid_id"])
            vender_todo_btc()
            grid_activo["grid_id"] = None

# ================= ANÁLISIS Y CREACIÓN DE GRID (CADA 5h) =================
def ejecutar_analisis_y_crear_grid():
    global grid_activo
    logger.info("=== ANÁLISIS CADA 5 HORAS ===")
    telegram_mensaje("📊 Iniciando análisis técnico para grid bot...")

    # Verificar si ya existe un grid activo (evitar duplicados)
    gid, low, high = listar_grids_activos()
    if gid:
        logger.info(f"Ya hay grid activo {gid}, no se crea nuevo")
        grid_activo["grid_id"] = gid
        grid_activo["soporte"] = low
        grid_activo["resistencia"] = high
        return

    df = obtener_velas_diarias(SYMBOL, limit=100)
    if df.empty:
        logger.error("No se pudieron obtener velas")
        return
    df = calcular_indicadores(df)
    sop_hist, res_hist = soporte_resistencia_historicos(df)
    if sop_hist == 0 or res_hist == 0:
        logger.error("No se pudo calcular soporte/resistencia históricos")
        return

    soporte, resistencia, tendencia, confianza = analizar_con_ia(df, sop_hist, res_hist)
    logger.info(f"IA: soporte={soporte}, resistencia={resistencia}, tendencia={tendencia}, confianza={confianza}%")
    telegram_mensaje(
        f"🤖 IA:\nSoporte {soporte:.0f} / Resistencia {resistencia:.0f}\n"
        f"Tendencia {tendencia} (confianza {confianza}%)"
    )

    if soporte < resistencia and soporte > 0 and resistencia > 0:
        nuevo_id = crear_grid_spot(soporte, resistencia)
        if nuevo_id:
            grid_activo["grid_id"] = nuevo_id
            grid_activo["soporte"] = soporte
            grid_activo["resistencia"] = resistencia
            grid_activo["createdAt"] = datetime.now()
        else:
            logger.error("Fallo al crear grid")
    else:
        logger.error("Niveles inválidos, no se crea grid")

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
    logger.info("🚀 Bot Grid Neutral (BTCUSDT) iniciado")
    telegram_mensaje("🚀 Bot Grid Neutral activo - BTCUSDT | 10 USDT | análisis cada 5h")

    keep_alive()

    # Programar tareas
    schedule.every(5).hours.do(ejecutar_analisis_y_crear_grid)
    schedule.every(1).hours.do(monitorear_y_kill_switch)

    # Al inicio: recuperar grid activo si existe
    gid, low, high = listar_grids_activos()
    if gid:
        grid_activo["grid_id"] = gid
        grid_activo["soporte"] = low
        grid_activo["resistencia"] = high
        logger.info(f"Grid activo encontrado: {gid}")
        telegram_mensaje(f"🔄 Bot retomó grid existente: {gid}")
    else:
        # Forzar análisis y creación inmediata
        ejecutar_analisis_y_crear_grid()

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
