import os
import time
import json
import logging
import requests
import numpy as np
import pandas as pd
from scipy.stats import linregress
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
import io
import base64
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
SYMBOLS = ["BTC", "SOL"]   
MIN_INVEST_USDT = 20
MIN_APY_PERCENT = 180.0          
HORA_ANALISIS = "07:00"          

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY,
                default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Dual Asset Bot"})
bybit_session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

# Estado en memoria y contadores globales exigidos
decision_diaria = {}        
global_suscripciones_count = 0
ultimo_resumen = 0

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

# ================= SALDO Y CONVERSIONES EN SPOT =================
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

def obtener_saldo_cripto_unificado(coin):
    try:
        resp = bybit_session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        if resp.get("retCode") != 0:
            return 0.0
        for c_data in resp["result"]["list"][0]["coin"]:
            if c_data["coin"] == coin:
                return float(c_data["walletBalance"])
        return 0.0
    except Exception as e:
        logger.error(f"Error consultando saldo de {coin}: {e}")
        return 0.0

def comprar_activo_en_spot(coin, monto_usdt):
    """Compra el activo base en mercado Spot si la dirección exige depositar la cripto (Sell High)"""
    try:
        symbol = f"{coin}USDT"
        logger.info(f"Comprando {coin} en Spot con {monto_usdt} USDT para cumplir requisitos de suscripción...")
        order = bybit_session.place_order(
            category="spot",
            symbol=symbol,
            side="Buy",
            orderType="Market",
            qty=str(monto_usdt)
        )
        if order.get("retCode") == 0:
            time.sleep(2)  # Pausa de asentamiento en el balance unificado
            saldo_adquirido = obtener_saldo_cripto_unificado(coin)
            logger.info(f"Compra exitosa en Spot. Saldo disponible actual de {coin}: {saldo_adquirido}")
            return True, saldo_adquirido
        else:
            logger.error(f"Fallo al ejecutar orden Spot de compra: {order}")
            return False, 0.0
    except Exception as e:
        logger.error(f"Excepción en compra Spot de emergencia: {e}")
        return False, 0.0

def liquidar_cripto_a_usdt(coin, amount):
    try:
        symbol = f"{coin}USDT"
        order = bybit_session.place_order(
            category="spot",
            symbol=symbol,
            side="Sell",
            orderType="Market",
            qty=str(amount)
        )
        if order.get("retCode") == 0:
            logger.info(f"Liquidación Spot preventiva: {amount} {coin} convertidos a USDT.")
            return True
        return False
    except Exception as e:
        logger.error(f"Error liquidando {coin}: {e}")
        return False

# ================= VELAS DIARIAS Y GRÁFICOS INTERNOS =================
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
    df['trend_intercept'] = intercept
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
        ax.axhline(soporte, color='cyan', ls='--', lw=2, label=f'Soporte ({soporte:.2f})')
    if resistencia:
        ax.axhline(resistencia, color='magenta', ls='--', lw=2, label=f'Resistencia ({resistencia:.2f})')
    if 'ema20' in df_plot.columns:
        ax.plot(x, df_plot['ema20'], 'yellow', lw=2, label='EMA20')
        
    if slope != 0:
        # Corrección del cálculo visual de la línea de tendencia usando su posición real sobre los últimos 60 días
        x_start_idx = len(df_plot) - 60
        x_trend = np.array([x_start_idx, len(df_plot) - 1])
        y_trend = intercept + slope * np.array([0, 59])
        ax.plot(x_trend, y_trend, color='white', linestyle='-.', lw=2, label='Línea de Tendencia', alpha=0.8)
        
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
    
    ax.set_title(f"Análisis Técnico de Tendencias Estructurales - {symbol}", color='white', fontsize=14)
    ax.set_xlabel('Velas Diarias (Historial)', color='white')
    ax.set_ylabel('Precio Referencia (USDT)', color='white')
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

# ================= FILTRADO AVANZADO DE ZONA OPERATIVA =================
def verificar_zona_operativa(precio_actual, soporte, resistencia):
    """
    Define por completo el contexto de la zona operativa.
    Evita operar si el precio está comprimido en el centro exacto del rango dinámico (zona de ruido).
    """
    rango_total = resistencia - soporte
    if rango_total <= 0:
        return False
    
    porcentaje_posicion = ((precio_actual - soporte) / rango_total) * 100
    # Definimos como zona fuera de rango si está en tierra de nadie (entre 40% y 60% del rango de 20 días)
    if 40.0 <= porcentaje_posicion <= 60.0:
        return False
    return True

# ================= PROMPT COMPLETO Y EJECUCIÓN CON IA =================
def analizar_con_gemini(img, symbol):
    try:
        img_b64 = pil_to_base64(img)
        prompt = f"""
        Eres un analista algorítmico experto en estructuras de mercado y productos Earn estructurados de Bybit.
        Examina con detenimiento el gráfico diario proporcionado para {symbol}.
        Tu único objetivo es determinar el movimiento direccional más probable para las próximas 24 horas.
        
        Debes elegir estrictamente bajo las siguientes definiciones operativas:
        - Recomienda "Sell High" si concluyes una estructura alcista sólida, rebote en soporte o RSI saliendo de sobreventa. Esto implica depositar la criptomoneda base para capturar rendimientos mientras se busca vender caro.
        - Recomienda "Buy Low" si concluyes una estructura bajista o agotamiento en resistencia con RSI en sobrecompra. Esto implica colocar USDT en stake para comprar el activo a un precio inferior estructurado.
        - Recomienda "Hold" exclusivamente si hay absoluta neutralidad o falta de confirmación de patrones visuales claros.

        Devuelve única y exclusivamente un objeto JSON plano en una sola línea sin bloques de formato de código:
        {{"decision": "Buy Low"/"Sell High"/"Hold", "razon": "Explicación macro de la estructura detectada", "explicacion": "Análisis técnico exhaustivo de los indicadores", "confianza": 0-100}}
        """
        response = client.chat.completions.create(
            model=MODELO_VISION,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": img_b64}}]}],
            temperature=0.1,
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
        logger.error(f"Error en procesamiento de visión IA para {symbol}: {e}")
        return "Hold", f"Excepción de análisis: {e}", "", 0

def ejecutar_analisis_diario():
    global decision_diaria
    logger.info("=================== INICIANDO LOG DE HEARTBEAT DIARIO ===================")
    telegram_mensaje("📈 Ejecutando escaneo estructural y visual diario con IA...")
    
    for base in SYMBOLS:
        df = obtener_velas_diarias(base, limit=100)
        if df.empty:
            logger.warning(f"No se pudieron descargar datos del par para {base}")
            continue
            
        df = calcular_indicadores_diarios(df)
        soporte, resistencia = detectar_soportes_resistencias_diario(df)
        precio_actual = df['close'].iloc[-1]
        slope = df['trend_slope'].iloc[-1]
        intercept = df['trend_intercept'].iloc[-1]
        
        # Validación obligatoria de contexto operativo completo
        en_zona = verificar_zona_operativa(precio_actual, soporte, resistencia)
        if not en_zona:
            msg_fuera_zona = f"⚠️ Patrón detectado pero fuera de zona operativa para {base} (Precio: {precio_actual:.2f}, Rango: {soporte:.2f} - {resistencia:.2f})"
            logger.warning(msg_fuera_zona)
            telegram_mensaje(msg_fuera_zona)
            decision_diaria[base] = "Hold"
            continue
            
        img = generar_grafico_diario(df, base, soporte, resistencia, slope, intercept)
        if img:
            img_path = f"/tmp/duplo_{base}.png"
            img.save(img_path)
            
            # Identificar el sesgo de la tendencia para cumplir el log de estructura lineal
            sentido_mercado = "BULLISH" if slope > 0 else "BEARISH"
            logger.info(f"Log de Estructura de Mercado para {base}: {sentido_mercado} | Soporte: {soporte} | Resistencia: {resistencia}")
            
            telegram_enviar_imagen(img_path, caption=f"📊 Gráfico de Contexto Técnico Diario - {base}\nTendencia estructural: {sentido_mercado}")
            decision, razon, explicacion, confianza = analizar_con_gemini(img, base)
        else:
            decision, razon, confianza = "Hold", "Error crítico generando matriz del gráfico", 0

        log_completo_heartbeat = (
            f"🤖 Heartbeat de Análisis [{base}] -\n"
            f"Dirección sugerida por Visión: {decision}\n"
            f"Confianza de análisis: {confianza}%\n"
            f"Rango de Red de Soporte/Resistencia: [{soporte:.2f} - {resistencia:.2f}]\n"
            f"Razón técnica: {razon}"
        )
        logger.info(log_completo_heartbeat)
        telegram_mensaje(log_completo_heartbeat)

        if decision != "Hold" and confianza >= 60:
            decision_diaria[base] = decision
        else:
            logger.info(f"Suscripción rechazada o en espera para {base}. Razón: Confianza insuficiente o recomendación neutral de mantenimiento (Hold).")
            decision_diaria[base] = "Hold"
            
    telegram_mensaje(f"📌 Estado consolidado de intenciones del día: BTC={decision_diaria.get('BTC')}, SOL={decision_diaria.get('SOL')}")

# ================= SOLUCIÓN DE CONSULTA DE PRODUCTOS DUAL ASSET =================
def obtener_productos_dual_asset():
    """
    CORRECCIÓN CRÍTICA: Se eliminan filtros iniciales restrictivos en los parámetros de URL.
    Bybit devuelve listas paginadas basadas en el colateral. Consultamos de forma expandida para unificar todo el pool.
    """
    productos_encontrados = []
    # Consultamos tanto productos basados en inversión cripto como inversiones en dólares estables (USDT)
    monedas_consulta = ["USDT", "BTC", "SOL"]
    
    try:
        for m_coin in monedas_consulta:
            timestamp = str(int(time.time() * 1000))
            recv_window = "5000"
            query = f"status=active&coin={m_coin}"
            payload = timestamp + BYBIT_API_KEY + recv_window + query
            signature = hmac.new(BYBIT_API_SECRET.encode('utf-8'), payload.encode('utf-8'), hashlib.sha256).hexdigest()
            
            headers = {
                "X-BAPI-API-KEY": BYBIT_API_KEY,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": recv_window,
                "X-BAPI-SIGN": signature,
            }
            url = f"https://api.bybit.com/v5/earn/dual-asset/product?{query}"
            resp = requests.get(url, headers=headers, timeout=12)
            data = resp.json()
            
            if data.get("retCode") == 0 and "result" in data and "list" in data["result"]:
                for prod in data["result"]["list"]:
                    product_id = prod.get("productId")
                    # El campo 'coin' es la moneda de depósito, 'linkToCoin' es el activo subyacente vinculado
                    dep_coin = prod.get("coin")
                    linked_coin = prod.get("linkToCoin")
                    direction = prod.get("direction") 
                    
                    apy_e8 = prod.get("apyE8", "0")
                    apy_parseado = float(apy_e8) / 1e8 if float(apy_e8) > 0 else 0.0
                    
                    min_amount = float(prod.get("minAmount", 0))
                    max_amount = float(prod.get("maxAmount", 0))
                    
                    productos_encontrados.append({
                        "productId": product_id,
                        "depositCoin": dep_coin,
                        "linkToCoin": linked_coin,
                        "direction": direction,
                        "apy": apy_parseado,
                        "minAmount": min_amount,
                        "maxAmount": max_amount
                    })
        logger.info(f"Búsqueda global unificada: Total de productos activos parseados de Bybit: {len(productos_encontrados)}")
        return productos_encontrados
    except Exception as e:
        logger.error(f"Excepción severa escaneando API de productos: {e}")
        return []

def obtener_mejor_producto(coin, decision):
    """Filtra y selecciona dinámicamente la oferta con mayor rendimiento."""
    pool_completo = obtener_productos_dual_asset()
    filtrados = []
    
    for p in pool_completo:
        # Si queremos comprar abajo (Buy Low), depositamos USDT vinculados a la moneda base
        if decision == "Buy Low":
            if p["depositCoin"] == "USDT" and p["linkToCoin"] == coin and p["direction"] == "Buy Low":
                filtrados.append(p)
        # Si queremos vender arriba (Sell High), depositamos la cripto base directamente
        elif decision == "Sell High":
            if p["depositCoin"] == coin and p["direction"] == "Sell High":
                filtrados.append(p)
                
    # Filtrar por requerimientos mínimos exigidos por el usuario
    validos = [f for f in filtrados if f["apy"] >= MIN_APY_PERCENT]
    if not validos:
        logger.warning(f"Filtro de Selección Vacío: No hay ofertas en la dirección {decision} para {coin} que superen el APR mínimo de {MIN_APY_PERCENT}%")
        return None
        
    mejor = max(validos, key=lambda x: x["apy"])
    logger.info(f"✔ Oferta óptima localizada para {coin} [{decision}]: ID {mejor['productId']} - APR Encontrado: {mejor['apy']}%")
    return mejor

# ================= ADQUISICIÓN DE QUOTE FIJO DE EJERCICIO =================
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
            logger.error(f"Error adquiriendo parámetros de cotización para {product_id}: {data}")
            return None
    except Exception as e:
        logger.error(f"Excepción en paso de cotización fija: {e}")
        return None

# ================= EJECUCIÓN FIRME DE LA SUSCRIPCIÓN =================
def suscribir_dual_asset(product_id, amount, decision, quote_info):
    global global_suscripciones_count
    order_link_id = f"duplo_{int(time.time())}"
    side = "Buy" if decision == "Buy Low" else "Sell"
    
    body = {
        "productId": product_id,
        "side": side,
        "orderType": "Stake",
        "amount": str(amount),
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
            global_suscripciones_count += 1
            logger.info(f"Suscripción confirmada con éxito. ID de producto: {product_id}. Volumen colocado: {amount}")
            return True, order_link_id
        else:
            logger.error(f"Rechazo de colocación en Bybit: {result}")
            telegram_mensaje(f"❌ Error de suscripción para el producto {product_id}: {result.get('retMsg')}")
            return False, None
    except Exception as e:
        logger.error(f"Excepción crítica durante la inyección de la suscripción: {e}")
        return False, None

# ================= ESCANEO DE SUSCRIPCIONES ACTIVAS =================
def obtener_posiciones_activas():
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        query = ""  
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
        logger.error(f"Error obteniendo pool de suscripciones activas: {e}")
        return []

def reporte_posiciones_bloqueadas():
    global ultimo_resumen
    ahora = time.time()
    if ahora - ultimo_resumen >= 10800: # Registro recurrente informativo de cada 3 horas
        posiciones = obtener_posiciones_activas()
        if not posiciones:
            telegram_mensaje("📊 Estado actual: No existen suscripciones bloqueadas en Dual Asset de momento.")
        else:
            texto = f"📊 BITÁCORA DE SUSCRIPCIONES BLOQUEADAS EN POOL ({len(posiciones)})\n"
            texto += f"Contador de operaciones históricas del bot: {global_suscripciones_count}\n"
            for pos in posiciones:
                venc = datetime.fromtimestamp(pos["expireTime"] / 1000).strftime("%Y-%m-%d %H:%M UTC")
                texto += f"🔹 Prod: {pos['productId']} | Capital asignado: {pos['amount']:.4f} {pos['coin']} | Fecha vencimiento: {venc}\n"
            telegram_mensaje(texto)
        ultimo_resumen = ahora

# ================= CICLO EJECUTOR HORARIO HORARIO =================
def ciclo_horario():
    try:
        logger.info("🕒 INICIANDO REVISIÓN EN CICLO HORARIO DE OFERTAS Y BALANCES")
        reporte_posiciones_bloqueadas()
        
        saldo_usdt = obtener_saldo_usdt_unificado()
        logger.info(f"Registro de balances - USDT Disponible en Cuenta Unificada: {saldo_usdt:.2f}")
        
        for base in SYMBOLS:
            decision = decision_diaria.get(base)
            if not decision or decision == "Hold":
                logger.info(f"Acción para {base}: Omitido (Estado actual en Hold/Neutral). No se buscan suscripciones.")
                continue
            
            # Buscar el producto idóneo libre de restricciones erróneas de filtrado de monedas
            mejor_producto = obtener_mejor_producto(base, decision)
            if not mejor_producto:
                logger.info(f"Rechazo en ciclo para {base}: No se localizaron ofertas vigentes que cumplan las reglas para la dirección {decision}.")
                continue
            
            product_id = mejor_producto["productId"]
            apy = mejor_producto["apy"]
            max_amount = mejor_producto["maxAmount"]
            coin_deposito = mejor_producto["depositCoin"]
            
            quote = obtener_quote_fijo_dual_asset(product_id)
            if not quote:
                logger.error(f"Fallo de comunicación al solicitar cotización en firme para {product_id}")
                continue
            
            # Gestión Dinámica de Colaterales según Requisitos del Contrato Dual Asset
            monto_inversion = 0.0
            
            if decision == "Buy Low":
                # Requiere depositar USDT de forma nativa
                if saldo_usdt < MIN_INVEST_USDT:
                    logger.info(f"Suscripción cancelada por liquidez: Saldo USDT ({saldo_usdt:.2f}) insuficiente para mínimo requerido ({MIN_INVEST_USDT}).")
                    continue
                monto_inversion = min(saldo_usdt, max_amount, quote["maxInvestmentAmount"])
                
                logger.info(f"Suscripción autorizada para {base} [Buy Low] con {monto_inversion} USDT. APY pactado: {apy}%")
                exito, order_id = suscribir_dual_asset(product_id, monto_inversion, decision, quote)
                if exito:
                    telegram_mensaje(f"🟢 SUSCRIPCIÓN ESTABLECIDA\nProducto: {product_id}\nMoneda Base: {base}\nDirección: {decision}\nMonto: {monto_inversion:.2f} USDT\nAPY: {apy}%")
                    saldo_usdt -= monto_inversion
                    
            elif decision == "Sell High":
                # Requiere stakear la Criptomoneda nativa (BTC/SOL). Verificamos si ya hay balance o compramos en Spot.
                saldo_cripto_actual = obtener_saldo_cripto_unificado(base)
                precio_par = quote["currentPrice"]
                
                # Definimos un valor equivalente estimado de la inversión traducido a cripto
                monto_usdt_a_usar = min(saldo_usdt, max_amount * precio_par, quote["maxInvestmentAmount"] * precio_par)
                if monto_usdt_a_usar < MIN_INVEST_USDT:
                    logger.info("Monto de inversión remanente insuficiente en balance USDT general.")
                    continue
                
                # Ejecutar compra spot preventiva inmediata si no contamos con el colateral bloqueable
                monto_cripto_objetivo = monto_usdt_a_usar / precio_par
                if saldo_cripto_actual < monto_cripto_objetivo:
                    exito_compra, nuevo_saldo_cripto = comprar_activo_en_spot(base, monto_usdt_a_usar)
                    if not exito_compra or nuevo_saldo_cripto <= 0:
                        logger.error(f"Cancelación de operación: No se pudo adquirir el colateral {base} en Spot necesario para Sell High.")
                        continue
                    saldo_cripto_actual = nuevo_saldo_cripto
                
                # Asignamos el monto final en unidades de la moneda cripto correspondiente
                monto_inversion = min(saldo_cripto_actual, max_amount, quote["maxInvestmentAmount"])
                
                logger.info(f"Suscripción autorizada para {base} [Sell High] con {monto_inversion} {base}. APY pactado: {apy}%")
                exito, order_id = suscribir_dual_asset(product_id, monto_inversion, decision, quote)
                if exito:
                    telegram_mensaje(f"🟢 SUSCRIPCIÓN ESTABLECIDA\nProducto: {product_id}\nMoneda Base: {base}\nDirección: {decision}\nMonto: {monto_inversion:.6f} {base}\nAPY: {apy}%")
                    saldo_usdt = obtener_saldo_usdt_unificado() # Refrescar balance remanente real
                    
            time.sleep(3)
    except Exception as e:
        logger.error(f"Excepción capturada en ejecución de bucle horario de suscripciones: {str(e)}\n{traceback.format_exc()}")

# ================= MANTENIMIENTO DE INSTANCIA / RAILWAY KEEP-ALIVE =================
def keep_alive():
    def ping():
        try:
            requests.get("https://railway.app/health", timeout=5)
        except:
            pass
        threading.Timer(600, ping).start()
    threading.Timer(600, ping).start()

# ================= FILTRADO PRINCIPAL DE INICIO =================
def main():
    logger.info("🚀 Inicializando ejecutor nativo para Bybit Dual Asset Mining - Modo Producción")
    keep_alive()
    time.sleep(2)
    
    # Programación exacta de tareas recurrentes cronometradas
    schedule.every().day.at(HORA_ANALISIS).do(ejecutar_analisis_diario)
    schedule.every(1).hours.do(ciclo_horario)
    
    # Ejecución inmediata inicial al encendido del contenedor de Railway
    ejecutar_analisis_diario()
    ciclo_horario()
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()
