# =================================================================
# BOT BYBIT EARN DUAL ASSET (DUPLOS) con IA Multimodal
# - Analiza gráficos diarios (velas, EMAs, RSI, MACD, S/R, tendencia)
# - IA decide activo (BTC/SOL) y tipo (Buy Low / Sell High)
# - Suscribe a producto de 1 día con mejor APR
# - Espera activamente hasta vencimiento (evita hibernación en Railway)
# - Convierte saldos no USDT automáticamente y reinicia ciclo
# - Reportes completos con gráfico por Telegram
# =================================================================

import os
import time
import io
import base64
import hmac
import hashlib
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
from openai import OpenAI

# ======================================================
# CONFIGURACIÓN
# ======================================================

ASSET_PAIRS = ["BTCUSDT", "SOLUSDT"]           # Activos a considerar
TIMEFRAME = "D"                                # Velas diarias (1 día)
LOOKBACK_DAYS = 60                             # Días de histórico para gráfico
INVEST_PERCENT = 1.0                           # Invertir 100% del USDT disponible
SPOT_COMMISSION = 0.001                        # 0.1% comisión spot
KEEP_AWAKE_INTERVAL_SECONDS = 300              # Ping cada 5 min para evitar hibernación

# ======================================================
# CREDENCIALES (variables de entorno)
# ======================================================

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not BYBIT_API_KEY or not BYBIT_API_SECRET:
    raise Exception("❌ BYBIT_API_KEY o BYBIT_API_SECRET no configuradas")
if not OPENROUTER_API_KEY:
    raise Exception("❌ OPENROUTER_API_KEY no configurada")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    default_headers={"HTTP-Referer": "https://railway.app", "X-Title": "Dual Asset Bot"}
)
MODELO_IA = "google/gemini-3.1-flash-image-preview"   # Cambia si lo deseas

# ======================================================
# FUNCIONES BYBIT API v5
# ======================================================

BASE_URL = "https://api.bybit.com"

def _bybit_request(method, endpoint, params=None, body=None):
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    query_string = ""
    if method == "GET" and params:
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
    elif method == "POST" and body:
        query_string = json.dumps(body)
    
    signature_payload = timestamp + BYBIT_API_KEY + recv_window + query_string
    signature = hmac.new(
        BYBIT_API_SECRET.encode("utf-8"),
        signature_payload.encode("utf-8"),
        hashlib.sha256
    ).hexdigest()

    headers = {
        "X-BAPI-API-KEY": BYBIT_API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "X-BAPI-RECV-WINDOW": recv_window,
        "Content-Type": "application/json"
    }
    url = BASE_URL + endpoint
    if method == "GET":
        response = requests.get(url, headers=headers, params=params, timeout=15)
    else:
        response = requests.post(url, headers=headers, json=body, timeout=15)
    
    try:
        data = response.json()
    except:
        raise Exception(f"Respuesta no JSON: {response.text}")
    
    if data.get("retCode") != 0:
        raise Exception(f"Bybit error: {data.get('retMsg')} (code {data.get('retCode')})")
    return data.get("result", {})

def obtener_velas(symbol, interval="D", limit=60):
    """Obtiene velas diarias (o del intervalo indicado) para un símbolo."""
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    if data.get("retCode") != 0:
        raise Exception(f"Error velas {symbol}: {data}")
    df = pd.DataFrame(data["result"]["list"], columns=['time','open','high','low','close','volume','turnover'])
    df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    df['time'] = pd.to_datetime(df['time'].astype(np.int64), unit='ms', utc=True)
    df.set_index('time', inplace=True)
    return df

def calcular_indicadores(df):
    """Añade EMA20, EMA50, RSI, MACD, ATR, soporte/resistencia, tendencia."""
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # ATR
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    
    # Soporte / Resistencia (mínimos y máximos de 20 velas)
    df['soporte'] = df['low'].rolling(20).min()
    df['resistencia'] = df['high'].rolling(20).max()
    
    # Tendencia lineal en últimos 30 días
    y = df['close'].values[-30:]
    x = np.arange(len(y))
    slope, intercept = np.polyfit(x, y, 1)
    if slope > 0.02:
        tendencia = "ALCISTA"
    elif slope < -0.02:
        tendencia = "BAJISTA"
    else:
        tendencia = "LATERAL"
    df['tendencia'] = tendencia
    df['tendencia_slope'] = slope
    return df

def generar_grafico_base64(df, symbol, titulo_extra=""):
    """Genera gráfico profesional con velas, EMAs, RSI, MACD y lo devuelve en base64."""
    try:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True, 
                                             gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Gráfico principal de velas
        x = np.arange(len(df))
        for i, (idx, row) in enumerate(df.iterrows()):
            o, h, l, c = row['open'], row['high'], row['low'], row['close']
            color = 'green' if c >= o else 'red'
            ax1.vlines(i, l, h, color=color, linewidth=1)
            cuerpo_y = min(o, c)
            cuerpo_h = abs(c - o)
            if cuerpo_h == 0:
                cuerpo_h = 0.0001
            rect = plt.Rectangle((i - 0.3, cuerpo_y), 0.6, cuerpo_h, color=color, alpha=0.9)
            ax1.add_patch(rect)
        
        # EMAs
        ax1.plot(x, df['ema20'].values, color='orange', linewidth=2, label='EMA20')
        ax1.plot(x, df['ema50'].values, color='blue', linewidth=2, label='EMA50')
        
        # Soporte / Resistencia
        ax1.axhline(df['soporte'].iloc[-1], color='cyan', linestyle='--', alpha=0.7, label=f"Soporte {df['soporte'].iloc[-1]:.2f}")
        ax1.axhline(df['resistencia'].iloc[-1], color='magenta', linestyle='--', alpha=0.7, label=f"Resistencia {df['resistencia'].iloc[-1]:.2f}")
        
        # Línea de tendencia
        y_tend = df['close'].values[-30:]
        x_tend = np.arange(len(df)-30, len(df))
        slope = df['tendencia_slope'].iloc[-1]
        intercept = np.polyfit(x_tend - x_tend[0], y_tend, 1)[1]
        tend_line = intercept + slope * (x_tend - x_tend[0])
        ax1.plot(x_tend, tend_line, color='red', linewidth=2, linestyle='-', label=f"Tendencia (slope {slope:.4f})")
        
        ax1.set_title(f"{symbol} - Velas diarias | {titulo_extra}")
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.2)
        
        # RSI
        ax2.plot(x, df['rsi'].values, color='purple', linewidth=1.5)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.2)
        
        # MACD
        ax3.plot(x, df['macd'].values, color='blue', linewidth=1.5, label='MACD')
        ax3.plot(x, df['macd_signal'].values, color='red', linewidth=1.5, label='Señal')
        ax3.bar(x, df['macd_hist'].values, color='gray', alpha=0.5, width=0.8, label='Histograma')
        ax3.set_ylabel('MACD')
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.2)
        
        # Formato eje X
        step = max(1, int(len(df)/6))
        ax3.set_xticks(x[::step])
        ax3.set_xticklabels([t.strftime('%Y-%m-%d') for t in df.index[::step]], rotation=45)
        
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        print(f"Error generando gráfico: {e}")
        return None

# ======================================================
# IA MULTIMODAL: Analiza gráfico y decide activo y tipo
# ======================================================

def ia_analizar_grafico(symbol, df):
    """Envía gráfico + datos a Gemini y devuelve: 'BUY_LOW' o 'SELL_HIGH' o 'HOLD'"""
    img_b64 = generar_grafico_base64(df, symbol, "Análisis para Dual Asset")
    if not img_b64:
        return "HOLD", ["Error gráfico"]
    
    ultimo = df.iloc[-1]
    precio_actual = ultimo['close']
    tendencia = ultimo['tendencia']
    slope = ultimo['tendencia_slope']
    rsi = ultimo['rsi']
    macd_hist = ultimo['macd_hist']
    soporte = ultimo['soporte']
    resistencia = ultimo['resistencia']
    atr = ultimo['atr']
    
    prompt = f"""
Eres un experto trader de criptomonedas. Analiza el gráfico diario de {symbol}.
Datos actuales:
- Precio: {precio_actual:.2f}
- Tendencia: {tendencia} (pendiente {slope:.5f})
- RSI: {rsi:.1f}
- MACD histograma: {macd_hist:.2f}
- Soporte: {soporte:.2f}
- Resistencia: {resistencia:.2f}
- ATR (volatilidad): {atr:.2f}

El objetivo es invertir en un producto **Dual Asset (Duplos) de Bybit con vencimiento en 1 día** para **acumular USDT**.
- Si seleccionas **Buy Low**: inviertes USDT. Recibirás USDT + intereses si el precio al vencimiento está **por encima** del precio objetivo (precio actual + pequeño margen). Si baja, recibes la cripto.
- Si seleccionas **Sell High**: inviertes la cripto (BTC o SOL). Recibirás USDT + intereses si el precio está **por debajo** del precio objetivo.

Decide cuál opción tiene mayor probabilidad de que al vencimiento (1 día) el pago sea en USDT, basándote en el análisis técnico del gráfico.
Si el mercado está muy incierto, responde "HOLD".

Responde ÚNICAMENTE en JSON:
{{"decision": "BUY_LOW" / "SELL_HIGH" / "HOLD", "razones": ["razón1", "razón2"]}}
"""
    try:
        response = client.chat.completions.create(
            model=MODELO_IA,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                ]}
            ],
            temperature=0.3,
            max_tokens=300
        )
        contenido = response.choices[0].message.content
        import re
        json_match = re.search(r'\{.*\}', contenido, re.DOTALL)
        data = json.loads(json_match.group(0))
        decision = data.get("decision", "HOLD")
        razones = data.get("razones", ["Sin razones"])
        if decision not in ["BUY_LOW", "SELL_HIGH", "HOLD"]:
            decision = "HOLD"
        return decision, razones, img_b64
    except Exception as e:
        print(f"Error IA: {e}")
        return "HOLD", [f"Error: {e}"], img_b64

# ======================================================
# TELEGRAM
# ======================================================

def telegram_mensaje(texto, imagen_base64=None):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": texto}, timeout=10)
        if imagen_base64:
            url_photo = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
            files = {'photo': ('chart.png', base64.b64decode(imagen_base64), 'image/png')}
            requests.post(url_photo, files=files, data={'chat_id': TELEGRAM_CHAT_ID}, timeout=15)
    except Exception as e:
        print(f"Telegram error: {e}")

# ======================================================
# FUNCIONES DE BYBIT: SALDOS, CONVERSIONES, DUPLOS
# ======================================================

def obtener_saldos():
    """Devuelve dict con saldos USDT, BTC, SOL."""
    result = _bybit_request("GET", "/v5/account/wallet-balance", params={"accountType": "UNIFIED"})
    saldos = {"USDT": 0.0, "BTC": 0.0, "SOL": 0.0}
    for coin_info in result.get("list", []):
        for coin in coin_info.get("coin", []):
            if coin["coin"] in saldos:
                saldos[coin["coin"]] = float(coin["walletBalance"])
    return saldos

def convertir_a_usdt(moneda, cantidad):
    """Vende la cantidad al mercado spot y devuelve USDT recibidos (estimado)."""
    if cantidad <= 0:
        return 0
    symbol = f"{moneda}USDT"
    # Obtener precio para estimación
    ticker = _bybit_request("GET", "/v5/market/tickers", params={"category": "spot", "symbol": symbol})
    precio = float(ticker["list"][0]["lastPrice"])
    usdt_esperado = cantidad * precio * (1 - SPOT_COMMISSION)
    body = {
        "category": "spot",
        "symbol": symbol,
        "side": "Sell",
        "orderType": "Market",
        "qty": str(cantidad),
        "timeInForce": "IOC"
    }
    try:
        _bybit_request("POST", "/v5/order/create", body=body)
        return usdt_esperado
    except Exception as e:
        telegram_mensaje(f"❌ Error convirtiendo {moneda}: {e}")
        return 0

def obtener_productos_dual(symbol, option_side, duration_days=1):
    """
    Obtiene productos Dual Asset para un símbolo (ej: BTCUSDT), tipo 'BuyLow' o 'SellHigh',
    y duración exacta en días. Devuelve el de mayor APR o None.
    """
    try:
        result = _bybit_request("GET", "/v5/earn/dual-asset/products", params={"symbol": symbol})
        mejores = []
        for item in result.get("list", []):
            if item.get("optionSide") != option_side:
                continue
            duration = int(item.get("duration", 0))
            if duration != duration_days:
                continue
            apy = float(item.get("apy", 0))
            mejores.append({
                "product_id": item["productId"],
                "apy": apy,
                "target_price": float(item["targetPrice"]),
                "duration": duration,
                "currency": item["currency"]
            })
        if mejores:
            return max(mejores, key=lambda x: x["apy"])
        return None
    except Exception as e:
        telegram_mensaje(f"⚠️ Error obteniendo productos {symbol} {option_side}: {e}")
        return None

def suscribir_dual(producto, cantidad_usdt):
    """Suscribe cantidad de USDT al producto."""
    body = {
        "productId": producto["product_id"],
        "amount": str(cantidad_usdt),
        "currency": "USDT"
    }
    result = _bybit_request("POST", "/v5/earn/dual-asset/subscribe", body=body)
    return result

def obtener_suscripciones_activas():
    """Devuelve lista de suscripciones en estado 'processing'."""
    result = _bybit_request("GET", "/v5/earn/dual-asset/orders", params={"status": "processing"})
    return result.get("list", [])

def esperar_hasta_vencimiento(expiry_timestamp):
    """
    Espera activamente hasta el timestamp de vencimiento (en ms).
    Hace pings periódicos para evitar hibernación en Railway.
    """
    expiry = datetime.fromtimestamp(expiry_timestamp / 1000, tz=timezone.utc)
    while True:
        ahora = datetime.now(timezone.utc)
        if ahora >= expiry:
            break
        restante = (expiry - ahora).total_seconds()
        if restante > 60:
            telegram_mensaje(f"⏳ Esperando vencimiento. Restan {int(restante//3600)}h {int((restante%3600)//60)}m.")
        # Espera en intervalos cortos pero hace ping para mantener activo
        tiempo_espera = min(restante, KEEP_AWAKE_INTERVAL_SECONDS)
        time.sleep(tiempo_espera)
        # Ping a Bybit para generar tráfico
        try:
            _bybit_request("GET", "/v5/market/tickers", params={"category": "spot", "symbol": "BTCUSDT"})
        except:
            pass

# ======================================================
# CICLO PRINCIPAL
# ======================================================

def ejecutar_ciclo():
    telegram_mensaje("🔄 *Iniciando ciclo de análisis y selección de Duplo*")
    
    # 1. Obtener saldos actuales
    saldos = obtener_saldos()
    usdt = saldos["USDT"]
    btc = saldos["BTC"]
    sol = saldos["SOL"]
    telegram_mensaje(f"💰 Saldos: USDT={usdt:.2f}, BTC={btc:.8f}, SOL={sol:.2f}")
    
    # 2. Convertir cualquier BTC o SOL a USDT
    if btc > 0:
        telegram_mensaje(f"🔄 Convirtiendo {btc:.8f} BTC → USDT")
        usdt += convertir_a_usdt("BTC", btc)
    if sol > 0:
        telegram_mensaje(f"🔄 Convirtiendo {sol:.2f} SOL → USDT")
        usdt += convertir_a_usdt("SOL", sol)
    
    if usdt <= 10:
        telegram_mensaje("⚠️ Saldo USDT insuficiente (<10). Esperando...")
        return
    
    # 3. Analizar cada activo con IA para elegir el mejor par y tipo
    mejor_opcion = None
    mejor_decision = None
    mejor_razones = []
    mejor_imagen = None
    mejor_df = None
    
    for symbol in ASSET_PAIRS:
        try:
            df = obtener_velas(symbol, TIMEFRAME, LOOKBACK_DAYS)
            df = calcular_indicadores(df)
            decision, razones, img_b64 = ia_analizar_grafico(symbol, df)
            # Enviar gráfico y decisión al Telegram
            telegram_mensaje(f"📊 *Análisis para {symbol}*\nDecisión IA: {decision}\nRazones: " + "\n".join(razones), img_b64)
            if decision != "HOLD":
                # Calcular un "score" simple: por ahora elegimos la que no sea HOLD (priorizamos la que aparezca)
                # Podríamos mejorar con métricas pero elegimos la primera que decida operar
                if mejor_opcion is None:
                    mejor_opcion = symbol
                    mejor_decision = decision
                    mejor_razones = razones
                    mejor_imagen = img_b64
                    mejor_df = df
        except Exception as e:
            telegram_mensaje(f"⚠️ Error analizando {symbol}: {e}")
    
    if mejor_opcion is None:
        telegram_mensaje("🤷 IA recomienda HOLD para todos los activos. No se opera.")
        return
    
    # 4. Buscar producto Dual Asset en Bybit para ese símbolo, misma decisión, duración 1 día
    option_side = "BuyLow" if mejor_decision == "BUY_LOW" else "SellHigh"
    producto = obtener_productos_dual(mejor_opcion, option_side, duration_days=1)
    if not producto:
        telegram_mensaje(f"❌ No se encontró producto Dual de 1 día para {mejor_opcion} con {option_side}")
        return
    
    # 5. Suscribir todo el USDT disponible
    cantidad_invertir = usdt * INVEST_PERCENT
    if cantidad_invertir < 10:
        telegram_mensaje(f"⚠️ Cantidad muy pequeña ({cantidad_invertir:.2f} USDT). Mínimo 10.")
        return
    
    try:
        resultado = suscribir_dual(producto, cantidad_invertir)
        # Obtener timestamp de expiración (Bybit lo devuelve en la respuesta o consultando la orden)
        # Simulamos: 1 día desde ahora
        expiry_ms = int((datetime.now(timezone.utc) + timedelta(days=1)).timestamp() * 1000)
        telegram_mensaje(
            f"✅ *SUSCRIPCIÓN REALIZADA*\n"
            f"📌 Activo: {mejor_opcion}\n"
            f"📈 Tipo: {mejor_decision}\n"
            f"💰 Monto: {cantidad_invertir:.2f} USDT\n"
            f"🎯 APR: {producto['apy']}%\n"
            f"⏳ Vence en 1 día\n"
            f"🧠 IA: {', '.join(mejor_razones)}"
        )
        # Esperar activamente hasta el vencimiento
        esperar_hasta_vencimiento(expiry_ms)
        # Una vez vencido, el ciclo se repite (la próxima iteración del bucle principal)
    except Exception as e:
        telegram_mensaje(f"❌ Error en suscripción: {e}")

# ======================================================
# BUCLE PRINCIPAL CON KEEP-AWAKE
# ======================================================

def run_bot():
    telegram_mensaje("🤖 *BOT DUAL ASSET CON IA INICIADO* - Modo 1 día")
    while True:
        try:
            # Verificar si hay suscripciones activas actualmente
            activas = obtener_suscripciones_activas()
            if activas:
                # Esperar hasta que venzan (ya hay un proceso de espera, pero si el bot se reinicia, esto evita duplicar)
                # Tomamos la que vence antes
                vencimientos = []
                for ord in activas:
                    expiry = datetime.fromtimestamp(int(ord["expiryTime"]) / 1000, tz=timezone.utc)
                    vencimientos.append(expiry)
                if vencimientos:
                    proximo = min(vencimientos)
                    ahora = datetime.now(timezone.utc)
                    if proximo > ahora:
                        restante = (proximo - ahora).total_seconds()
                        telegram_mensaje(f"⏳ Hay {len(activas)} suscripción(es) activa(s). Próximo vencimiento en {restante/3600:.1f}h. Esperando...")
                        # Esperar pero con pings
                        while datetime.now(timezone.utc) < proximo:
                            time.sleep(min(KEEP_AWAKE_INTERVAL_SECONDS, (proximo - datetime.now(timezone.utc)).total_seconds()))
                            # Ping
                            try:
                                _bybit_request("GET", "/v5/market/tickers", params={"category": "spot", "symbol": "BTCUSDT"})
                            except:
                                pass
                        continue
            # No hay activas, ejecutar ciclo completo
            ejecutar_ciclo()
            # Pequeña pausa antes de reiniciar para no saturar
            time.sleep(60)
        except Exception as e:
            telegram_mensaje(f"🚨 ERROR en bucle principal: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_bot()
