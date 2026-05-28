import os
import time
import logging
import json
from datetime import datetime, timedelta
from pybit.unified_trading import HTTP
import requests

# ================= CONFIGURACIÓN =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BYBIT_API_KEY = os.getenv("BYBIT_API_KEY")
BYBIT_API_SECRET = os.getenv("BYBIT_API_SECRET")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
session = HTTP(testnet=False, api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET)

SYMBOL = "BTCUSDT"
INVESTMENT = "10" # USDT fijos

# Estado global
last_ai_analysis = 0
grid_info = {"id": None, "last_price": 0}

def send_telegram(msg):
    requests.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", 
                  data={"chat_id": TELEGRAM_CHAT_ID, "text": msg})

def get_balance():
    resp = session.get_wallet_balance(accountType="UNIFIED", coin="USDT")
    return resp['result']['list'][0]['coin'][0]['walletBalance']

def run_ai_analysis():
    """Simula el análisis de IA y retorna niveles"""
    logger.info("Analizando mercado con IA...")
    # Aquí iría tu integración real con OpenAI/Gemini
    # Retornamos valores calculados (ejemplo dinámico)
    return 85000.0, 105000.0 # Soporte, Resistencia

def manage_grid():
    global last_ai_analysis, grid_info
    
    current_time = time.time()
    
    # 1. Análisis IA cada 5 horas
    if current_time - last_ai_analysis >= 18000:
        soporte, resistencia = run_ai_analysis()
        
        # Cerrar grid anterior si existe
        if grid_info["id"]:
            session.cancel_grid_order(category="spot", orderId=grid_info["id"])
            
        # Crear nuevo grid
        resp = session.create_grid_order(
            category="spot",
            symbol=SYMBOL,
            gridType="Neutral",
            qty=INVESTMENT,
            lowerPrice=str(soporte),
            upperPrice=str(resistencia),
            gridCount=20
        )
        grid_info["id"] = resp['result']['orderId']
        last_ai_analysis = current_time
        send_telegram(f"🤖 IA: Nuevo Grid creado para {SYMBOL}. Rango: {soporte}-{resistencia}")

    # 2. Reporte horario
    balance = get_balance()
    msg = f"📊 Reporte Horario:\nBalance Total: {balance} USDT\nInversión Grid: {INVESTMENT} USDT\nEstado: Operando"
    send_telegram(msg)

def main():
    logger.info("Sistema Iniciado...")
    while True:
        try:
            manage_grid()
        except Exception as e:
            logger.error(f"Error: {e}")
        time.sleep(3600) # Ciclo de 1 hora

if __name__ == "__main__":
    main()
