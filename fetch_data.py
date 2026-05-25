import os
import time
import pandas as pd
from pybit.unified_trading import HTTP
from datetime import datetime

def fetch_klines(symbol, interval, start_date, end_date, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    session = HTTP(testnet=False)
    start_ms = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    end_ms = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)
    all_klines = []
    current_start = start_ms
    print(f"Descargando {symbol} {interval}m desde {start_date} hasta {end_date}")
    while current_start < end_ms:
        resp = session.get_kline(
            category="linear",
            symbol=symbol,
            interval=str(interval),
            start=current_start,
            end=end_ms,
            limit=1000
        )
        if resp['retCode'] != 0:
            print(f"Error: {resp}")
            break
        klines = resp['result']['list']
        if not klines:
            break
        all_klines.extend(klines)
        current_start = int(klines[-1][0]) + 1
        time.sleep(0.5)
    if not all_klines:
        return None
    df = pd.DataFrame(all_klines, columns=['timestamp','open','high','low','close','volume','turnover'])
    df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
    for col in ['open','high','low','close','volume']:
        df[col] = df[col].astype(float)
    df = df.sort_values('timestamp')
    filename = f"{data_dir}/{symbol}_{interval}m_{start_date}_{end_date}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ Datos guardados en {filename}")
    return filename

if __name__ == "__main__":
    # Ejemplo: descargar 10m y 60m
    fetch_klines("BTCUSDT", 10, "2024-01-01", "2024-12-31")
    fetch_klines("BTCUSDT", 60, "2024-01-01", "2024-12-31")
