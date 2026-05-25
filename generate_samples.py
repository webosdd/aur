import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from datetime import timedelta

def calcular_indicadores(df):
    df = df.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    return df

def generar_grafico(df, titulo, entry_price=None):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [3,1,1]})
    ax1.plot(df.index, df['close'], 'black', linewidth=1, label='Close')
    if 'ema20' in df:
        ax1.plot(df.index, df['ema20'], 'orange', linewidth=1, label='EMA20')
    if 'ema50' in df:
        ax1.plot(df.index, df['ema50'], 'blue', linewidth=1, label='EMA50')
    if entry_price:
        ax1.axhline(entry_price, color='green', linestyle='--', label='Entry')
    ax1.legend()
    ax1.set_title(titulo)
    if 'rsi' in df:
        ax2.plot(df.index, df['rsi'], 'purple')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_ylabel('RSI')
    if 'macd' in df:
        ax3.plot(df.index, df['macd'], label='MACD')
        ax3.plot(df.index, df['macd_signal'], label='Signal')
        ax3.bar(df.index, df['macd_diff'], label='Histogram')
        ax3.legend()
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf

def detectar_movimientos(df, lookahead=10, umbral_pct=1.0):
    movimientos = []
    for i in range(len(df) - lookahead):
        precio_actual = df['close'].iloc[i]
        precio_futuro = df['close'].iloc[i+lookahead]
        cambio = (precio_futuro - precio_actual) / precio_actual * 100
        if cambio > umbral_pct:
            movimientos.append((df.index[i], 'BUY', cambio))
        elif cambio < -umbral_pct:
            movimientos.append((df.index[i], 'SELL', -cambio))
    return movimientos

def guardar_muestras(df, movimientos, output_dir='samples', max_samples=100, window_velas=80):
    os.makedirs(output_dir, exist_ok=True)
    muestras = []
    for idx, direccion, cambio in movimientos[:max_samples]:
        # Tomar ventana de velas anteriores al momento
        start_idx = max(0, df.index.get_loc(idx) - window_velas)
        end_idx = df.index.get_loc(idx)
        df_window = df.iloc[start_idx:end_idx+1].copy()
        if len(df_window) < 30:
            continue
        img = generar_grafico(df_window, f"{direccion} {cambio:.1f}%", entry_price=df.loc[idx]['close'])
        # Guardar imagen
        img_path = f"{output_dir}/{idx.strftime('%Y%m%d_%H%M')}_{direccion}_{cambio:.1f}.png"
        with open(img_path, 'wb') as f:
            f.write(img.getvalue())
        muestras.append({
            'timestamp': idx,
            'direction': direccion,
            'cambio': cambio,
            'image_path': img_path
        })
    return muestras

if __name__ == "__main__":
    # Cargar datos ya descargados
    df_10m = pd.read_csv('data/BTCUSDT_10m_2024-01-01_2024-12-31.csv', parse_dates=['timestamp'], index_col='timestamp')
    df_10m = calcular_indicadores(df_10m)
    movs = detectar_movimientos(df_10m, lookahead=10, umbral_pct=1.2)
    muestras = guardar_muestras(df_10m, movs, max_samples=50)
    print(f"Generadas {len(muestras)} muestras")
