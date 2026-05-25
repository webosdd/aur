import os
import pandas as pd
from fetch_data import fetch_klines
from backtest_engine import BacktestEngine, calcular_indicadores
from telegram_sender import enviar_resumen

def main():
    # Parámetros (puedes cambiarlos fácilmente)
    START_DATE = "2024-01-01"
    END_DATE = "2024-06-01"
    SYMBOL = "BTCUSDT"
    INTERVAL_LTF = 10
    INTERVAL_HTF = 60
    INITIAL_BALANCE = 1000.0

    # 1. Descargar datos si no existen localmente
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    ltf_file = f"{data_dir}/{SYMBOL}_{INTERVAL_LTF}m_{START_DATE}_{END_DATE}.csv"
    htf_file = f"{data_dir}/{SYMBOL}_{INTERVAL_HTF}m_{START_DATE}_{END_DATE}.csv"

    if not os.path.exists(ltf_file):
        fetch_klines(SYMBOL, INTERVAL_LTF, START_DATE, END_DATE, data_dir)
    if not os.path.exists(htf_file):
        fetch_klines(SYMBOL, INTERVAL_HTF, START_DATE, END_DATE, data_dir)

    # 2. Cargar datos y calcular indicadores
    df_ltf = pd.read_csv(ltf_file, parse_dates=['timestamp'], index_col='timestamp')
    df_htf = pd.read_csv(htf_file, parse_dates=['timestamp'], index_col='timestamp')
    df_ltf = calcular_indicadores(df_ltf)
    df_htf = calcular_indicadores(df_htf)

    # 3. Ejecutar backtest
    engine = BacktestEngine(initial_balance=INITIAL_BALANCE)
    trades, signals = engine.run(df_ltf, df_htf)

    # 4. Enviar resultados a Telegram
    enviar_resumen(trades, signals, INITIAL_BALANCE)

if __name__ == "__main__":
    main()
