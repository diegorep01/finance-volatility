"""
Descarga datos OHLCV históricos para todos los tickers del universo
y los guarda como CSV en data/.
"""

import yfinance as yf
import pandas as pd
from pathlib import Path
from config import TICKERS, START_DATE, END_DATE

DATA_DIR = Path('data')

def descargar(tickers=TICKERS, start=START_DATE, end=END_DATE):
    descargados, fallidos = [], []
    for ticker in tickers:
        try:
            df = yf.download(ticker, start=start, end=end,
                             auto_adjust=True, progress=False)
            if df.empty:
                fallidos.append(ticker)
                continue
            # yfinance puede devolver MultiIndex si se descarga en bloque
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(DATA_DIR / f'{ticker}.csv')
            descargados.append(ticker)
            print(f'  ✓ {ticker}: {len(df)} días')
        except Exception as e:
            fallidos.append(ticker)
            print(f'  ✗ {ticker}: {e}')
    print(f'\nDescargados: {len(descargados)} | Fallidos: {len(fallidos)}')
    if fallidos:
        print(f'Fallidos: {fallidos}')
    return descargados

def cargar(ticker):
    path = DATA_DIR / f'{ticker}.csv'
    if not path.exists():
        raise FileNotFoundError(f'No hay datos para {ticker}. Ejecuta download_data.py primero.')
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df

if __name__ == '__main__':
    print(f'Descargando {len(TICKERS)} tickers desde {START_DATE}...\n')
    descargar()
