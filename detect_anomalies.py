"""
Detecta caídas anómalas en los datos de precio.

Criterios (se aplican los DOS):
  1. Retorno diario < DROP_THRESHOLD_PCT  (caída absoluta significativa)
  2. Retorno diario < media_móvil - ZSCORE_THRESHOLD * std_móvil  (anomalía estadística)

También añade contexto de volumen y tendencia previa.
"""

import pandas as pd
import numpy as np
from download_data import cargar
from config import (DROP_THRESHOLD_PCT, ZSCORE_THRESHOLD,
                    ROLLING_WINDOW, TICKERS)


def detectar(ticker, df=None):
    if df is None:
        df = cargar(ticker)

    # Retorno diario en %
    df = df.copy()
    df['ret'] = df['Close'].pct_change() * 100

    # Media y std móviles del retorno
    df['ret_mean'] = df['ret'].rolling(ROLLING_WINDOW).mean()
    df['ret_std']  = df['ret'].rolling(ROLLING_WINDOW).std()

    # Z-score del retorno
    df['zscore'] = (df['ret'] - df['ret_mean']) / df['ret_std']

    # Tendencia previa: retorno acumulado de los últimos 20 días
    df['trend_20d'] = df['Close'].pct_change(20) * 100

    # Volumen relativo respecto a media de 20 días
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    else:
        df['vol_ratio'] = np.nan

    # Filtro doble: caída absoluta Y anomalía estadística
    mask = (
        (df['ret'] < DROP_THRESHOLD_PCT) &
        (df['zscore'] < ZSCORE_THRESHOLD)
    )
    eventos = df[mask].copy()
    eventos['ticker'] = ticker

    cols = ['ticker', 'ret', 'zscore', 'trend_20d', 'vol_ratio', 'Close']
    return eventos[[c for c in cols if c in eventos.columns]]


def detectar_todos(tickers=TICKERS):
    frames = []
    for t in tickers:
        try:
            ev = detectar(t)
            frames.append(ev)
        except FileNotFoundError:
            print(f'  Sin datos: {t}')
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()


if __name__ == '__main__':
    from config import TICKERS
    eventos = detectar_todos(TICKERS)
    print(f'\nTotal de caídas anómalas detectadas: {len(eventos)}')
    print(eventos.tail(20).to_string())
