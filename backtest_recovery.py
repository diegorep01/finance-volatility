"""
Para cada caída anómala detectada, calcula si el precio se recuperó
en los horizontes definidos y en qué medida.

Output:
  - tasa de acierto (% de eventos que se recuperaron >= RECOVERY_RATIO del drop)
  - retorno promedio a cada horizonte
  - distribución de recuperaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from download_data import cargar
from detect_anomalies import detectar
from config import TICKERS, RECOVERY_HORIZONS, RECOVERY_RATIO

OUTPUT_DIR = Path('output')


def analizar_ticker(ticker):
    df = cargar(ticker)
    eventos = detectar(ticker, df)
    if eventos.empty:
        return pd.DataFrame()

    precios = df['Close']
    resultados = []

    for fecha_drop in eventos.index:
        idx = precios.index.get_loc(fecha_drop)
        precio_pre  = precios.iloc[idx - 1] if idx > 0 else np.nan
        precio_drop = precios.iloc[idx]
        drop_abs    = precio_drop - precio_pre  # negativo

        fila = {
            'ticker':      ticker,
            'fecha':       fecha_drop,
            'ret_pct':     eventos.loc[fecha_drop, 'ret'] if fecha_drop in eventos.index else np.nan,
            'zscore':      eventos.loc[fecha_drop, 'zscore'] if 'zscore' in eventos.columns else np.nan,
            'precio_drop': round(precio_drop, 2),
        }

        for h in RECOVERY_HORIZONS:
            idx_h = idx + h
            if idx_h < len(precios):
                precio_h    = precios.iloc[idx_h]
                ret_h       = (precio_h - precio_drop) / abs(precio_drop) * 100
                recuperado  = (precio_h - precio_drop) >= abs(drop_abs) * RECOVERY_RATIO
            else:
                ret_h, recuperado = np.nan, np.nan
            fila[f'ret_{h}d']    = round(ret_h, 2) if not np.isnan(ret_h) else np.nan
            fila[f'recup_{h}d']  = recuperado

        resultados.append(fila)

    return pd.DataFrame(resultados)


def analizar_todos(tickers=TICKERS):
    frames = []
    for t in tickers:
        try:
            r = analizar_ticker(t)
            if not r.empty:
                frames.append(r)
        except Exception as e:
            print(f'  Error {t}: {e}')
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def resumen(df_res):
    if df_res.empty:
        print('Sin eventos.')
        return

    print(f'\n{"="*55}')
    print(f'  FINANCE VOLATILITY CHECK — BACKTESTING FASE 1')
    print(f'{"="*55}')
    print(f'  Universo:        {df_res["ticker"].nunique()} tickers')
    print(f'  Eventos totales: {len(df_res)}')
    print(f'  Período:         {df_res["fecha"].min().date()} → {df_res["fecha"].max().date()}')
    print(f'  Caída promedio:  {df_res["ret_pct"].mean():.2f}%')
    print(f'\n  Tasa de recuperación (>= {int(RECOVERY_RATIO*100)}% del drop):')
    print(f'  {"Horizonte":<12} {"Aciertos":>10} {"Tasa":>8} {"Ret. prom.":>12}')
    print(f'  {"-"*44}')
    for h in RECOVERY_HORIZONS:
        col_rec = f'recup_{h}d'
        col_ret = f'ret_{h}d'
        if col_rec not in df_res.columns:
            continue
        sub = df_res[df_res[col_rec].notna()]
        if sub.empty:
            continue
        tasa    = sub[col_rec].mean() * 100
        aciertos = sub[col_rec].sum()
        ret_avg = sub[col_ret].mean()
        print(f'  {h} días{"":<7} {int(aciertos):>10} {tasa:>7.1f}% {ret_avg:>+10.2f}%')
    print(f'{"="*55}\n')


def graficar_distribucion(df_res, horizonte=5):
    col = f'ret_{horizonte}d'
    if col not in df_res.columns:
        return
    datos = df_res[col].dropna()
    plt.figure(figsize=(10, 5))
    plt.hist(datos, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Sin cambio')
    plt.axvline(datos.mean(), color='green', linestyle='-', linewidth=1.5,
                label=f'Media: {datos.mean():.2f}%')
    plt.title(f'Distribución de retornos a {horizonte} días tras caída anómala')
    plt.xlabel('Retorno (%)')
    plt.ylabel('Frecuencia')
    plt.legend()
    plt.tight_layout()
    ruta = OUTPUT_DIR / f'distribucion_{horizonte}d.png'
    plt.savefig(ruta, dpi=120)
    plt.close()
    print(f'  Gráfico guardado: {ruta}')


if __name__ == '__main__':
    print('Analizando recuperaciones...')
    df_res = analizar_todos()
    if not df_res.empty:
        df_res.to_csv(OUTPUT_DIR / 'resultados_backtest.csv', index=False)
        resumen(df_res)
        for h in RECOVERY_HORIZONS:
            graficar_distribucion(df_res, h)
        print(f'Resultados completos: output/resultados_backtest.csv')
