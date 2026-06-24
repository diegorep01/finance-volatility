"""
Fase 2 — Filtro de contexto de tendencia.

Hipótesis: una caída en tendencia alcista es más probable que sea
"ruido" y se recupere rápido. Una caída en tendencia bajista puede
ser una señal real de deterioro y tardar mucho más (o no volver).

Indicadores de contexto:
  A. Precio > SMA200  → tendencia alcista de largo plazo
  B. SMA50 > SMA200   → golden cross (tendencia media confirmada)
  C. Retorno 20d > 0  → momentum positivo reciente

Compara tasas de recuperación entre:
  - Uptrend  (A + B + C todos positivos)
  - Neutral  (mezcla)
  - Downtrend (A + B + C todos negativos)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from download_data import cargar
from config import TICKERS

OUTPUT_DIR = Path('output')

BUCKETS = [
    ('Leve    (-1% a -3%)',     -3.0,  -1.0),
    ('Moderada(-3% a -5%)',    -5.0,  -3.0),
    ('Fuerte  (-5% a -10%)',  -10.0,  -5.0),
    ('Extrema (> -10%)',      -100.0, -10.0),
]

RECOVERY_DAYS = [5, 20, 60, 252]   # corto, medio, largo, 1 año


def enriquecer(df):
    """Añade indicadores de tendencia al DataFrame de precios."""
    c = df['Close'].copy()
    df = df.copy()
    df['sma50']    = c.rolling(50).mean()
    df['sma200']   = c.rolling(200).mean()
    df['ret_1d']   = c.pct_change() * 100
    df['ret_20d']  = c.pct_change(20) * 100

    # Señales de contexto
    df['sobre_sma200']   = c > df['sma200']                     # A
    df['golden_cross']   = df['sma50'] > df['sma200']           # B
    df['momentum_pos']   = df['ret_20d'] > 0                    # C

    # Contexto sintético: suma de señales alcistas (0-3)
    df['n_alcistas'] = (df['sobre_sma200'].astype(int) +
                        df['golden_cross'].astype(int) +
                        df['momentum_pos'].astype(int))

    return df


def contexto_label(n):
    if n == 3:   return 'Uptrend claro   (3/3)'
    if n == 2:   return 'Tendencia mixta (2/3)'
    if n == 1:   return 'Tendencia mixta (1/3)'
    return             'Downtrend claro (0/3)'


def dias_hasta_recuperacion(close_series, idx_drop):
    if idx_drop == 0:
        return np.inf
    precio_pre = close_series.iloc[idx_drop - 1]
    futuros    = close_series.iloc[idx_drop + 1:]
    recuperados = futuros[futuros >= precio_pre]
    if recuperados.empty:
        return np.inf
    return (recuperados.index[0] - close_series.index[idx_drop]).days


def recoger_eventos(tickers, low_pct, high_pct):
    rows = []
    for ticker in tickers:
        try:
            df    = enriquecer(cargar(ticker))
            close = df['Close']
            rets  = df['ret_1d']
            mask  = (rets < high_pct) & (rets >= low_pct)
            idxs  = np.where(mask.values)[0]

            for idx in idxs:
                row = df.iloc[idx]
                if pd.isna(row['sma200']):   # sin suficiente histórico
                    continue
                dias = dias_hasta_recuperacion(close, idx)
                rows.append({
                    'ticker':       ticker,
                    'fecha':        df.index[idx],
                    'ret_1d':       row['ret_1d'],
                    'n_alcistas':   int(row['n_alcistas']),
                    'sobre_sma200': bool(row['sobre_sma200']),
                    'golden_cross': bool(row['golden_cross']),
                    'momentum_pos': bool(row['momentum_pos']),
                    'dias_rec':     dias,
                })
        except Exception as e:
            print(f'  Error {ticker}: {e}')
    return pd.DataFrame(rows)


def tasa_recuperacion(df, max_dias):
    if df.empty:
        return np.nan
    return (df['dias_rec'] <= max_dias).mean() * 100


def imprimir_bloque(nombre_bucket, df_ev):
    n = len(df_ev)
    if n == 0:
        return

    # Agrupar por contexto
    grupos = {
        'Uptrend claro   (3/3)':  df_ev[df_ev['n_alcistas'] == 3],
        'Tendencia mixta (2/3)':  df_ev[df_ev['n_alcistas'] == 2],
        'Tendencia mixta (1/3)':  df_ev[df_ev['n_alcistas'] == 1],
        'Downtrend claro (0/3)':  df_ev[df_ev['n_alcistas'] == 0],
    }

    print(f'\n  BUCKET: {nombre_bucket}  (N total = {n})')
    print(f'  {"Contexto":<26} {"N":>5}  {"5d":>7}  {"20d":>7}  {"60d":>7}  {"1 año":>7}  {"Nunca":>7}')
    print(f'  {"-"*68}')

    for label, sub in grupos.items():
        ns = len(sub)
        if ns == 0:
            continue
        t5   = tasa_recuperacion(sub, 5)
        t20  = tasa_recuperacion(sub, 20)
        t60  = tasa_recuperacion(sub, 60)
        t252 = tasa_recuperacion(sub, 252)
        nunca = (sub['dias_rec'] == np.inf).mean() * 100
        print(f'  {label:<26} {ns:>5}  {t5:>6.1f}%  {t20:>6.1f}%  {t60:>6.1f}%  {t252:>6.1f}%  {nunca:>6.1f}%')

    # Fila total para referencia
    sub = df_ev
    t5   = tasa_recuperacion(sub, 5)
    t20  = tasa_recuperacion(sub, 20)
    t60  = tasa_recuperacion(sub, 60)
    t252 = tasa_recuperacion(sub, 252)
    nunca = (sub['dias_rec'] == np.inf).mean() * 100
    print(f'  {"─"*68}')
    print(f'  {"TOTAL (sin filtro)":<26} {n:>5}  {t5:>6.1f}%  {t20:>6.1f}%  {t60:>6.1f}%  {t252:>6.1f}%  {nunca:>6.1f}%')


def main():
    print('\n' + '='*72)
    print('  FASE 2 — FILTRO DE CONTEXTO DE TENDENCIA')
    print(f'  Universo: {len(TICKERS)} tickers  |  Período: 2015–2026')
    print('='*72)
    print('  Columnas: % recuperado en 5d / 20d / 60d / 1 año / % que NUNCA vuelve')
    print('  Contexto: precio>SMA200 (A) + SMA50>SMA200 (B) + ret20d>0 (C)\n')

    todos = []
    for nombre, low, high in BUCKETS:
        print(f'  Procesando {nombre}...')
        df_ev = recoger_eventos(TICKERS, low, high)
        df_ev['bucket'] = nombre
        todos.append(df_ev)
        imprimir_bloque(nombre, df_ev)

    df_all = pd.concat(todos, ignore_index=True)
    df_all.to_csv(OUTPUT_DIR / 'phase2_trend_context.csv', index=False)

    # ── Resumen ejecutivo ──
    print('\n\n' + '='*72)
    print('  RESUMEN EJECUTIVO — ¿Cuánto mejora filtrar por uptrend?')
    print('='*72)
    print(f'  {"Bucket":<26}  {"Sin filtro 20d":>14}  {"Solo uptrend 20d":>16}  {"Mejora":>8}')
    print(f'  {"-"*68}')
    for nombre, low, high in BUCKETS:
        sub = df_all[df_all['bucket'] == nombre]
        if sub.empty:
            continue
        base   = tasa_recuperacion(sub, 20)
        up     = tasa_recuperacion(sub[sub['n_alcistas'] == 3], 20)
        delta  = up - base if not np.isnan(up) else np.nan
        s_up   = f'{up:.1f}%' if not np.isnan(up) else '—'
        s_delta= f'+{delta:.1f}pp' if not np.isnan(delta) else '—'
        print(f'  {nombre:<26}  {base:>13.1f}%  {s_up:>16}  {s_delta:>8}')

    print(f'\n  CSV guardado: output/phase2_trend_context.csv')
    print('='*72 + '\n')


if __name__ == '__main__':
    main()
