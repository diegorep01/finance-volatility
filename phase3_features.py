"""
Fase 3 — Features granulares para cada caída anómala.

Nuevas variables añadidas sobre cada evento:
  1. vol_ratio      — volumen del día vs media 20d (pánico = vol alto)
  2. rsi_14         — RSI de 14 días en el momento de la caída
  3. dias_consec    — días consecutivos bajando antes del evento
  4. tipo_activo    — ETF (casi siempre vuelve) vs ACCION (puede quebrar)

Objetivo: construir el dataset enriquecido que usará XGBoost en Fase 4.
Target: ¿se recuperó al precio previo en ≤ 20 días?
"""

import pandas as pd
import numpy as np
from pathlib import Path
from download_data import cargar
from config import TICKERS

OUTPUT_DIR = Path('output')

# ── Clasificación de activos ──────────────────────────────────────────────────
ETFS = {
    'SPY', 'QQQ', 'DIA', 'IWM',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLC',
    'GLD', 'SLV', 'USO', 'UNG',
}

BUCKETS_MAGNITUD = [
    ('Leve (-1% a -3%)',      -3.0,  -1.0),
    ('Moderada (-3% a -5%)',  -5.0,  -3.0),
    ('Fuerte (-5% a -10%)',  -10.0,  -5.0),
    ('Extrema (>-10%)',      -100.0, -10.0),
]

DROP_THRESHOLD = -3.0   # misma lógica que Fase 1/2
ZSCORE_THRESH  = -2.0
ROLLING_WIN    = 60
RECOVERY_DAYS  = [5, 10, 20, 60]


# ── Indicadores técnicos ──────────────────────────────────────────────────────

def calcular_rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def dias_consecutivos_bajando(rets_array, idx):
    """Días seguidos con retorno negativo ANTES del evento (sin contar el día del evento)."""
    count = 0
    i = idx - 1
    while i >= 0 and rets_array[i] < 0:
        count += 1
        i -= 1
    return count


def bucket_magnitud(ret):
    for nombre, low, high in BUCKETS_MAGNITUD:
        if low <= ret < high:
            return nombre
    return 'Otra'


def bucket_volumen(vol_ratio):
    if np.isnan(vol_ratio):   return 'Sin datos'
    if vol_ratio < 0.8:       return 'Bajo (<0.8x)'
    if vol_ratio < 1.5:       return 'Normal (0.8-1.5x)'
    if vol_ratio < 2.5:       return 'Alto (1.5-2.5x)'
    return                           'Muy alto (>2.5x)'


def bucket_rsi(rsi):
    if np.isnan(rsi):  return 'Sin datos'
    if rsi < 30:       return 'Sobrevendido (<30)'
    if rsi < 70:       return 'Neutral (30-70)'
    return                    'Sobrecomprado (>70)'


def bucket_streak(dias):
    if dias == 0:   return '0 — caída aislada'
    if dias <= 2:   return '1-2 días bajando'
    if dias <= 4:   return '3-4 días bajando'
    return                 '5+ días bajando'


# ── Procesamiento por ticker ──────────────────────────────────────────────────

def enriquecer_ticker(ticker):
    df = cargar(ticker).copy()

    close  = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else None

    # Retorno diario
    df['ret_1d'] = close.pct_change() * 100

    # Z-score (mismo que Fase 1)
    df['ret_mean'] = df['ret_1d'].rolling(ROLLING_WIN).mean()
    df['ret_std']  = df['ret_1d'].rolling(ROLLING_WIN).std()
    df['zscore']   = (df['ret_1d'] - df['ret_mean']) / df['ret_std']

    # Tendencia (de Fase 2)
    df['sma50']  = close.rolling(50).mean()
    df['sma200'] = close.rolling(200).mean()
    df['ret_20d'] = close.pct_change(20) * 100
    df['n_alcistas'] = (
        (close > df['sma200']).astype(int) +
        (df['sma50'] > df['sma200']).astype(int) +
        (df['ret_20d'] > 0).astype(int)
    )

    # Volumen relativo
    if volume is not None and volume.sum() > 0:
        df['vol_ratio'] = volume / volume.rolling(20).mean()
    else:
        df['vol_ratio'] = np.nan

    # RSI 14
    df['rsi_14'] = calcular_rsi(close, 14)

    # Detección de eventos (mismo filtro doble que Fase 1)
    mask = (
        (df['ret_1d'] < DROP_THRESHOLD) &
        (df['zscore'] < ZSCORE_THRESH)
    )
    idxs = np.where(mask.values)[0]

    rows = []
    rets_arr = df['ret_1d'].values

    for idx in idxs:
        if idx == 0:
            continue
        row   = df.iloc[idx]
        fecha = df.index[idx]

        if pd.isna(row['sma200']):
            continue

        # Días hasta recuperación total al precio previo
        precio_pre = close.iloc[idx - 1]
        futuros    = close.iloc[idx + 1:]
        rec_idxs   = futuros[futuros >= precio_pre]
        if rec_idxs.empty:
            dias_rec = np.inf
        else:
            dias_rec = (rec_idxs.index[0] - fecha).days

        # Recuperación a horizontes fijos
        rec_horizons = {}
        for h in RECOVERY_DAYS:
            idx_h = idx + h
            if idx_h < len(close):
                rec_horizons[f'recup_{h}d'] = int(close.iloc[idx_h] >= precio_pre)
                rec_horizons[f'ret_{h}d']   = round(
                    (close.iloc[idx_h] - close.iloc[idx]) / abs(close.iloc[idx]) * 100, 2
                )
            else:
                rec_horizons[f'recup_{h}d'] = np.nan
                rec_horizons[f'ret_{h}d']   = np.nan

        streaks = dias_consecutivos_bajando(rets_arr, idx)

        event = {
            'ticker':        ticker,
            'fecha':         fecha,
            'tipo_activo':   'ETF' if ticker in ETFS else 'ACCION',
            'ret_1d':        round(row['ret_1d'], 3),
            'zscore':        round(row['zscore'], 3),
            'vol_ratio':     round(row['vol_ratio'], 3) if not np.isnan(row['vol_ratio']) else np.nan,
            'rsi_14':        round(row['rsi_14'], 1)    if not np.isnan(row['rsi_14'])    else np.nan,
            'dias_consec':   streaks,
            'n_alcistas':    int(row['n_alcistas']),
            'magnitud':      bucket_magnitud(row['ret_1d']),
            'dias_rec':      dias_rec,
            **rec_horizons,
        }
        rows.append(event)

    return pd.DataFrame(rows)


# ── Análisis por feature ──────────────────────────────────────────────────────

def tabla_recovery(df, col_grupo, orden=None, titulo=''):
    grupos = df.groupby(col_grupo)
    filas  = []
    for nombre, sub in grupos:
        fila = {'grupo': nombre, 'N': len(sub)}
        for h in RECOVERY_DAYS:
            col = f'recup_{h}d'
            if col in sub.columns:
                vals = sub[col].dropna()
                fila[f'rec_{h}d_%'] = round(vals.mean() * 100, 1) if len(vals) else np.nan
        mediana = sub['dias_rec'].replace(np.inf, np.nan).median()
        fila['mediana_dias'] = round(mediana, 0) if not np.isnan(mediana) else '∞'
        nunca = (sub['dias_rec'] == np.inf).mean() * 100
        fila['nunca_%'] = round(nunca, 1)
        filas.append(fila)

    tbl = pd.DataFrame(filas)
    if orden:
        tbl['_ord'] = tbl['grupo'].map({v: i for i, v in enumerate(orden)})
        tbl = tbl.sort_values('_ord').drop(columns='_ord')

    print(f'\n  {titulo}')
    print(f'  {"Grupo":<26} {"N":>5}  {"5d":>7}  {"10d":>7}  {"20d":>7}  {"60d":>7}  {"Med.días":>9}  {"Nunca":>7}')
    print(f'  {"-"*82}')
    for _, r in tbl.iterrows():
        print(
            f'  {str(r["grupo"]):<26} {int(r["N"]):>5} '
            f' {r.get("rec_5d_%", "—"):>6}%'
            f'  {r.get("rec_10d_%", "—"):>6}%'
            f'  {r.get("rec_20d_%", "—"):>6}%'
            f'  {r.get("rec_60d_%", "—"):>6}%'
            f'  {str(r["mediana_dias"]):>9}'
            f'  {r["nunca_%"]:>6}%'
        )
    return tbl


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print('\n' + '=' * 72)
    print('  FASE 3 — FEATURES GRANULARES')
    print(f'  Universo: {len(TICKERS)} tickers  |  Período: 2015-2026')
    print('=' * 72)

    frames = []
    for ticker in TICKERS:
        try:
            df_t = enriquecer_ticker(ticker)
            if not df_t.empty:
                frames.append(df_t)
                print(f'  {ticker:<6} → {len(df_t):>4} eventos')
        except Exception as e:
            print(f'  {ticker:<6} → ERROR: {e}')

    if not frames:
        print('Sin datos.')
        return

    df = pd.concat(frames, ignore_index=True)
    print(f'\n  Total eventos: {len(df)}\n')

    # ── Feature 1: Tipo de activo ─────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  FEATURE 1 — TIPO DE ACTIVO (ETF vs ACCIÓN)')
    print('=' * 72)
    tabla_recovery(df, 'tipo_activo', orden=['ETF', 'ACCION'],
                   titulo='Recuperación según tipo de activo')

    # ── Feature 2: Volumen relativo ───────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  FEATURE 2 — VOLUMEN RELATIVO')
    print('=' * 72)
    df['bucket_vol'] = df['vol_ratio'].apply(bucket_volumen)
    orden_vol = ['Bajo (<0.8x)', 'Normal (0.8-1.5x)', 'Alto (1.5-2.5x)', 'Muy alto (>2.5x)', 'Sin datos']
    tabla_recovery(df, 'bucket_vol', orden=orden_vol,
                   titulo='Recuperación según volumen del día de la caída')

    # ── Feature 3: RSI ───────────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print('  FEATURE 3 — RSI EN EL MOMENTO DE LA CAÍDA')
    print('=' * 72)
    df['bucket_rsi'] = df['rsi_14'].apply(bucket_rsi)
    orden_rsi = ['Sobrevendido (<30)', 'Neutral (30-70)', 'Sobrecomprado (>70)', 'Sin datos']
    tabla_recovery(df, 'bucket_rsi', orden=orden_rsi,
                   titulo='Recuperación según RSI (14 días)')

    # ── Feature 4: Días consecutivos bajando ─────────────────────────────────
    print('\n' + '=' * 72)
    print('  FEATURE 4 — DÍAS CONSECUTIVOS BAJANDO ANTES DEL EVENTO')
    print('=' * 72)
    df['bucket_streak'] = df['dias_consec'].apply(bucket_streak)
    orden_streak = ['0 — caída aislada', '1-2 días bajando', '3-4 días bajando', '5+ días bajando']
    tabla_recovery(df, 'bucket_streak', orden=orden_streak,
                   titulo='Recuperación según racha bajista previa')

    # ── Cruce: ETF vs ACCION × Magnitud ──────────────────────────────────────
    print('\n' + '=' * 72)
    print('  CRUCE — TIPO × MAGNITUD DE CAÍDA')
    print('=' * 72)
    df['tipo_x_mag'] = df['tipo_activo'] + ' | ' + df['magnitud']
    tabla_recovery(df, 'tipo_x_mag', titulo='ETF vs Acción por magnitud de caída')

    # ── Guardar CSV para Fase 4 ───────────────────────────────────────────────
    csv_path = OUTPUT_DIR / 'phase3_features.csv'
    df.to_csv(csv_path, index=False)

    print('\n' + '=' * 72)
    print('  RESUMEN EJECUTIVO')
    print('=' * 72)

    for feature, col in [('Tipo activo', 'tipo_activo'), ('Vol bucket', 'bucket_vol'),
                          ('RSI bucket', 'bucket_rsi'), ('Streak', 'bucket_streak')]:
        best = (
            df.groupby(col)['recup_20d']
              .apply(lambda x: round(x.dropna().mean() * 100, 1))
              .idxmax()
        )
        best_val = df.groupby(col)['recup_20d'].apply(
            lambda x: round(x.dropna().mean() * 100, 1)
        ).max()
        print(f'  Mejor {feature:<15}: {best:<28} → {best_val}% recuperación a 20d')

    print(f'\n  Dataset Phase 4 guardado: {csv_path}')
    print(f'  Columnas: {list(df.columns)}')
    print('=' * 72 + '\n')


if __name__ == '__main__':
    main()
