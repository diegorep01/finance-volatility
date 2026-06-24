"""
Análisis profundo de recuperación por magnitud de caída.

Para cada bucket de caída (leve / moderada / fuerte / extrema),
calcula por cada 100 eventos: cuántos se recuperan y en qué plazo.

Recuperación = precio vuelve al cierre del día anterior a la caída.
No requiere filtro Z-score — analiza TODAS las caídas por magnitud.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from download_data import cargar
from config import TICKERS

OUTPUT_DIR = Path('output')

# Buckets de magnitud de caída (en % de retorno diario)
BUCKETS = [
    ('Leve (-1% a -3%)',       -3.0,  -1.0),
    ('Moderada (-3% a -5%)',   -5.0,  -3.0),
    ('Fuerte (-5% a -10%)',   -10.0,  -5.0),
    ('Extrema (> -10%)',      -100.0, -10.0),
]

# Categorías de tiempo (en días hábiles)
TIME_CATS = [
    ('Corto plazo  (1–5 días)',      1,    5),
    ('Medio plazo  (6–30 días)',     6,   30),
    ('Largo plazo  (31d–1 año)',    31,  252),
    ('Muy largo    (1–2 años)',    253,  504),
    ('Terror       (> 2 años)',    505, 9999),
]


def dias_hasta_recuperacion(precios, idx_drop):
    """
    Devuelve el número de días hábiles hasta que el precio vuelve
    a superar el cierre del día anterior al drop.
    Devuelve np.inf si nunca se recupera en los datos disponibles.
    """
    if idx_drop == 0:
        return np.inf
    precio_pre = precios.iloc[idx_drop - 1]
    futuros    = precios.iloc[idx_drop + 1:]
    recuperados = futuros[futuros >= precio_pre]
    if recuperados.empty:
        return np.inf
    return (recuperados.index[0] - precios.index[idx_drop]).days


def analizar_bucket(tickers, low_pct, high_pct):
    """Recoge todos los eventos del bucket y calcula días a recuperación."""
    resultados = []
    for ticker in tickers:
        try:
            df    = cargar(ticker)
            close = df['Close']
            rets  = close.pct_change() * 100

            # Filtrar por bucket
            mask    = (rets < high_pct) & (rets >= low_pct)
            indices = np.where(mask)[0]

            for idx in indices:
                ret  = rets.iloc[idx]
                dias = dias_hasta_recuperacion(close, idx)
                resultados.append({'ticker': ticker, 'ret': ret, 'dias': dias})
        except Exception:
            continue
    return pd.DataFrame(resultados)


def categorizar_dias(dias):
    if np.isinf(dias):
        return 'Terror       (> 2 años)'
    for nombre, lo, hi in TIME_CATS:
        if lo <= dias <= hi:
            return nombre
    return 'Terror       (> 2 años)'


def imprimir_tabla(nombre_bucket, df_ev):
    n = len(df_ev)
    if n == 0:
        print(f'  Sin eventos en este bucket.\n')
        return

    df_ev = df_ev.copy()
    df_ev['categoria'] = df_ev['dias'].apply(categorizar_dias)

    conteos = df_ev['categoria'].value_counts()
    # recuperados = todos menos los Terror que son np.inf
    terror_inf = (df_ev['dias'] == np.inf).sum()
    terror_largo = (df_ev['categoria'] == 'Terror       (> 2 años)').sum() - terror_inf

    print(f'  Eventos totales: {n}  |  Caída promedio: {df_ev["ret"].mean():.2f}%')
    print(f'  Por cada 100 caídas:\n')
    print(f'  {"Plazo de recuperación":<32} {"N":>6}  {"/ 100":>7}  {"Acum.":>7}')
    print(f'  {"-"*55}')

    acum = 0
    for nombre, lo, hi in TIME_CATS:
        c    = conteos.get(nombre, 0)
        pct  = c / n * 100
        acum += pct
        # Terror: separar "nunca" de "> 2 años pero en datos"
        if nombre == 'Terror       (> 2 años)':
            print(f'  {nombre:<32} {c:>6}  {pct:>6.1f}%  {acum:>6.1f}%')
            nunca_pct = terror_inf / n * 100
            print(f'    └ de los cuales, NUNCA recuperado: {terror_inf}  ({nunca_pct:.1f}%)')
        else:
            print(f'  {nombre:<32} {c:>6}  {pct:>6.1f}%  {acum:>6.1f}%')

    # Tiempo mediano de recuperación (solo los que sí se recuperaron)
    recuperados = df_ev[df_ev['dias'] != np.inf]['dias']
    if not recuperados.empty:
        mediana  = recuperados.median()
        promedio = recuperados.mean()
        print(f'\n  Tiempo mediano de recuperación (los que sí vuelven): {mediana:.0f} días hábiles')
        print(f'  Tiempo promedio de recuperación:                      {promedio:.0f} días hábiles')
    print()


def main():
    print('\n' + '='*65)
    print('  FINANCE VOLATILITY CHECK — ANÁLISIS DE RECUPERACIÓN')
    print(f'  Universo: {len(TICKERS)} tickers  |  Período: 2015–2026')
    print('='*65)
    print('  Recuperación = precio vuelve al cierre del día anterior')
    print('  Columna "Acum." = % acumulado que se recuperó hasta ese plazo\n')

    resumen_global = []

    for nombre, low, high in BUCKETS:
        print(f'\n{"─"*65}')
        print(f'  BUCKET: {nombre}')
        print(f'{"─"*65}')
        df_ev = analizar_bucket(TICKERS, low, high)
        imprimir_tabla(nombre, df_ev)

        # Para el CSV de resumen
        if not df_ev.empty:
            df_ev['bucket'] = nombre
            resumen_global.append(df_ev)

    # Guardar CSV completo
    if resumen_global:
        df_all = pd.concat(resumen_global, ignore_index=True)
        df_all['recuperado'] = df_all['dias'] != np.inf
        df_all.to_csv(OUTPUT_DIR / 'recovery_deep.csv', index=False)
        print(f'\n  CSV completo guardado: output/recovery_deep.csv')

    print('\n' + '='*65)


if __name__ == '__main__':
    main()
