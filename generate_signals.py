"""
Paso 2 — Generar señales del día.

Descarga datos frescos de yfinance, detecta caídas anómalas en el
último día disponible, puntúa cada señal con el modelo entrenado
y exporta docs/data/signals_live.json para el dashboard.

Diseñado para ejecutarse localmente o en GitHub Actions (cron diario).
"""

import json
import warnings
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Rutas ──────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path('output')
DOCS_DIR   = Path('docs/data')

# ── Parámetros (deben coincidir con phase4_v2.py) ─────────────────────────────
TICKERS = [
    'SPY', 'QQQ', 'DIA', 'IWM',
    'XLF', 'XLE', 'XLK', 'XLV', 'XLC',
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA',
    'JPM', 'GS', 'BAC', 'XOM', 'JNJ', 'WMT',
    'GLD', 'SLV', 'USO', 'UNG',
]

ETFS = {'SPY','QQQ','DIA','IWM','XLF','XLE','XLK','XLV','XLC','GLD','SLV','USO','UNG'}

DROP_THRESHOLD = -4.5
ZSCORE_THRESH  = -2.0
ROLLING_WIN    = 60
PROB_UMBRAL    = 0.55


# ── Helpers de indicadores ─────────────────────────────────────────────────────

def calcular_rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calcular_atr(df, period=14):
    high  = df['High']  if 'High'  in df.columns else df['Close']
    low   = df['Low']   if 'Low'   in df.columns else df['Close']
    close = df['Close']
    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── Descarga de datos ──────────────────────────────────────────────────────────

def descargar(ticker, periodo='3y'):
    df = yf.download(ticker, period=periodo, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ── Feature engineering sobre el día más reciente ─────────────────────────────

def calcular_features_hoy(df, spy_ret_hoy):
    """
    Recibe el histórico de un ticker y el retorno de SPY hoy.
    Devuelve un dict con las 13 features del último día,
    o None si no hay suficiente histórico o no hubo caída anómala.
    """
    if len(df) < 252:
        return None

    close = df['Close']
    ret   = close.pct_change() * 100

    # Indicadores sobre toda la serie
    ret_mean = ret.rolling(ROLLING_WIN).mean()
    ret_std  = ret.rolling(ROLLING_WIN).std()
    zscore   = (ret - ret_mean) / ret_std
    rsi      = calcular_rsi(close)
    atr      = calcular_atr(df)
    sma50    = close.rolling(50).mean()
    sma200   = close.rolling(200).mean()
    ret_20d  = close.pct_change(20) * 100
    low_252  = close.rolling(252).min()
    dist_low = (close - low_252) / low_252 * 100

    vol_ratio = pd.Series(np.nan, index=df.index)
    if 'Volume' in df.columns and df['Volume'].sum() > 0:
        vol_ratio = df['Volume'] / df['Volume'].rolling(20).mean()

    # Último día
    i       = -1
    ret_1d  = ret.iloc[i]
    z       = zscore.iloc[i]

    # Filtro de anomalía
    if pd.isna(ret_1d) or pd.isna(z):
        return None
    if not (ret_1d < DROP_THRESHOLD and z < ZSCORE_THRESH):
        return None

    # Días consecutivos bajando antes de hoy
    streak = 0
    for k in range(2, min(len(ret), 30)):
        if ret.iloc[-k] < 0:
            streak += 1
        else:
            break

    atr_val   = atr.iloc[i]
    drop_norm = ret_1d / atr_val if (not pd.isna(atr_val) and atr_val > 0) else np.nan

    n_alcistas = int(
        (close.iloc[i] > sma200.iloc[i]) +
        (sma50.iloc[i]  > sma200.iloc[i]) +
        (ret_20d.iloc[i] > 0)
    )

    idio_ret       = ret_1d - spy_ret_hoy if not pd.isna(spy_ret_hoy) else np.nan
    es_sistematico = int(spy_ret_hoy < -1.0) if not pd.isna(spy_ret_hoy) else 0

    mag_ord = (
        1 if ret_1d >= -3  else
        2 if ret_1d >= -5  else
        3 if ret_1d >= -10 else 4
    )

    return {
        'ret_1d':         round(float(ret_1d), 3),
        'zscore':         round(float(z), 3),
        'vol_ratio':      float(vol_ratio.iloc[i]) if not pd.isna(vol_ratio.iloc[i]) else None,
        'rsi_14':         float(rsi.iloc[i])       if not pd.isna(rsi.iloc[i])       else None,
        'dias_consec':    streak,
        'n_alcistas':     n_alcistas,
        'idio_ret':       round(float(idio_ret), 3) if not pd.isna(idio_ret)         else None,
        'es_sistematico': es_sistematico,
        'dist_low_52w':   round(float(dist_low.iloc[i]), 2) if not pd.isna(dist_low.iloc[i]) else None,
        'drop_norm_atr':  round(float(drop_norm), 3)        if not pd.isna(drop_norm)        else None,
        'es_etf':         0,
        'mag_ord':        mag_ord,
        'spy_ret':        round(float(spy_ret_hoy), 3) if not pd.isna(spy_ret_hoy) else None,
        'precio_cierre':  round(float(close.iloc[i]), 2),
        'fecha':          str(df.index[-1].date()),
    }


# ── Preparar X para el modelo ─────────────────────────────────────────────────

def preparar_fila(feat_dict, features_orden, medianas):
    row = {}
    for f in features_orden:
        val = feat_dict.get(f)
        row[f] = val if val is not None else float(medianas.get(f, 0))
    return pd.DataFrame([row])[features_orden]


# ── Señales históricas recientes ───────────────────────────────────────────────

def precios_en_dias(close, fecha_signal, dias=(10, 20, 30)):
    """Dado un índice de precios, devuelve el precio de cierre N días hábiles
    después de la fecha de señal. Devuelve None si no hay datos suficientes."""
    fecha_ts = pd.Timestamp(fecha_signal)
    loc = close.index.searchsorted(fecha_ts)
    if loc >= len(close):
        return {d: None for d in dias}
    p0 = float(close.iloc[loc])
    resultado = {}
    for d in dias:
        i = loc + d
        resultado[d] = round(float(close.iloc[i]), 2) if i < len(close) else None
    return p0, resultado


def cargar_historico_reciente(datos_cache, n=30):
    """Carga las últimas n señales del CSV de test y enriquece con precios reales."""
    csv_path = OUTPUT_DIR / 'phase4v2_señales_test.csv'
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path, parse_dates=['fecha'])
    df = df.sort_values('fecha', ascending=False).head(n)

    registros = []
    for _, row in df.iterrows():
        ticker = row['ticker']
        fecha  = row['fecha']

        precio_caida = precio_10d = precio_20d = precio_30d = None
        if ticker in datos_cache:
            close  = datos_cache[ticker]['Close']
            result = precios_en_dias(close, fecha, dias=[10, 20, 30])
            if result:
                p0, dias_precios = result
                precio_caida = round(p0, 2)
                precio_10d   = dias_precios[10]
                precio_20d   = dias_precios[20]
                precio_30d   = dias_precios[30]

        registros.append({
            'ticker':         ticker,
            'fecha':          str(fecha.date()),
            'ret_1d':         round(float(row['ret_1d']), 3),
            'ret_10d':        round(float(row['ret_10d']), 3),
            'prob':           round(float(row['prob']), 3),
            'es_sistematico': int(row.get('es_sistematico', 0)),
            'tipo_caida':     'Sistémica' if row.get('es_sistematico', 0) == 1 else 'Idiosincrásica',
            'resultado':      'Ganó' if float(row['ret_10d']) > 1.0 else 'No ganó',
            'precio_caida':   precio_caida,
            'precio_10d':     precio_10d,
            'precio_20d':     precio_20d,
            'precio_30d':     precio_30d,
        })
    return registros


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print('=' * 55)
    print('  GENERATE SIGNALS — Finance Volatility Check')
    print('=' * 55)

    # Cargar modelo
    print('\n  [1/4] Cargando modelo...')
    model      = joblib.load(OUTPUT_DIR / 'model.pkl')
    medianas   = joblib.load(OUTPUT_DIR / 'medianas.pkl')
    features   = json.load(open(OUTPUT_DIR / 'features.json'))
    print(f'        Modelo OK | {len(features)} features')

    # Descargar SPY
    print('  [2/4] Descargando SPY...')
    spy_df      = descargar('SPY')
    spy_ret_hoy = float(spy_df['Close'].pct_change().iloc[-1] * 100)
    fecha_mercado = str(spy_df.index[-1].date())
    print(f'        SPY ret hoy: {spy_ret_hoy:+.2f}%  |  Fecha: {fecha_mercado}')

    # Escanear tickers y guardar datos en caché para reutilizar en histórico
    print(f'  [3/4] Escaneando {len(TICKERS)} tickers...')
    señales    = []
    datos_cache = {}

    for ticker in TICKERS:
        try:
            df = descargar(ticker)
            datos_cache[ticker] = df

            feats = calcular_features_hoy(df, spy_ret_hoy)
            if feats is None:
                continue

            feats['es_etf'] = int(ticker in ETFS)
            X_fila  = preparar_fila(feats, features, medianas)
            prob    = float(model.predict_proba(X_fila)[0][1])

            señal = {
                'ticker':         ticker,
                'fecha':          feats['fecha'],
                'ret_1d':         feats['ret_1d'],
                'zscore':         feats['zscore'],
                'prob':           round(prob, 3),
                'es_sistematico': feats['es_sistematico'],
                'tipo_caida':     'Sistémica' if feats['es_sistematico'] == 1 else 'Idiosincrásica',
                'idio_ret':       feats['idio_ret'],
                'dist_low_52w':   feats['dist_low_52w'],
                'rsi_14':         round(feats['rsi_14'], 1) if feats['rsi_14'] else None,
                'spy_ret':        feats['spy_ret'],
                'precio_cierre':  feats['precio_cierre'],
                'mag_ord':        feats['mag_ord'],
                'recomendacion':  'SEÑAL' if prob >= PROB_UMBRAL else 'Débil',
            }
            señales.append(señal)
            print(f'        [{ticker}] ret={feats["ret_1d"]:+.2f}%  prob={prob:.2f}  -> {señal["recomendacion"]}')

        except Exception as e:
            print(f'        [{ticker}] Error: {e}')

    señales.sort(key=lambda x: x['prob'], reverse=True)

    # Señales históricas enriquecidas con precios reales
    historico = cargar_historico_reciente(datos_cache, n=30)

    # Exportar JSON
    print('  [4/4] Exportando JSON...')
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        'generated_at':    datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'market_date':     fecha_mercado,
        'spy_ret_hoy':     round(spy_ret_hoy, 3),
        'prob_umbral':     PROB_UMBRAL,
        'stats': {
            'tickers_analizados': len(TICKERS),
            'señales_hoy':        len(señales),
            'señales_fuertes':    sum(1 for s in señales if s['recomendacion'] == 'SEÑAL'),
        },
        'signals':   señales,
        'historico': historico,
    }

    json_path = DOCS_DIR / 'signals_live.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'        Guardado en {json_path}')
    print()
    print(f'  Senales detectadas hoy: {len(señales)}')
    for s in señales:
        print(f'    {s["ticker"]:<6} ret={s["ret_1d"]:+.2f}%  prob={s["prob"]:.2f}  {s["recomendacion"]}')
    if not señales:
        print('    (ninguna caida anomala hoy)')
    print('=' * 55)


if __name__ == '__main__':
    main()
