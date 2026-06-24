"""
Fase 4 v2 — Estrategia rediseñada desde la raíz.

Cambios respecto a v1:
  1. Target: ret_10d > 1%  (¿ganas dinero en 10 días?)
             más accionable que "volver al precio exacto"
  2. Feature estrella: drop idiosincrásico vs drop sistémico
       idio_ret = ret_1d - spy_ret_1d
       Si SPY cae 3% y AAPL cae 5% → diferencia -2% (parcialmente sistémico)
       Si SPY sube 0.5% y AAPL cae 5% → diferencia -5.5% (idiosincrásico puro)
  3. Más features: distancia a mínimo 52 semanas, drop normalizado por ATR
  4. Validación: TimeSeriesSplit (5 folds) → AUC honesto sin data leakage
  5. Output: backtest real en $ → ¿cuánto habría ganado la estrategia?
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from download_data import cargar
from config import TICKERS

OUTPUT_DIR = Path('output')

ETFS = {'SPY','QQQ','DIA','IWM','XLF','XLE','XLK','XLV','XLC','GLD','SLV','USO','UNG'}

DROP_THRESHOLD = -3.0
ZSCORE_THRESH  = -2.0
ROLLING_WIN    = 60
TARGET_DAYS    = 10
TARGET_MIN_RET = 1.0      # ganas si ret_10d > +1%
FECHA_CORTE    = '2023-01-01'


# ── Cargar SPY como contexto de mercado ───────────────────────────────────────

def cargar_spy():
    df = cargar('SPY').copy()
    df['spy_ret'] = df['Close'].pct_change() * 100
    return df[['spy_ret']]


# ── Calcular ATR 14 días ──────────────────────────────────────────────────────

def calcular_atr(df, period=14):
    high  = df['High'] if 'High' in df.columns else df['Close']
    low   = df['Low']  if 'Low'  in df.columns else df['Close']
    close = df['Close']
    tr    = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ── RSI ───────────────────────────────────────────────────────────────────────

def calcular_rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.where(delta > 0, 0.0)
    loss     = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period-1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period-1, min_periods=period).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ── Construcción del dataset ──────────────────────────────────────────────────

def construir_dataset(spy_rets):
    rows = []

    for ticker in TICKERS:
        try:
            df = cargar(ticker).copy()
        except FileNotFoundError:
            continue

        close = df['Close']
        df['ret_1d']   = close.pct_change() * 100
        df['ret_mean'] = df['ret_1d'].rolling(ROLLING_WIN).mean()
        df['ret_std']  = df['ret_1d'].rolling(ROLLING_WIN).std()
        df['zscore']   = (df['ret_1d'] - df['ret_mean']) / df['ret_std']
        df['rsi_14']   = calcular_rsi(close)
        df['atr_14']   = calcular_atr(df)

        # Volumen relativo
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        else:
            df['vol_ratio'] = np.nan

        # Tendencia
        df['sma50']      = close.rolling(50).mean()
        df['sma200']     = close.rolling(200).mean()
        df['ret_20d']    = close.pct_change(20) * 100
        df['n_alcistas'] = (
            (close > df['sma200']).astype(int) +
            (df['sma50'] > df['sma200']).astype(int) +
            (df['ret_20d'] > 0).astype(int)
        )

        # Distancia al mínimo de 52 semanas
        df['low_252']   = close.rolling(252).min()
        df['dist_low']  = (close - df['low_252']) / df['low_252'] * 100

        # Join con SPY para calcular componente idiosincrásico
        df = df.join(spy_rets, how='left')

        # Días consecutivos bajando
        rets_arr = df['ret_1d'].values

        # Detección de eventos
        mask = (df['ret_1d'] < DROP_THRESHOLD) & (df['zscore'] < ZSCORE_THRESH)
        idxs = np.where(mask.values)[0]

        for idx in idxs:
            if idx < 252:   # necesitamos histórico suficiente
                continue
            row   = df.iloc[idx]
            fecha = df.index[idx]

            if pd.isna(row.get('sma200', np.nan)):
                continue

            # Target: ret a 10 días > 1%
            idx_t = idx + TARGET_DAYS
            if idx_t >= len(close):
                continue
            ret_10d = (close.iloc[idx_t] - close.iloc[idx]) / abs(close.iloc[idx]) * 100
            target  = int(ret_10d > TARGET_MIN_RET)

            # Días consecutivos bajando
            streak = 0
            i = idx - 1
            while i >= 0 and rets_arr[i] < 0:
                streak += 1
                i -= 1

            # Drop normalizado por ATR (¿cuán extremo es respecto a volatilidad habitual?)
            atr = row['atr_14']
            drop_norm = row['ret_1d'] / atr if (not pd.isna(atr) and atr > 0) else np.nan

            # Componente idiosincrásico
            spy_r  = row.get('spy_ret', np.nan)
            idio   = row['ret_1d'] - spy_r if not pd.isna(spy_r) else np.nan
            es_sistematico = int(spy_r < -1.0) if not pd.isna(spy_r) else 0

            rows.append({
                'ticker':         ticker,
                'fecha':          fecha,
                'tipo_activo':    'ETF' if ticker in ETFS else 'ACCION',
                'es_etf':         int(ticker in ETFS),
                # Features originales
                'ret_1d':         round(row['ret_1d'], 3),
                'zscore':         round(row['zscore'], 3),
                'vol_ratio':      row['vol_ratio'],
                'rsi_14':         row['rsi_14'],
                'dias_consec':    streak,
                'n_alcistas':     int(row['n_alcistas']),
                # Features nuevas
                'spy_ret':        spy_r,
                'idio_ret':       idio,          # cuánto cae MÁS que el mercado
                'es_sistematico': es_sistematico, # ¿el mercado también cayó >1%?
                'dist_low_52w':   row['dist_low'],# distancia al mínimo 52 semanas
                'drop_norm_atr':  drop_norm,      # drop / ATR (extremidad normalizada)
                # Magnitud
                'mag_ord': (
                    1 if row['ret_1d'] >= -3  else
                    2 if row['ret_1d'] >= -5  else
                    3 if row['ret_1d'] >= -10 else 4
                ),
                # Target y resultado real
                'ret_10d':  round(ret_10d, 3),
                'target':   target,
            })

    df_out = pd.DataFrame(rows).sort_values('fecha').reset_index(drop=True)
    print(f'  Dataset construido: {len(df_out)} eventos')
    print(f'  Positivos (ret_10d > {TARGET_MIN_RET}%): {df_out["target"].mean()*100:.1f}%')
    return df_out


# ── Features y preparación ────────────────────────────────────────────────────

FEATURES = [
    'ret_1d', 'zscore', 'vol_ratio', 'rsi_14', 'dias_consec',
    'n_alcistas', 'idio_ret', 'es_sistematico', 'dist_low_52w',
    'drop_norm_atr', 'es_etf', 'mag_ord', 'spy_ret',
]

def preparar_X(df, medianas=None):
    X = df[FEATURES].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    if medianas is None:
        medianas = X.median()
    X = X.fillna(medianas)
    return X, medianas


# ── Validación TimeSeriesSplit ────────────────────────────────────────────────

def validacion_temporal(df):
    print(f'\n{"="*62}')
    print(f'  VALIDACIÓN CRUZADA TEMPORAL (5 folds)')
    print(f'{"="*62}')

    df_sorted = df.sort_values('fecha').reset_index(drop=True)
    X_all, med = preparar_X(df_sorted)
    y_all = df_sorted['target'].values

    tscv   = TimeSeriesSplit(n_splits=5)
    aucs_xgb, aucs_lr = [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_all), 1):
        X_tr, y_tr   = X_all.iloc[tr_idx], y_all[tr_idx]
        X_val, y_val = X_all.iloc[val_idx], y_all[val_idx]

        med_fold = X_tr.median()
        X_tr  = X_tr.fillna(med_fold)
        X_val = X_val.fillna(med_fold)

        if y_val.sum() < 5:
            continue

        ratio = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1)

        # XGBoost regularizado
        xgb_m = xgb.XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.7, colsample_bytree=0.7,
            min_child_weight=10, reg_lambda=5, reg_alpha=2,
            scale_pos_weight=ratio, eval_metric='auc',
            random_state=42, n_jobs=-1,
        )
        xgb_m.fit(X_tr, y_tr, verbose=False)
        auc_xgb = roc_auc_score(y_val, xgb_m.predict_proba(X_val)[:, 1])
        aucs_xgb.append(auc_xgb)

        # Logistic Regression como baseline
        sc  = StandardScaler()
        lr  = LogisticRegression(C=0.1, class_weight='balanced', max_iter=500)
        lr.fit(sc.fit_transform(X_tr), y_tr)
        auc_lr = roc_auc_score(y_val, lr.predict_proba(sc.transform(X_val))[:, 1])
        aucs_lr.append(auc_lr)

        fecha_ini = df_sorted['fecha'].iloc[val_idx[0]].date()
        fecha_fin = df_sorted['fecha'].iloc[val_idx[-1]].date()
        print(f'  Fold {fold} ({fecha_ini} → {fecha_fin}): '
              f'XGB={auc_xgb:.3f}  LR={auc_lr:.3f}')

    print(f'\n  AUC medio XGBoost:  {np.mean(aucs_xgb):.3f} ± {np.std(aucs_xgb):.3f}')
    print(f'  AUC medio LR:       {np.mean(aucs_lr):.3f} ± {np.std(aucs_lr):.3f}')
    return np.mean(aucs_xgb)


# ── Modelo final en train/test ────────────────────────────────────────────────

def modelo_final(df):
    print(f'\n{"="*62}')
    print(f'  MODELO FINAL — train < {FECHA_CORTE}  |  test >= {FECHA_CORTE}')
    print(f'{"="*62}')

    train = df[df['fecha'] < FECHA_CORTE]
    test  = df[df['fecha'] >= FECHA_CORTE]

    X_tr, med = preparar_X(train)
    y_tr      = train['target'].values
    X_te, _   = preparar_X(test, med)
    y_te      = test['target'].values

    ratio = max((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), 1)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        min_child_weight=10, reg_lambda=5, reg_alpha=2,
        scale_pos_weight=ratio, eval_metric='auc',
        random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, verbose=False)

    prob_tr = model.predict_proba(X_tr)[:, 1]
    prob_te = model.predict_proba(X_te)[:, 1]

    auc_tr = roc_auc_score(y_tr, prob_tr)
    auc_te = roc_auc_score(y_te, prob_te)

    print(f'  Train: {len(y_tr)} eventos  |  Test: {len(y_te)} eventos')
    print(f'  Positivos train: {y_tr.mean()*100:.1f}%  |  test: {y_te.mean()*100:.1f}%')
    print(f'\n  ROC-AUC train : {auc_tr:.4f}')
    print(f'  ROC-AUC test  : {auc_te:.4f}  (gap: {auc_tr-auc_te:.3f})')

    print(f'\n{classification_report(y_te, (prob_te >= 0.55).astype(int), target_names=["No gana","Gana >1%"], digits=3)}')

    return model, prob_te, test, med


# ── Importancia de features ───────────────────────────────────────────────────

def graficar_importancia(model):
    imp = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['tomato' if f in ('idio_ret','es_sistematico','spy_ret','dist_low_52w','drop_norm_atr')
              else 'steelblue' for f in imp.index]
    imp.plot(kind='barh', ax=ax, color=colors)
    ax.set_title('Importancia de features — XGBoost v2\n(rojo = features nuevas)')
    ax.set_xlabel('Importancia (gain)')
    plt.tight_layout()
    ruta = OUTPUT_DIR / 'phase4v2_importancia.png'
    plt.savefig(ruta, dpi=130)
    plt.close()

    print(f'\n{"="*62}')
    print(f'  IMPORTANCIA DE FEATURES')
    print(f'{"="*62}')
    print(f'  {"Feature":<18} {"Importancia":>12}  {"Nueva?":>8}')
    print(f'  {"-"*42}')
    nuevas = {'idio_ret','es_sistematico','spy_ret','dist_low_52w','drop_norm_atr'}
    for feat, val in imp.sort_values(ascending=False).items():
        marca = '⭐ NUEVA' if feat in nuevas else ''
        print(f'  {feat:<18} {val:>12.4f}  {marca}')
    print(f'  Gráfico: {ruta}')


# ── Backtest de la estrategia ─────────────────────────────────────────────────

def backtest_estrategia(test_df, prob_te):
    print(f'\n{"="*62}')
    print(f'  BACKTEST DE ESTRATEGIA — $1.000 por señal')
    print(f'{"="*62}')
    print(f'  Lógica: si prob >= umbral → invertir $1.000, vender a 10 días')
    print(f'  Benchmark: invertir en TODAS las caídas anómalas')
    print()

    df = test_df.copy()
    df['prob'] = prob_te

    # Benchmark sin modelo
    ret_bench  = df['ret_10d'].mean()
    total_inv_bench = len(df) * 1000
    ganancia_bench  = (df['ret_10d'] / 100 * 1000).sum()
    print(f'  Benchmark (todas las señales):')
    print(f'    Operaciones: {len(df)}  |  Ret. medio: {ret_bench:+.2f}%  |  P&L: ${ganancia_bench:+,.0f}')
    print()

    print(f'  {"Umbral":>8}  {"Ops":>5}  {"Ret.medio":>10}  {"P&L":>10}  {"Win%":>7}  {"Mejora vs bench":>16}')
    print(f'  {"-"*65}')

    for umbral in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel = df[df['prob'] >= umbral]
        if len(sel) < 5:
            continue
        ret_medio = sel['ret_10d'].mean()
        pnl       = (sel['ret_10d'] / 100 * 1000).sum()
        win_pct   = (sel['ret_10d'] > 1).mean() * 100
        mejora    = ret_medio - ret_bench
        print(f'  {umbral:>8.2f}  {len(sel):>5}  {ret_medio:>+9.2f}%  ${pnl:>+9,.0f}  {win_pct:>6.1f}%  {mejora:>+14.2f}pp')

    # Mejor umbral por retorno medio
    resultados = []
    for umbral in np.arange(0.40, 0.80, 0.02):
        sel = df[df['prob'] >= umbral]
        if len(sel) < 5:
            break
        resultados.append({'umbral': umbral, 'ret': sel['ret_10d'].mean(), 'n': len(sel)})
    if resultados:
        mejor = max(resultados, key=lambda x: x['ret'])
        print(f'\n  Umbral óptimo por retorno medio: {mejor["umbral"]:.2f} '
              f'→ {mejor["ret"]:+.2f}% medio  ({mejor["n"]} operaciones)')


# ── Análisis idiosincrásico vs sistémico ──────────────────────────────────────

def analisis_idio_vs_sistematico(df):
    print(f'\n{"="*62}')
    print(f'  CAÍDA IDIOSINCRÁSICA vs SISTÉMICA')
    print(f'{"="*62}')
    print(f'  Idiosincrásica: SPY plano o sube / stock cae solo')
    print(f'  Sistémica:      SPY también cae más de -1%')
    print()

    for label, mask in [
        ('Idiosincrásica (SPY >= -1%)', df['es_sistematico'] == 0),
        ('Sistémica      (SPY < -1%)',  df['es_sistematico'] == 1),
    ]:
        sub = df[mask]
        if sub.empty:
            continue
        rec_10d = (sub['ret_10d'] > 1).mean() * 100
        ret_med = sub['ret_10d'].mean()
        print(f'  {label}')
        print(f'    N={len(sub)}  |  Ret.medio={ret_med:+.2f}%  |  Win% a 10d={rec_10d:.1f}%')
    print()

    # Por cuartiles de idio_ret
    print(f'  Por componente idiosincrásico (idio_ret = ret_stock - ret_SPY):')
    df['idio_q'] = pd.qcut(df['idio_ret'].dropna(), q=4,
                           labels=['Q1 menos idio','Q2','Q3','Q4 más idio'])
    for q, sub in df.groupby('idio_q', observed=True):
        ret_med = sub['ret_10d'].mean()
        win     = (sub['ret_10d'] > 1).mean() * 100
        print(f'    {str(q):<20} N={len(sub):>4}  Ret.med={ret_med:+.2f}%  Win%={win:.1f}%')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'\n{"="*62}')
    print(f'  FINANCE VOLATILITY — FASE 4 v2')
    print(f'  Target: ret_10d > {TARGET_MIN_RET}%  |  Horizonte: {TARGET_DAYS} días')
    print(f'{"="*62}\n')

    print('  Cargando SPY como contexto de mercado...')
    spy_rets = cargar_spy()

    print('  Construyendo dataset con features nuevas...')
    df = construir_dataset(spy_rets)

    df.to_csv(OUTPUT_DIR / 'phase4v2_dataset.csv', index=False)

    # Validación cruzada temporal
    validacion_temporal(df)

    # Modelo final
    model, prob_te, test_df, medianas = modelo_final(df)

    # Importancia
    graficar_importancia(model)

    # Análisis idiosincrásico
    analisis_idio_vs_sistematico(df[df['fecha'] >= FECHA_CORTE])

    # Backtest
    backtest_estrategia(test_df, prob_te)

    # Guardar predicciones test
    test_df = test_df.copy()
    test_df['prob'] = prob_te
    test_df.sort_values('prob', ascending=False).to_csv(
        OUTPUT_DIR / 'phase4v2_señales_test.csv', index=False
    )

    print(f'\n  Archivos guardados en output/')
    print(f'{"="*62}\n')


if __name__ == '__main__':
    main()
