"""
Fase 4 — Modelo XGBoost supervisado.

Pregunta: dado un evento de caída anómala, ¿se recuperará al precio previo
en ≤ 20 días hábiles?

Target : recup_20d  (1 = sí, 0 = no)
Split  : temporal — train 2015-2022 / test 2023-2026 (sin data leakage)
Métrica principal: ROC-AUC + Precision-Recall (más informativas que accuracy)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay,
)
import xgboost as xgb

OUTPUT_DIR = Path('output')

# ── Configuración ─────────────────────────────────────────────────────────────
FECHA_CORTE   = '2023-01-01'   # train < corte, test >= corte
TARGET        = 'recup_20d'
FEATURES_NUM  = ['ret_1d', 'zscore', 'vol_ratio', 'rsi_14', 'dias_consec', 'n_alcistas']
FEATURES_CAT  = ['tipo_activo', 'magnitud']

XGB_PARAMS = dict(
    n_estimators      = 400,
    max_depth         = 4,
    learning_rate     = 0.05,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    min_child_weight  = 5,
    eval_metric       = 'auc',
    random_state      = 42,
    n_jobs            = -1,
)


# ── Preparación de datos ──────────────────────────────────────────────────────

def preparar(df):
    df = df.copy()
    df['fecha'] = pd.to_datetime(df['fecha'])

    # Eliminar filas sin target
    df = df[df[TARGET].notna()].copy()

    # Codificar categóricas
    df['es_etf'] = (df['tipo_activo'] == 'ETF').astype(int)

    orden_mag = {
        'Leve (-1% a -3%)':     1,
        'Moderada (-3% a -5%)': 2,
        'Fuerte (-5% a -10%)':  3,
        'Extrema (>-10%)':      4,
    }
    df['mag_ord'] = df['magnitud'].map(orden_mag).fillna(2).astype(int)

    features = FEATURES_NUM + ['es_etf', 'mag_ord']

    # Imputar NaN con mediana del train (se recalculará en split)
    for col in features:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df, features


def split_temporal(df, features):
    mask_train = df['fecha'] < FECHA_CORTE
    mask_test  = df['fecha'] >= FECHA_CORTE

    X_train = df.loc[mask_train, features].copy()
    y_train = df.loc[mask_train, TARGET].astype(int)
    X_test  = df.loc[mask_test,  features].copy()
    y_test  = df.loc[mask_test,  TARGET].astype(int)

    # Imputar con medianas del train
    medianas = X_train.median()
    X_train  = X_train.fillna(medianas)
    X_test   = X_test.fillna(medianas)

    return X_train, y_train, X_test, y_test, medianas


# ── Entrenamiento ─────────────────────────────────────────────────────────────

def entrenar(X_train, y_train, X_test, y_test):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(scale_pos_weight=ratio, **XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )
    return model


# ── Evaluación ────────────────────────────────────────────────────────────────

def evaluar(model, X_train, y_train, X_test, y_test):
    prob_train = model.predict_proba(X_train)[:, 1]
    prob_test  = model.predict_proba(X_test)[:, 1]
    pred_test  = model.predict(X_test)

    auc_train = roc_auc_score(y_train, prob_train)
    auc_test  = roc_auc_score(y_test, prob_test)

    print(f'\n{"="*62}')
    print(f'  FASE 4 — RESULTADOS DEL MODELO XGBOOST')
    print(f'{"="*62}')
    print(f'  Split temporal: train < {FECHA_CORTE} | test >= {FECHA_CORTE}')
    print(f'  Train: {len(y_train)} eventos  |  Test: {len(y_test)} eventos')
    print(f'  Positivos train: {y_train.mean()*100:.1f}%  |  test: {y_test.mean()*100:.1f}%')
    print(f'\n  ROC-AUC train : {auc_train:.4f}')
    print(f'  ROC-AUC test  : {auc_test:.4f}')
    if auc_train - auc_test > 0.07:
        print(f'  ⚠ Posible overfitting (diferencia: {auc_train-auc_test:.3f})')
    print(f'\n  Reporte de clasificación (test):')
    print(classification_report(y_test, pred_test,
                                target_names=['No recupera', 'Recupera'],
                                digits=3))

    return prob_test, pred_test, auc_test


# ── Importancia de features ───────────────────────────────────────────────────

def graficar_importancia(model, features):
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    imp.plot(kind='barh', ax=ax, color='steelblue')
    ax.set_title('Importancia de features — XGBoost Fase 4')
    ax.set_xlabel('Importancia (gain)')
    plt.tight_layout()
    ruta = OUTPUT_DIR / 'phase4_feature_importance.png'
    plt.savefig(ruta, dpi=130)
    plt.close()
    print(f'\n  Gráfico importancia: {ruta}')

    print(f'\n  {"Feature":<18} {"Importancia":>12}')
    print(f'  {"-"*32}')
    for feat, val in imp.sort_values(ascending=False).items():
        print(f'  {feat:<18} {val:>12.4f}')

    return imp


# ── Curvas ROC y Precision-Recall ─────────────────────────────────────────────

def graficar_curvas(model, X_test, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
    axes[0].set_title('Curva ROC — Test (2023-2026)')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    axes[1].set_title('Precision-Recall — Test (2023-2026)')

    plt.tight_layout()
    ruta = OUTPUT_DIR / 'phase4_curvas.png'
    plt.savefig(ruta, dpi=130)
    plt.close()
    print(f'  Gráfico curvas:     {ruta}')


# ── Análisis por umbral de probabilidad ──────────────────────────────────────

def analizar_umbrales(df_test, prob_test):
    print(f'\n{"="*62}')
    print(f'  SEÑALES POR UMBRAL DE CONFIANZA')
    print(f'{"="*62}')
    print(f'  ¿Qué pasa si solo operamos cuando el modelo está seguro?')
    print(f'\n  {"Umbral":>8}  {"Señales":>8}  {"Precision":>10}  {"Recall":>8}  {"% universo":>11}')
    print(f'  {"-"*52}')

    y = df_test[TARGET].astype(int).values
    for umbral in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        mask  = prob_test >= umbral
        n_sel = mask.sum()
        if n_sel == 0:
            continue
        prec  = y[mask].mean()
        rec   = y[mask].sum() / max(y.sum(), 1)
        pct   = n_sel / len(y) * 100
        print(f'  {umbral:>8.2f}  {n_sel:>8}  {prec:>9.1%}  {rec:>7.1%}  {pct:>10.1f}%')


# ── Top señales activas ────────────────────────────────────────────────────────

def top_señales(df_test, prob_test, n=15):
    df_out = df_test.copy()
    df_out['prob_recupera'] = prob_test
    df_out = df_out.sort_values('prob_recupera', ascending=False)

    print(f'\n{"="*62}')
    print(f'  TOP {n} EVENTOS MÁS CONFIABLES DEL PERIODO TEST')
    print(f'{"="*62}')
    cols = ['fecha', 'ticker', 'ret_1d', 'rsi_14', 'vol_ratio',
            'dias_consec', 'prob_recupera', TARGET]
    cols = [c for c in cols if c in df_out.columns]
    print(df_out[cols].head(n).to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f'\n{"="*62}')
    print(f'  Cargando dataset Fase 3...')
    csv = OUTPUT_DIR / 'phase3_features.csv'
    df_raw = pd.read_csv(csv)
    print(f'  {len(df_raw)} eventos cargados')

    df, features = preparar(df_raw)
    X_train, y_train, X_test, y_test, medianas = split_temporal(df, features)

    print(f'\n  Entrenando XGBoost...')
    model = entrenar(X_train, y_train, X_test, y_test)

    prob_test, pred_test, auc_test = evaluar(model, X_train, y_train, X_test, y_test)

    imp = graficar_importancia(model, features)
    graficar_curvas(model, X_test, y_test)

    df_test = df[df['fecha'] >= FECHA_CORTE].copy()
    analizar_umbrales(df_test, prob_test)
    top_señales(df_test, prob_test)

    # Guardar predicciones completas
    df_test = df_test.copy()
    df_test['prob_recupera'] = prob_test
    df_test['pred_recupera'] = pred_test
    out_csv = OUTPUT_DIR / 'phase4_predicciones_test.csv'
    df_test.to_csv(out_csv, index=False)

    print(f'\n  CSV completo: {out_csv}')
    print(f'{"="*62}\n')


if __name__ == '__main__':
    main()
