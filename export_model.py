"""
Paso 1 — Exportar el modelo entrenado a disco.

Entrena el modelo sobre todos los datos históricos disponibles
y guarda tres archivos en output/:
  - model.pkl      → el modelo XGBoost listo para predecir
  - medianas.pkl   → valores de relleno para datos ausentes
  - features.json  → lista de features en el orden exacto que espera el modelo
"""

import json
import joblib
from pathlib import Path

from phase4_v2 import (
    cargar_spy,
    construir_dataset,
    preparar_X,
    FEATURES,
)
import xgboost as xgb

OUTPUT_DIR = Path('output')

ETFS = {'SPY','QQQ','DIA','IWM','XLF','XLE','XLK','XLV','XLC','GLD','SLV','USO','UNG'}

DROP_THRESHOLD = -3.0
TARGET_MIN_RET = 1.0


def entrenar_modelo_completo(df):
    """Entrena XGBoost sobre TODO el dataset (sin separar test)."""
    X, medianas = preparar_X(df)
    y = df['target'].values

    ratio = max((y == 0).sum() / max((y == 1).sum(), 1), 1)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.7,
        min_child_weight=10, reg_lambda=5, reg_alpha=2,
        scale_pos_weight=ratio, eval_metric='auc',
        random_state=42, n_jobs=-1,
    )
    model.fit(X, y, verbose=False)
    return model, medianas


def main():
    print('=' * 55)
    print('  EXPORT MODEL — Finance Volatility Check')
    print('=' * 55)

    print('\n  [1/4] Cargando SPY como contexto de mercado...')
    spy_rets = cargar_spy()

    print('  [2/4] Construyendo dataset completo...')
    df = construir_dataset(spy_rets)
    print(f'        {len(df)} eventos totales')

    print('  [3/4] Entrenando modelo sobre todo el histórico...')
    model, medianas = entrenar_modelo_completo(df)
    print(f'        Features: {len(FEATURES)}')
    print(f'        Positivos: {df["target"].mean()*100:.1f}%')

    print('  [4/4] Guardando archivos...')
    joblib.dump(model,    OUTPUT_DIR / 'model.pkl')
    joblib.dump(medianas, OUTPUT_DIR / 'medianas.pkl')
    with open(OUTPUT_DIR / 'features.json', 'w') as f:
        json.dump(FEATURES, f)

    print()
    print(f'  OK output/model.pkl')
    print(f'  OK output/medianas.pkl')
    print(f'  OK output/features.json')
    print()
    print('  Modelo listo. Siguiente paso: generate_signals.py')
    print('=' * 55)


if __name__ == '__main__':
    main()
