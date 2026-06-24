"""
Pipeline completo — Finance Volatility Check Fase 1

Uso:
    python main.py            → descarga datos + detecta + backtesta
    python main.py --nodown   → salta la descarga (usa datos existentes)
"""

import sys
from download_data import descargar
from backtest_recovery import analizar_todos, resumen, graficar_distribucion
from config import TICKERS, RECOVERY_HORIZONS

def main():
    skip_download = '--nodown' in sys.argv

    if not skip_download:
        print('── PASO 1: Descarga de datos ──')
        descargar(TICKERS)
    else:
        print('── Paso 1 omitido (--nodown) ──')

    print('\n── PASO 2: Detección de caídas anómalas + Backtesting ──')
    df_res = analizar_todos(TICKERS)

    if df_res.empty:
        print('No se encontraron eventos. Revisa los parámetros en config.py')
        return

    print('\n── PASO 3: Resultados ──')
    resumen(df_res)

    print('── PASO 4: Gráficos ──')
    for h in RECOVERY_HORIZONS:
        graficar_distribucion(df_res, h)

    print('\nFase 1 completada.')

if __name__ == '__main__':
    main()
