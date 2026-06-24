# ── Parámetros del proyecto Finance_Volatility_Check ──

# Universo de tickers a analizar
# Mezcla: índices, ETFs sectoriales y large caps líquidas
TICKERS = [
    # Índices / ETFs amplios
    'SPY', 'QQQ', 'DIA', 'IWM',
    # Sectoriales
    'XLF', 'XLE', 'XLK', 'XLV', 'XLC',
    # Large caps tech
    'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'AMZN', 'TSLA',
    # Large caps otras
    'JPM', 'GS', 'BAC', 'XOM', 'JNJ', 'WMT',
    # Commodities ETF
    'GLD', 'SLV', 'USO', 'UNG',
]

# Período de descarga
START_DATE = '2015-01-01'
END_DATE   = None  # None = hoy

# Detección de caídas anómalas
DROP_THRESHOLD_PCT  = -3.0   # caída mínima absoluta en % (ej: -3%)
ZSCORE_THRESHOLD    = -2.0   # desviaciones std por debajo de la media móvil
ROLLING_WINDOW      = 60     # días para calcular media y std móviles

# Backtesting: horizontes de recuperación a evaluar (en días hábiles)
RECOVERY_HORIZONS   = [1, 3, 5, 10, 20]

# Una caída "se recuperó" si el precio sube al menos este % del drop original
RECOVERY_RATIO      = 0.5    # 50% de recuperación del drop
