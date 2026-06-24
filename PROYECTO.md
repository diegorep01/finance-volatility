# Finance Volatility Check

Proyecto de inversión algorítmica. Detecta caídas anómalas de mercado con alta probabilidad de corrección usando ML.

**Hipótesis:** Los mercados producen caídas estadísticamente anómalas (no fundamentales) que tienden a corregirse en días o semanas. Si el algoritmo las detecta con alta precisión, se puede invertir en el dip y ganar la reversión.

Sin NLP — solo datos cuantitativos: precio, volumen, indicadores técnicos.

---

## Stack

- Python 3.12 (Anaconda: `C:\Users\diego\anaconda3\python.exe`)
- yfinance · pandas · numpy · matplotlib · scikit-learn (próximo)
- Datos: Yahoo Finance (gratis, 26 tickers descargados en `data/`)

---

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Todos los parámetros (tickers, umbrales, horizontes) |
| `download_data.py` | Descarga OHLCV con yfinance → `data/*.csv` |
| `detect_anomalies.py` | Detecta caídas anómalas: retorno < -3% Y z-score < -2 |
| `backtest_recovery.py` | Tasa de recuperación por horizonte + gráficos |
| `main.py` | Pipeline completo (`--nodown` para saltar descarga) |
| `recovery_deep_analysis.py` | Análisis por buckets de magnitud: leve/moderada/fuerte/extrema |
| `phase2_trend_filter.py` | Fase 2: contexto de tendencia (SMA50, SMA200, momentum) |

**Outputs en `output/`:**
- `resultados_backtest.csv` — eventos + retornos por horizonte
- `recovery_deep.csv` — análisis por magnitud de caída
- `phase2_trend_context.csv` — análisis con filtro de tendencia
- `distribucion_Xd.png` — gráficos de distribución de retornos

---

## Cómo ejecutar

```powershell
$env:PYTHONIOENCODING = 'utf-8'
cd "D:\3.Proyectos Personales\Finance_Volatility_Check"

# Pipeline completo (descarga + detecta + backtest)
& "C:\Users\diego\anaconda3\python.exe" main.py

# Solo con datos ya descargados
& "C:\Users\diego\anaconda3\python.exe" main.py --nodown

# Análisis por magnitud de caída
& "C:\Users\diego\anaconda3\python.exe" recovery_deep_analysis.py

# Fase 2 — filtro de tendencia
& "C:\Users\diego\anaconda3\python.exe" phase2_trend_filter.py
```

---

## Resultados al 22 jun 2026

### Fase 1 — Backtesting básico (1.341 caídas anómalas, 26 tickers, 2015-2026)

| Horizonte | Tasa recuperación (≥50% del drop) | Ret. promedio |
|---|---|---|
| 1 día | 14.6% | +0.24% |
| 5 días | 32.9% | +0.05% |
| 10 días | 42.4% | +0.38% |
| 20 días | 46.5% | +1.45% |

### Análisis por magnitud (recuperación total al precio previo)

| Bucket | N eventos | Mediana rec. | Recuperado a 20d | Nunca |
|---|---|---|---|---|
| Leve (-1/-3%) | 11.417 | 8 días | 70.7% | 1.2% |
| Moderada (-3/-5%) | 2.325 | 13 días | 60.0% | 2.7% |
| Fuerte (-5/-10%) | 802 | 21 días | 46.7% | 5.3% |
| Extrema (>-10%) | 99 | 27 días | 32.7% | 8.2% |

### Fase 2 — Hallazgo clave

El filtro de tendencia (SMA50 > SMA200 etc.) **no mejora** la tasa de recuperación para caídas pequeñas/moderadas — las caídas en downtrend rebotan igual o más rápido (sobrevendidas). Solo ayuda en caídas extremas (+9pp).

**Conclusión:** la tendencia sola no discrimina bien. Necesitamos features más granulares.

---

### Fase 3 — Features granulares (23 jun 2026) — 1.269 eventos

#### Feature 1: Tipo de activo
| Tipo | N | Rec 20d | Rec 60d | Nunca |
|---|---|---|---|---|
| ETF | 531 | 31.4% | 51.9% | 2.4% |
| Acción | 738 | **35.9%** | **56.8%** | 1.9% |

**Hallazgo:** Las acciones se recuperan más que los ETFs. Los ETFs con caída extrema son los más peligrosos (18.4% nunca vuelven).

#### Feature 2: Volumen relativo
| Volumen | N | Rec 20d | Nunca |
|---|---|---|---|
| Bajo (<0.8x) | 14 | 28.6% | 0.0% |
| Normal (0.8-1.5x) | 441 | **38.7%** | 3.4% |
| Alto (1.5-2.5x) | 625 | 33.7% | 0.8% |
| Muy alto (>2.5x) | 189 | 24.9% | 3.7% |

**Hallazgo contraintuitivo:** volumen de pánico extremo (>2.5x) es el PEOR segmento para rebote a 20d. El volumen normal es el mejor. El pánico masivo NO garantiza reversión rápida.

#### Feature 3: RSI en el momento de la caída
| RSI | N | Rec 20d | Nunca |
|---|---|---|---|
| Sobrevendido (<30) | 266 | **36.2%** | 0.4% |
| Neutral (30-70) | 1002 | 33.5% | 2.6% |

**Hallazgo:** RSI sobrevendido mejora ligeramente la recuperación pero la diferencia es pequeña (+2.7pp). No es un discriminador potente solo.

#### Feature 4: Días consecutivos bajando
| Racha | N | Rec 20d | Rec 60d |
|---|---|---|---|
| 0 — caída aislada | 612 | **34.1%** | **56.8%** |
| 1-2 días bajando | 512 | 36.2% | 53.1% |
| 3-4 días bajando | 108 | 28.7% | 55.6% |
| 5+ días bajando | 37 | 18.9% | 43.2% |

**Hallazgo:** rachas de 5+ días son una señal de alerta — solo 18.9% se recuperan en 20d.

#### Cruce: Mejor y peor segmento
- **Mejor:** Acción moderada (-3% a -5%), vol normal, RSI sobrevendido → ~38-40% rec a 20d
- **Peor:** ETF con caída extrema (>-10%) → 18.4% nunca se recuperan

---

## Hoja de ruta

- [x] **Fase 1** — Descarga + detección Z-score + backtesting ✅
- [x] **Fase 2** — Análisis por magnitud + filtro de tendencia ✅ → conclusión: tendencia sola no basta
- [x] **Fase 3** — Features granulares: vol_ratio, RSI, días consecutivos, ETF vs acción ✅ → dataset listo en `output/phase3_features.csv`
- [x] **Fase 4** — Modelo XGBoost supervisado ✅ → AUC test 0.56 | precision 57% a umbral 0.70 | overfitting detectado → pendiente regularización
- [ ] **Fase 5** — Dashboard GitHub Pages con señales activas del día
- [ ] **Fase 6** — Teoría 2: predicción de quiebras/delisting

---

## Dos teorías del proyecto

1. **Ruido de mercado** — caídas anómalas con reversión a corto plazo → estrategia de días/semanas
2. **Predicción de quiebras** — identificar empresas que desaparecerán usando histórico financiero

**Uso final:** portafolio de IA para recruiters + herramienta personal de inversión (eToro/Coinbase).
