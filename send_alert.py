"""
send_alert.py
=============
Lee docs/data/signals_live.json y envía por Telegram:
  1) Alerta de COMPRA cuando hay señales fuertes (prob >= PROB_UMBRAL).
  2) Recordatorio de SALIDA un día hábil antes de que se cumplan los
     HORIZONTE_DIAS_HABILES (10) días desde la entrada (D+1 open tras la
     señal), para cada señal que se haya alertado antes.

Diseñado para ejecutarse desde GitHub Actions, una vez al día.

Requiere variables de entorno:
  TELEGRAM_BOT_TOKEN — token del bot (de @BotFather)
  TELEGRAM_CHAT_ID    — chat_id de Diego

Reemplaza al envío por Gmail (obsoleto desde 25 nov 2026 — ver incidente
21 nov: Gmail cerró la conexión en los 3 reintentos, 4 señales reales
nunca llegaron). Ambos tipos de mensaje comparten el mismo mecanismo de
"catch-up": si el envío falla, el mensaje queda en
docs/data/alert_state.json y se reintenta en el SIGUIENTE run (no solo
dentro del mismo run), hasta un máximo de ALERTA_MAX_EDAD_DIAS días.

Nota: el horizonte de 10 días se cuenta en días hábiles de calendario
(lunes-viernes), sin calendario de festivos NYSE — es una aproximación
razonable para un recordatorio, no para el backtest (que ya usa el
índice real de trading days de yfinance).
"""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import date, datetime, timedelta
from pathlib import Path

SIGNALS_PATH    = Path('docs/data/signals_live.json')
STATE_PATH      = Path('docs/data/alert_state.json')
POSICIONES_PATH = Path('docs/data/posiciones_abiertas.json')

PROB_UMBRAL           = 0.55
HORIZONTE_DIAS_HABILES = 10
ALERTA_MAX_EDAD_DIAS  = 5   # pasado este umbral, se descarta una pendiente (ya no es accionable)


# ---------------------------------------------------------------- Telegram --

def enviar_telegram(texto: str, intentos: int = 3) -> bool:
    token   = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print('  [alert] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID no configurados todavía — se deja pendiente.')
        return False
    url     = f'https://api.telegram.org/bot{token}/sendMessage'
    # Sin parse_mode: el Markdown "legacy" de Telegram es muy frágil con
    # texto generado dinámicamente (rompe con asteriscos/paréntesis sueltos,
    # ver incidente 25 nov 2026 — "can't parse entities"). Texto plano es
    # más robusto para este caso de uso.
    payload = json.dumps({
        'chat_id': chat_id,
        'text': texto,
    }).encode('utf-8')

    for intento in range(1, intentos + 1):
        try:
            req = urllib.request.Request(
                url, data=payload, headers={'Content-Type': 'application/json'}
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                resp.read()
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            print(f'  [alert] Intento {intento}/{intentos} falló: {e!r}')
            if intento < intentos:
                time.sleep(5 * intento)

    return False


def intentar_o_encolar(texto: str, fecha: str, pendientes: list):
    """Envía ya mismo; si falla, lo deja en la cola de catch-up."""
    if enviar_telegram(texto):
        return True
    pendientes.append({'fecha': fecha, 'cuerpo': texto})
    return False


# -------------------------------------------------------------- Días hábiles --

def es_habil(d: date) -> bool:
    return d.weekday() < 5  # 0=lunes ... 6=domingo


def siguiente_dia_habil(d: date) -> date:
    d = d + timedelta(days=1)
    while not es_habil(d):
        d += timedelta(days=1)
    return d


def dia_habil_anterior(d: date) -> date:
    d = d - timedelta(days=1)
    while not es_habil(d):
        d -= timedelta(days=1)
    return d


def sumar_dias_habiles(d: date, n: int) -> date:
    while n > 0:
        d = d + timedelta(days=1)
        if es_habil(d):
            n -= 1
    return d


# -------------------------------------------------------------------- Estado --

def cargar_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def guardar_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def dias_desde(fecha_iso: str) -> int:
    try:
        return (date.today() - date.fromisoformat(fecha_iso)).days
    except ValueError:
        return 0


# ------------------------------------------------------------------- Mensajes --

def construir_cuerpo_compra(data: dict) -> str:
    signals = [s for s in data.get('signals', []) if s['recomendacion'] == 'SEÑAL']
    fecha   = data.get('market_date', '—')
    spy     = data.get('spy_ret_hoy', 0)

    lineas = [
        f"📈 Finance Volatility Check — {fecha}",
        f"SPY hoy: {spy:+.2f}%",
        "",
        f"{len(signals)} SEÑAL(ES) DETECTADA(S)",
        "",
    ]

    for s in signals:
        tipo = s.get('tipo_caida', '—')
        lineas += [
            f"{s['ticker']}",
            f"  Caída: {s['ret_1d']:+.2f}%   Z-score: {s['zscore']:.2f}",
            f"  Prob: {s['prob']:.0%}   Tipo: {tipo}",
            f"  Precio: ${s.get('precio_cierre', '—')}   RSI-14: {s.get('rsi_14', '—')}",
            "",
        ]

    lineas += [
        "Estrategia ganadora: caída < -4.5%, Z < -2.0, horizonte 10 días hábiles.",
        "Regla validada: entra al Open D+1 o no entres (ver Seguimiento_FVC).",
        "Dashboard: https://diegorep01.github.io/finance-volatility/",
    ]

    return "\n".join(lineas)


def construir_cuerpo_recordatorio(pos: dict) -> str:
    return (
        f"🔔 Recordatorio de salida — {pos['ticker']}\n\n"
        f"Entrada: {pos['fecha_entrada']}\n"
        f"Mañana ({pos['fecha_salida']}) se cumplen los {HORIZONTE_DIAS_HABILES} días hábiles.\n"
        f"Si entraste en esta señal, mañana toca cerrar/evaluar la posición."
    )


# ----------------------------------------------------------------------- Main --

def main():
    estado     = cargar_json(STATE_PATH, {'pendientes': []})
    pendientes = estado.get('pendientes', [])
    posiciones = cargar_json(POSICIONES_PATH, [])
    hoy        = date.today()

    # 1) Reintentar primero cualquier mensaje de un run anterior que no se pudo enviar.
    siguen_pendientes = []
    for p in pendientes:
        if dias_desde(p['fecha']) > ALERTA_MAX_EDAD_DIAS:
            print(f"  [alert] Descartando pendiente vieja ({p['fecha']}) — ya no es accionable.")
            continue
        texto = f"⏳ Pendiente del {p['fecha']} (reenviada)\n\n{p['cuerpo']}"
        print(f"  [alert] Reintentando pendiente del {p['fecha']}...")
        if enviar_telegram(texto):
            print('  [alert] Pendiente enviada correctamente.')
        else:
            siguen_pendientes.append(p)
    pendientes = siguen_pendientes

    # 2) Señal de compra de hoy (si hay) + registrar posición para su recordatorio de salida.
    if SIGNALS_PATH.exists():
        data    = json.loads(SIGNALS_PATH.read_text(encoding='utf-8'))
        signals = [s for s in data.get('signals', []) if s['recomendacion'] == 'SEÑAL']
        fecha_senal = data.get('market_date', hoy.isoformat())

        if signals:
            cuerpo = construir_cuerpo_compra(data)
            print(f'  [alert] Enviando alerta de compra: {len(signals)} señal(es) — {fecha_senal}')
            if intentar_o_encolar(cuerpo, fecha_senal, pendientes):
                print('  [alert] Alerta de compra enviada.')
            else:
                print('  [alert] Falló el envío — queda en cola para el próximo run.')

            fecha_senal_d = date.fromisoformat(fecha_senal)
            entrada = siguiente_dia_habil(fecha_senal_d)
            salida  = sumar_dias_habiles(entrada, HORIZONTE_DIAS_HABILES)
            for s in signals:
                ya_existe = any(
                    p['ticker'] == s['ticker'] and p['fecha_entrada'] == entrada.isoformat()
                    for p in posiciones
                )
                if not ya_existe:
                    posiciones.append({
                        'ticker': s['ticker'],
                        'fecha_senal': fecha_senal,
                        'fecha_entrada': entrada.isoformat(),
                        'fecha_salida': salida.isoformat(),
                        'avisado_salida': False,
                    })
        else:
            print(f'  [alert] Sin señales fuertes hoy ({fecha_senal}).')
    else:
        print('  [alert] JSON de señales no encontrado — nada que procesar hoy.')

    # 3) Recordatorios de salida: un día hábil antes de fecha_salida.
    posiciones_vigentes = []
    for pos in posiciones:
        fecha_salida = date.fromisoformat(pos['fecha_salida'])

        if not pos['avisado_salida'] and dia_habil_anterior(fecha_salida) == hoy:
            texto = construir_cuerpo_recordatorio(pos)
            print(f"  [alert] Enviando recordatorio de salida: {pos['ticker']} (sale {pos['fecha_salida']})")
            if intentar_o_encolar(texto, hoy.isoformat(), pendientes):
                pos['avisado_salida'] = True
            else:
                print('  [alert] Falló el envío del recordatorio — queda en cola para el próximo run.')

        # Se descarta la posición un par de días después de la salida (ya no es relevante).
        if fecha_salida >= hoy - timedelta(days=2):
            posiciones_vigentes.append(pos)

    guardar_json(STATE_PATH, {'pendientes': pendientes, 'ultima_ejecucion': datetime.utcnow().isoformat()})
    guardar_json(POSICIONES_PATH, posiciones_vigentes)


if __name__ == '__main__':
    main()
