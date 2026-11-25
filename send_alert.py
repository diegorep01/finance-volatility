"""
send_alert.py
=============
Lee docs/data/signals_live.json y envía una alerta por Telegram si hay
señales fuertes (prob >= PROB_UMBRAL). Diseñado para ejecutarse desde
GitHub Actions.

Requiere variables de entorno:
  TELEGRAM_BOT_TOKEN — token del bot (de @BotFather)
  TELEGRAM_CHAT_ID    — chat_id de Diego

Reemplaza al envío por Gmail (obsoleto desde 25 nov 2026 — ver incidente
21 nov: Gmail cerró la conexión en las 3 reintentos, 4 señales reales
nunca llegaron). Telegram es más fiable para este caso de uso y además
añade "catch-up": si el envío falla, la alerta queda en
docs/data/alert_state.json y se reintenta en el SIGUIENTE run (no solo
dentro del mismo run), hasta un máximo de ALERTA_MAX_EDAD_DIAS días.
"""

import json
import os
import time
import urllib.request
import urllib.error
from datetime import date, datetime
from pathlib import Path

SIGNALS_PATH = Path('docs/data/signals_live.json')
STATE_PATH   = Path('docs/data/alert_state.json')
PROB_UMBRAL  = 0.55
ALERTA_MAX_EDAD_DIAS = 5   # pasado este umbral, se descarta una pendiente (ya no es accionable)


def construir_cuerpo(data: dict) -> str:
    signals = [s for s in data.get('signals', []) if s['recomendacion'] == 'SEÑAL']
    fecha   = data.get('market_date', '—')
    spy     = data.get('spy_ret_hoy', 0)

    lineas = [
        f"📈 *Finance Volatility Check* — {fecha}",
        f"SPY hoy: {spy:+.2f}%",
        "",
        f"*{len(signals)} SEÑAL(ES) DETECTADA(S)*",
        "",
    ]

    for s in signals:
        tipo = s.get('tipo_caida', '—')
        lineas += [
            f"*{s['ticker']}*",
            f"  Caída: {s['ret_1d']:+.2f}%   Z-score: {s['zscore']:.2f}",
            f"  Prob: {s['prob']:.0%}   Tipo: {tipo}",
            f"  Precio: ${s.get('precio_cierre', '—')}   RSI-14: {s.get('rsi_14', '—')}",
            "",
        ]

    lineas += [
        "Estrategia ganadora: caída < -4.5%, Z < -2.0, horizonte 10 días hábiles.",
        "Dashboard: https://diegorep01.github.io/finance-volatility/",
    ]

    return "\n".join(lineas)


def enviar_telegram(texto: str, intentos: int = 3) -> bool:
    token   = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        print('  [alert] TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID no configurados todavía — se deja pendiente.')
        return False
    url     = f'https://api.telegram.org/bot{token}/sendMessage'
    payload = json.dumps({
        'chat_id': chat_id,
        'text': texto,
        'parse_mode': 'Markdown',
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


def cargar_estado() -> dict:
    if not STATE_PATH.exists():
        return {'pendientes': []}
    return json.loads(STATE_PATH.read_text(encoding='utf-8'))


def guardar_estado(estado: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(estado, ensure_ascii=False, indent=2), encoding='utf-8')


def dias_desde(fecha_iso: str) -> int:
    try:
        return (date.today() - date.fromisoformat(fecha_iso)).days
    except ValueError:
        return 0


def main():
    estado = cargar_estado()
    pendientes = estado.get('pendientes', [])

    # 1) Reintentar primero cualquier alerta de un run anterior que no se pudo enviar.
    siguen_pendientes = []
    for p in pendientes:
        if dias_desde(p['fecha']) > ALERTA_MAX_EDAD_DIAS:
            print(f"  [alert] Descartando pendiente vieja ({p['fecha']}) — ya no es accionable.")
            continue
        texto = f"⏳ *Alerta pendiente del {p['fecha']} (reenviada)*\n\n{p['cuerpo']}"
        print(f"  [alert] Reintentando pendiente del {p['fecha']}...")
        if enviar_telegram(texto):
            print('  [alert] Pendiente enviada correctamente.')
        else:
            siguen_pendientes.append(p)

    # 2) Procesar la señal de hoy (si hay).
    nuevas_pendientes = siguen_pendientes
    if SIGNALS_PATH.exists():
        data    = json.loads(SIGNALS_PATH.read_text(encoding='utf-8'))
        signals = [s for s in data.get('signals', []) if s['recomendacion'] == 'SEÑAL']

        if signals:
            fecha  = data.get('market_date', date.today().isoformat())
            cuerpo = construir_cuerpo(data)
            print(f'  [alert] Enviando alerta Telegram: {len(signals)} señal(es) — {fecha}')
            if enviar_telegram(cuerpo):
                print('  [alert] Alerta enviada correctamente por Telegram.')
            else:
                print('  [alert] Falló el envío — queda guardada para reintentar en el próximo run.')
                nuevas_pendientes.append({'fecha': fecha, 'cuerpo': cuerpo})
        else:
            print(f'  [alert] Sin señales fuertes hoy ({data.get("market_date")}).')
    else:
        print('  [alert] JSON de señales no encontrado — nada que procesar hoy.')

    guardar_estado({'pendientes': nuevas_pendientes, 'ultima_ejecucion': datetime.utcnow().isoformat()})


if __name__ == '__main__':
    main()
