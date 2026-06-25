"""
send_alert.py
=============
Lee docs/data/signals_live.json y envía un correo si hay señales fuertes
(prob >= PROB_UMBRAL). Diseñado para ejecutarse desde GitHub Actions.

Requiere variables de entorno:
  GMAIL_USER         — tu dirección Gmail (ej. usuario@gmail.com)
  GMAIL_APP_PASSWORD — App Password de 16 caracteres generada en myaccount.google.com
"""

import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path

JSON_PATH  = Path('docs/data/signals_live.json')
PROB_UMBRAL = 0.55


def construir_cuerpo(data: dict) -> str:
    signals = [s for s in data.get('signals', []) if s['recomendacion'] == 'SEÑAL']
    fecha   = data.get('market_date', '—')
    spy     = data.get('spy_ret_hoy', 0)

    lineas = [
        f"Finance Volatility Check — {fecha}",
        f"SPY hoy: {spy:+.2f}%",
        "",
        f"{'='*45}",
        f"  {len(signals)} SEÑAL(ES) DETECTADA(S)",
        f"{'='*45}",
        "",
    ]

    for s in signals:
        tipo  = s.get('tipo_caida', '—')
        lineas += [
            f"  TICKER : {s['ticker']}",
            f"  Caída  : {s['ret_1d']:+.2f}%   Z-score: {s['zscore']:.2f}",
            f"  Prob.  : {s['prob']:.0%}   Tipo: {tipo}",
            f"  Precio : ${s.get('precio_cierre', '—')}   RSI-14: {s.get('rsi_14', '—')}",
            "",
        ]

    lineas += [
        "---",
        "Estrategia: comprar si prob >= 0.55, horizonte 10 días hábiles.",
        "Dashboard:  https://diegorep01.github.io/finance-volatility/",
    ]

    return "\n".join(lineas)


def enviar_correo(asunto: str, cuerpo: str):
    remitente   = os.environ['GMAIL_USER']
    destinatario = os.environ['GMAIL_USER']   # te lo mandas a ti mismo
    password    = os.environ['GMAIL_APP_PASSWORD']

    msg = MIMEMultipart('alternative')
    msg['Subject'] = asunto
    msg['From']    = remitente
    msg['To']      = destinatario
    msg.attach(MIMEText(cuerpo, 'plain', 'utf-8'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(remitente, password)
        server.sendmail(remitente, destinatario, msg.as_string())


def main():
    if not JSON_PATH.exists():
        print('  [alert] JSON no encontrado — nada que enviar.')
        return

    data    = json.loads(JSON_PATH.read_text(encoding='utf-8'))
    signals = [s for s in data.get('signals', []) if s['recomendacion'] == 'SEÑAL']

    if not signals:
        print(f'  [alert] Sin señales fuertes hoy ({data.get("market_date")}) — no se envía correo.')
        return

    fecha  = data.get('market_date', '—')
    asunto = f'[Finance] {len(signals)} señal(es) detectada(s) — {fecha}'
    cuerpo = construir_cuerpo(data)

    print(f'  [alert] Enviando correo: {asunto}')
    enviar_correo(asunto, cuerpo)
    print('  [alert] Correo enviado correctamente.')


if __name__ == '__main__':
    main()
