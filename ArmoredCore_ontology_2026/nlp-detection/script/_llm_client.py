# -*- coding: utf-8 -*-
"""Client LLM minimale e OPZIONALE.

Serve a due cose previste dalla traccia ma che NON devono bloccare la pipeline:
  1) generazione sintetica dei dati "via LLM" (Slide 2 pilastro 2 / Slide 5);
  2) il punto-dati 'validatore LLM/SLM' del benchmark (Architettura C, Slide 9-10).

Se non ci sono credenziali (ANTHROPIC_API_KEY) o manca la rete, `available()`
ritorna False e i chiamanti usano il fallback deterministico: la pipeline resta
eseguibile out-of-the-box senza alcun servizio esterno.
"""
from __future__ import annotations
import json
import os
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def complete(prompt: str, system: str = "", max_tokens: int = 512, temperature: float = 0.7):
    """Ritorna il testo della risposta, o None in caso di errore/assenza chiave."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    body = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        body["system"] = system
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    except Exception:
        return None
