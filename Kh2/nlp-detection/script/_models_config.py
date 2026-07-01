# -*- coding: utf-8 -*-
"""Registry dei modelli per il benchmark multi-architettura (Slide 13 della
Project Description). Ogni voce descrive una FAMIGLIA di Transformer da
confrontare come VERIFIER (validatore binario VALID/INVALID), piu' i baseline
non-Transformer (regola dominio/codominio, SpaCy generator) che vengono aggiunti
automaticamente dal driver di benchmark.

`arch`:
  - "encoder"   -> caricabile con AutoModelForSequenceClassification senza modifiche
  - "seq2seq"   -> encoder-decoder (T5): gestito a parte nel training (text-to-text
                   oppure T5ForSequenceClassification)
  - "llm"       -> validatore via prompt LLM/SLM (Architettura C, Slide 9-10)
  - "gliner"    -> NER zero-shot label-guided (estrazione entita', Arch. A/B, Slide 9);
                   non si allena, si valuta direttamente in inferenza.
  - "glirel"    -> relation extraction zero-shot label-guided (candidate generation,
                   Arch. A/B, Slide 9); non si allena, si valuta in inferenza.

`needs_sentencepiece`: tokenizer SentencePiece (XLNet/DeBERTa/T5) -> richiede il
pacchetto `sentencepiece` in requirements.

`cpu_default`: incluso nel set "leggero" usato quando si lancia il benchmark senza
specificare i modelli (per non scaricare tutte le famiglie su CPU).
"""
from __future__ import annotations

MODELS = [
    # slug                 hf_id                              arch       sentencepiece cpu  note (asse Slide 13)
    {"slug": "distilbert",  "hf_id": "distilbert-base-uncased", "arch": "encoder", "needs_sentencepiece": False, "cpu_default": True,  "note": "leggero/veloce"},
    {"slug": "bert",        "hf_id": "bert-base-uncased",       "arch": "encoder", "needs_sentencepiece": False, "cpu_default": True,  "note": "BERT 'ontology-compiled model' (generatore/baseline encoder)"},
    {"slug": "roberta",     "hf_id": "roberta-base",            "arch": "encoder", "needs_sentencepiece": False, "cpu_default": True,  "note": "Supervised RoBERTa (validatore binario VALID/INVALID)"},
    {"slug": "deberta-v3",  "hf_id": "microsoft/deberta-v3-base","arch": "encoder","needs_sentencepiece": True,  "cpu_default": False, "note": "disentangled attention"},
    {"slug": "xlnet",       "hf_id": "xlnet-base-cased",        "arch": "encoder", "needs_sentencepiece": True,  "cpu_default": False, "note": "permutation autoregressive"},
    {"slug": "bigbird",     "hf_id": "google/bigbird-roberta-base","arch": "encoder","needs_sentencepiece": False,"cpu_default": False, "note": "sparse attention (full su input corti)"},
    {"slug": "t5",          "hf_id": "t5-base",                 "arch": "seq2seq", "needs_sentencepiece": True,  "cpu_default": False, "note": "encoder-decoder (text-to-text)"},
    # Modelli zero-shot label-guided citati nelle architetture avanzate (Slide 9):
    # NON si allenano (nessun retraining): si valutano direttamente in inferenza.
    {"slug": "gliner",      "hf_id": "urchade/gliner_medium-v2.1", "arch": "gliner", "needs_sentencepiece": False, "cpu_default": False, "note": "GLiNER: estrazione entita' zero-shot (Arch. A/B)"},
    {"slug": "glirel",      "hf_id": "jackboyla/glirel-large-v0",  "arch": "glirel", "needs_sentencepiece": False, "cpu_default": False, "note": "GLiREL: candidate generation relazioni zero-shot (Arch. A/B)"},
    # Punto-dati LLM/SLM richiesto da 'validatori BERT/LLM' (Slide 2 pilastro 4).
    # Usato solo se _llm_client trova credenziali; altrimenti viene saltato.
    {"slug": "llm",         "hf_id": "claude (api)",            "arch": "llm",     "needs_sentencepiece": False, "cpu_default": False, "note": "validatore dinamico via prompt"},
]

BY_SLUG = {m["slug"]: m for m in MODELS}


def cpu_default_slugs():
    return [m["slug"] for m in MODELS if m["cpu_default"]]


def transformer_slugs():
    return [m["slug"] for m in MODELS if m["arch"] in ("encoder", "seq2seq")]


def zeroshot_slugs():
    """Modelli zero-shot label-guided (GLiNER/GLiREL): non si allenano."""
    return [m["slug"] for m in MODELS if m["arch"] in ("gliner", "glirel")]


def resolve(slugs):
    """slugs lista o None -> lista di dict del registry (None => set cpu_default)."""
    if not slugs:
        slugs = cpu_default_slugs()
    out = []
    for s in slugs:
        if s in BY_SLUG:
            out.append(BY_SLUG[s])
        else:
            # consenti un hf_id arbitrario passato dall'utente
            out.append({"slug": s.replace("/", "-"), "hf_id": s, "arch": "encoder",
                        "needs_sentencepiece": False, "cpu_default": False, "note": "custom"})
    return out
