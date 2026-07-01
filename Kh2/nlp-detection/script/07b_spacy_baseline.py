# -*- coding: utf-8 -*-
"""STAGE 7b (baseline generatore, LOCALE). Baseline SpaCy per la RELATION
EXTRACTION richiesta esplicitamente dalla Slide 7 ("The SpaCy Baseline") e dal
Deliverable #4 ("baseline SpaCy vs. Transformer architectures").

A differenza della pipeline principale (dizionario + gate dominio/codominio),
questo baseline e' volutamente FRAGILE, come da slide:
  - individua i trigger lessicalmente (copula / frasi-trigger dell'ontologia);
  - accoppia l'entita' piu' vicina a sinistra (subject) e a destra (object)
    entro `distance_pairing <= 10 token`;
  - NON applica alcun vincolo semantico di dominio/codominio.
Cosi' su frasi implicite/annidate genera falsi positivi/negativi: e' il punto.

Usa spaCy se installato (en_core_web_sm o pipeline blank), altrimenti un
tokenizer interno: il baseline produce comunque le sue triple e le sue metriche.

Output:
  generated/spacy_baseline_triples.jsonl   (schema compatibile con 08)
  generated/spacy_baseline_metrics.json    (P/R/F1 a livello tripla vs gold)
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

import kh2_lib as L

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
ONTO = L.parse_ontology(BASE / "ontology" / "kingdom_hearts2.owx")
ED = json.loads((GEN / "ontology_interface.json").read_text(encoding="utf-8"))["entity_dictionary"]

MAX_TOKEN_DISTANCE = 10


def _tokenize(sent):
    """Ritorna lista di (token_text, char_start, char_end). spaCy se c'e'."""
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
        except Exception:
            nlp = spacy.blank("en")
        doc = nlp(sent)
        return [(t.text, t.idx, t.idx + len(t.text)) for t in doc]
    except Exception:
        toks, i = [], 0
        for piece in sent.replace(".", " .").split():
            j = sent.find(piece, i)
            if j < 0:
                j = i
            toks.append((piece, j, j + len(piece)))
            i = j + len(piece)
        return toks


def _char_to_token_index(tokens, char_pos):
    for idx, (_, s, e) in enumerate(tokens):
        if s <= char_pos < e or char_pos <= s:
            return idx
    return len(tokens) - 1


def all_mentions(sent):
    mentions = L.detect_mentions(sent, ED)
    occ = [(m["start"], m["end"]) for m in mentions]
    for r in L.detect_rank_mentions(sent, ONTO):
        if not any(not (r["end"] <= os or r["start"] >= oe) for os, oe in occ):
            mentions.append(r)
    mentions.sort(key=lambda x: x["start"])
    return mentions


def _ground_entity_short(m):
    """Sceglie un individuo per la mention SENZA usare dominio/codominio."""
    for e in m.get("entries", []):
        if e.get("kind") == "individual":
            return e["short"]
    return m.get("short")


def extract(sent, sid):
    tokens = _tokenize(sent)
    mentions = all_mentions(sent)
    ents = [m for m in mentions if m["kind"] in ("individual", "class")]
    triggers = [m for m in mentions if m["kind"] == "object_property"]
    out = []
    for tr in triggers:
        ti = _char_to_token_index(tokens, tr["start"])
        # entita' piu' vicina a sinistra / destra entro la finestra di token
        left = [e for e in ents if e["end"] <= tr["start"]]
        right = [e for e in ents if e["start"] >= tr["end"]]
        subj = max(left, key=lambda e: e["end"], default=None)
        obj = min(right, key=lambda e: e["start"], default=None)
        if not subj or not obj:
            continue
        si = _char_to_token_index(tokens, subj["start"])
        oi = _char_to_token_index(tokens, obj["start"])
        if abs(ti - si) > MAX_TOKEN_DISTANCE or abs(oi - ti) > MAX_TOKEN_DISTANCE:
            continue
        out.append({
            "subject_short": _ground_entity_short(subj),
            "predicate": tr["short"],
            "object_short": _ground_entity_short(obj),
            "sentence_id": sid,
            "sentence": sent,
        })
    return out


def score(pred_path):
    gold = set()
    for r in (json.loads(l) for l in (GEN / "kh2_instance_triples.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()):
        t = r["triple"]
        gold.add((t["subject"], t["predicate"], t["object"]))
    pred = set()
    for r in (json.loads(l) for l in Path(pred_path).read_text(encoding="utf-8").splitlines() if l.strip()):
        pred.add((r["subject_short"], r["predicate"], r["object_short"]))
    tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    by = Counter(p for _, p, _ in gold)
    hit = Counter(p for tr in (gold & pred) for p in [tr[1]])
    return {"system": "spacy_baseline", "gold": len(gold), "predicted": len(pred),
            "tp": tp, "fp": fp, "fn": fn, "precision": P, "recall": R, "f1": F,
            "per_relation_recall": {r: {"hit": hit[r], "total": by[r]} for r in sorted(by)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(BASE / "data-input" / "kh2_corpus.txt"))
    ap.add_argument("--out", default=str(GEN / "spacy_baseline_triples.jsonl"))
    ap.add_argument("--retrain", "--reitrain", dest="retrain", action="store_true",
                    help="forza il ricalcolo del baseline anche se gia' salvato (default: riusa)")
    args = ap.parse_args()

    # RIUSO: il baseline SpaCy non si allena, ma e' costoso da ricalcolare ogni volta.
    # Se le triple e le metriche esistono gia' e non si forza, si riusano.
    out_p = Path(args.out)
    metrics_p = GEN / "spacy_baseline_metrics.json"
    if not args.retrain and out_p.exists() and metrics_p.exists():
        m = json.loads(metrics_p.read_text(encoding="utf-8"))
        print(f"[reuse] SpaCy baseline: trovato risultato precedente -> {out_p.name} "
              f"(precision={m['precision']:.3f} recall={m['recall']:.3f} f1={m['f1']:.3f}). "
              f"Usa --retrain per ricalcolare.")
        return

    p = Path(args.input)
    paths = sorted(p.glob("*.txt")) if p.is_dir() else [p]
    triples, seen = [], set()
    for path in paths:
        for si, sent in enumerate(L.sentence_split(path.read_text(encoding="utf-8")), 1):
            for tr in extract(sent, f"{path.stem}_s{si:03d}"):
                key = (tr["subject_short"], tr["predicate"], tr["object_short"])
                if key in seen:
                    continue
                seen.add(key)
                triples.append(tr)
    Path(args.out).write_text(
        "\n".join(json.dumps(t, ensure_ascii=False) for t in triples) + "\n", encoding="utf-8")
    metrics = score(args.out)
    (GEN / "spacy_baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"SpaCy baseline: {len(triples)} triple -> {args.out}")
    print(f"  precision={metrics['precision']:.3f} recall={metrics['recall']:.3f} f1={metrics['f1']:.3f}")
    print("metrics -> generated/spacy_baseline_metrics.json")


if __name__ == "__main__":
    main()
