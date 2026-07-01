# -*- coding: utf-8 -*-
"""STAGE 10 (query the trained model, LOCAL). Two ways to interrogate the models
produced by 03_train_verifier.py and 04_train_ner.py.

VERIFIER - is a (subject, relation, object) claim VALID according to the model?
  python script\\10_query_model.py verifier --rel pilots ^
      --subject Character --object ArmoredCore
  python script\\10_query_model.py verifier --text "[REL] kills [/REL] A [E1]character[/E1] kills a [E2]character[/E2]."

NER - tag the ontology entities found in a free sentence:
  python script\\10_query_model.py ner --text "Raven pilots the Steel Haze in Watchpoint."

Defaults assume models live in models/ac6_verifier/best_model and models/ac6_ner/best_model.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]

REL_PHRASE = {"hasAlias": "has alias", "hasRank": "has rank", "participatesIn": "participates in",
              "isBossOf": "is boss of", "takesPlaceIn": "takes place in", "pilots": "pilots",
              "kills": "kills", "worksFor": "works for", "isFriendlyTowards": "is friendly towards",
              "isHostileToward": "is hostile toward"}


def camel(s):
    return " ".join(p.lower() for p in re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s).replace("_", " ").split())


def build_verifier_text(rel, subject, obj):
    phrase = REL_PHRASE.get(rel, camel(rel))
    return f"[REL] {rel} [/REL] A [E1]{camel(subject)}[/E1] {phrase} a [E2]{camel(obj)}[/E2]."


def run_verifier(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    mdir = args.model or str(BASE / "models" / "ac6_verifier" / "best_model")
    tok = AutoTokenizer.from_pretrained(mdir)
    mdl = AutoModelForSequenceClassification.from_pretrained(mdir); mdl.eval()
    text = args.text or build_verifier_text(args.rel, args.subject, args.object)
    with torch.no_grad():
        logits = mdl(**tok(text, return_tensors="pt", truncation=True, max_length=192)).logits
        probs = logits.softmax(-1)[0]
    idx = int(probs.argmax())
    label = mdl.config.id2label[idx]
    print("input :", text)
    print("verdict:", label, f"(confidence {float(probs[idx]):.3f})")
    print("probs :", {mdl.config.id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))})


def run_ner(args):
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    mdir = args.model or str(BASE / "models" / "ac6_ner" / "best_model")
    tok = AutoTokenizer.from_pretrained(mdir)
    mdl = AutoModelForTokenClassification.from_pretrained(mdir); mdl.eval()
    words = args.text.split()
    enc = tok(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        pred = mdl(**enc).logits.argmax(-1)[0].tolist()
    word_ids = enc.word_ids(0); id2label = mdl.config.id2label
    out = []; seen = set()
    for i, wid in enumerate(word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid); out.append((words[wid], id2label[pred[i]]))
    print("sentence:", args.text)
    ent, cur = [], None
    for w, t in out:
        if t.startswith("B-"):
            if cur: ent.append(cur)
            cur = [w, t[2:]]
        elif t.startswith("I-") and cur and cur[1] == t[2:]:
            cur[0] += " " + w
        else:
            if cur: ent.append(cur); cur = None
    if cur: ent.append(cur)
    print("entities:", [(w, t) for w, t in ent] or "(none)")
    print("tokens  :", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verifier"); v.add_argument("--model"); v.add_argument("--text")
    v.add_argument("--rel", default="pilots"); v.add_argument("--subject", default="Character")
    v.add_argument("--object", default="ArmoredCore"); v.set_defaults(func=run_verifier)
    n = sub.add_parser("ner"); n.add_argument("--model")
    n.add_argument("--text", required=True); n.set_defaults(func=run_ner)
    args = ap.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
