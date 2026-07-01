# -*- coding: utf-8 -*-
"""STAGE 4-5 (inference, LOCAL). End-to-end extraction from raw text:
  detection (NER model or dictionary) -> candidate relations (trigger + domain/range)
  -> verifier validation (BERT verifier or domain/range rule) -> grounding -> triples.

Runs out-of-the-box in rule mode (no models needed):
  python script/05_run_end_to_end.py --input data-input/ac6_corpus.txt
Upgrade with the trained verifier:
  python script/05_run_end_to_end.py --input my.txt --verifier-model models/ac6_verifier/best_model
Writes generated/extracted_triples.jsonl and generated/last_inference_result.json
(the latter is also copied to inference_demo/ for parity with the professor's project).
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import ac6_lib as L

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
INFER = BASE / "inference_demo"
onto = L.parse_ontology(BASE / "ontology" / "armoredcore.owx")
ED = json.loads((GEN / "ontology_interface.json").read_text(encoding="utf-8"))["entity_dictionary"]


def all_mentions(sent):
    mentions = L.detect_mentions(sent, ED)
    occ = [(m["start"], m["end"]) for m in mentions]
    for r in L.detect_rank_mentions(sent, onto):
        if not any(not (r["end"] <= os or r["start"] >= oe) for os, oe in occ):
            mentions.append(r)
    mentions.sort(key=lambda x: x["start"])
    return mentions


def candidates_for_sentence(sent):
    mentions = all_mentions(sent)
    ents = [m for m in mentions if m["kind"] in ("individual", "class")]
    triggers = [m for m in mentions if m["kind"] == "object_property"]
    out = []
    for tr in triggers:
        p = tr["short"]; dom = onto["odomain"].get(p); rng = onto["orange"].get(p)
        if not dom or not rng:
            continue
        subj = next((m for m in sorted([e for e in ents if e["end"] <= tr["start"]], key=lambda x: -x["end"]) if L.compat_role(onto, m, dom)), None)
        obj = next((m for m in sorted([e for e in ents if e["start"] >= tr["end"]], key=lambda x: x["start"]) if L.compat_role(onto, m, rng)), None)
        if subj and obj:
            out.append((subj, p, obj))
    return out


def verifier_text(p, subj_cls, obj_cls):
    return f"[REL] {p} [/REL] A [E1]{L.camel_to_words(subj_cls)}[/E1] {L.relation_phrase(p)} a [E2]{L.camel_to_words(obj_cls)}[/E2]."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(BASE / "data-input" / "ac6_corpus.txt"))
    ap.add_argument("--ner-model", default=None)
    ap.add_argument("--verifier-model", default=None)
    ap.add_argument("--validator", choices=["rule", "model", "llm"], default="rule",
                    help="backend di validazione: regola dominio/range, verifier allenato, o LLM (Arch. C)")
    ap.add_argument("--out", default=str(GEN / "extracted_triples.jsonl"))
    args = ap.parse_args()

    verifier = None
    if args.validator == "llm":
        import _llm_client as LLM
        if LLM.available():
            def verifier(text):
                out = (LLM.complete(
                    "You are a strict ontology relation validator. Answer with a single "
                    "word VALID or INVALID.\n" + text, max_tokens=4, temperature=0.0) or "").strip().upper()
                return out.startswith("VALID")
        else:
            print("LLM non disponibile (ANTHROPIC_API_KEY mancante): uso la regola dominio/range.")
    elif (args.validator == "model" or args.verifier_model):
        if args.verifier_model:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tok = AutoTokenizer.from_pretrained(args.verifier_model)
            mdl = AutoModelForSequenceClassification.from_pretrained(args.verifier_model)
            mdl.eval()
            def verifier(text):
                with torch.no_grad():
                    logits = mdl(**tok(text, return_tensors="pt", truncation=True, max_length=192)).logits
                return mdl.config.id2label[int(logits.argmax(-1))] == "VALID"
        else:
            print("--validator model richiede --verifier-model: uso la regola dominio/range.")
    # NER model is optional; dictionary detection already grounds entities. A trained
    # NER (models/ac6_ner/best_model) can replace detect_mentions for noisier text.

    p = Path(args.input)
    paths = sorted(p.glob("*.txt")) if p.is_dir() else [p]
    triples = []; seen = set()
    for path in paths:
        for si, sent in enumerate(L.sentence_split(path.read_text(encoding="utf-8")), 1):
            for subj, pred, obj in candidates_for_sentence(sent):
                dom = onto["odomain"].get(pred); rng = onto["orange"].get(pred)
                gs = L.ground_role(onto, subj, dom); go = L.ground_role(onto, obj, rng)
                if not gs or not go:
                    continue
                subj_cls = onto["primary_nature"](gs["short"]); obj_cls = onto["primary_nature"](go["short"])
                # validation: learned verifier if available, else domain/range already enforced
                if verifier is not None and not verifier(verifier_text(pred, subj_cls, obj_cls)):
                    continue
                key = (gs["short"], pred, go["short"])
                if key in seen:
                    continue
                seen.add(key)
                triples.append({"subject": gs["uri"], "subject_short": gs["short"], "predicate": pred,
                                "object": go["uri"], "object_short": go["short"],
                                "sentence_id": f"{path.stem}_s{si:03d}", "sentence": sent})
    with open(args.out, "w", encoding="utf-8") as h:
        for t in triples:
            h.write(json.dumps(t, ensure_ascii=False) + "\n")

    # parity with the professor: persist a single inference-result JSON.
    # Record the input path RELATIVE to the project root (never an absolute /home/... path).
    try:
        input_rel = str(p.resolve().relative_to(BASE.parent)).replace("\\", "/")
    except Exception:
        input_rel = p.name
    result = {"input": input_rel, "mode": "verifier+rule" if verifier else "rule (domain/range)",
              "accepted_triples": triples}
    (GEN / "last_inference_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    INFER.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(GEN / "last_inference_result.json", INFER / "last_inference_result.json")

    print(f"extracted {len(triples)} unique triples -> {args.out}")
    print("mode:", result["mode"])


if __name__ == "__main__":
    main()
