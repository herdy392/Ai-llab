# -*- coding: utf-8 -*-
"""STAGE 1 (weak supervision from text). Reads data-input/*.txt, detects ontology
mentions with the entity dictionary, infers candidate relations (subject/object
grounded by domain/range) and emits, mirroring the professor's weak dataset:
  - generated/kh2_sentences_all.jsonl
  - generated/kh2_relation_candidates_{all,train,val,test}.jsonl   ([E1]/[E2] + candidate_relation)
  - generated/kh2_ner_{train,val,test}.jsonl                       (BIO token classification)
  - generated/kh2_weak_stats.json
Run:  python script/02_build_weak_dataset.py

Mirrors the professor's build_logistics_weak_dataset.py.
"""
from __future__ import annotations
import json, re
from collections import Counter, defaultdict
from pathlib import Path
import kh2_lib as L

BASE = Path(__file__).resolve().parents[1]
OWX = BASE / "ontology" / "kingdom_hearts2.owx"
GEN = BASE / "generated"
DATA_IN = BASE / "data-input"
onto = L.parse_ontology(OWX)
ed = json.loads((GEN / "ontology_interface.json").read_text(encoding="utf-8"))["entity_dictionary"]
TOK = re.compile(r"\w+|[^\w\s]")


def types_closure(short, kind):
    if kind == "class":
        return {short} | onto["ancestors"](short)
    cs = set(onto["ind_classes"].get(short, []))
    out = set(cs)
    for c in cs:
        out |= onto["ancestors"](c)
    return out


def compat(mention, target):
    return target in types_closure(mention["short"], mention["kind"])


def mark(sent, subj, obj):
    spans = sorted([(subj["start"], subj["end"], "E1"), (obj["start"], obj["end"], "E2")], key=lambda x: x[0], reverse=True)
    for s, e, tag in spans:
        sent = sent[:s] + f"[{tag}]" + sent[s:e] + f"[/{tag}]" + sent[e:]
    return sent


def all_mentions(sent):
    """Dictionary mentions + context-grounded rank mentions (single-letter ranks)."""
    mentions = L.detect_mentions(sent, ed)
    rank = L.detect_rank_mentions(sent, onto)
    # avoid double-counting overlaps
    occ = [(m["start"], m["end"]) for m in mentions]
    for r in rank:
        if not any(not (r["end"] <= os or r["start"] >= oe) for os, oe in occ):
            mentions.append(r)
    mentions.sort(key=lambda x: x["start"])
    return mentions


def infer_candidates(sent, mentions):
    ents = [m for m in mentions if m["kind"] in ("individual", "class")]
    triggers = [m for m in mentions if m["kind"] == "object_property"]
    cands = []
    for tr in triggers:
        p = tr["short"]
        dom = onto["odomain"].get(p); rng = onto["orange"].get(p)
        if not dom or not rng:
            continue
        subj = None
        for m in sorted([e for e in ents if e["end"] <= tr["start"]], key=lambda x: -x["end"]):
            if L.compat_role(onto, m, dom):
                subj = m; break
        obj = None
        for m in sorted([e for e in ents if e["start"] >= tr["end"]], key=lambda x: x["start"]):
            if L.compat_role(onto, m, rng):
                obj = m; break
        if subj and obj:
            gs = L.ground_role(onto, subj, dom) or {"uri": subj["uri"], "short": subj["short"]}
            go = L.ground_role(onto, obj, rng) or {"uri": obj["uri"], "short": obj["short"]}
            cands.append({"candidate_relation": p, "candidate_relation_uri": "#" + p,
                          "weak_signal": "property_with_grounded_domain_and_range",
                          "subject_uri": gs["uri"], "subject_short": gs["short"],
                          "object_uri": go["uri"], "object_short": go["short"],
                          "marked": mark(sent, subj, obj), "trigger": tr})
    return cands


def bio_tags(sent, mentions):
    toks = [(m.group(), m.start(), m.end()) for m in TOK.finditer(sent)]
    tags = ["O"] * len(toks)
    for men in [m for m in mentions if m["kind"] == "individual"]:
        typ = L.ner_type(onto, men["short"]); first = True
        for i, (tk, s, e) in enumerate(toks):
            if s < men["end"] and e > men["start"]:
                tags[i] = ("B-" if first else "I-") + typ; first = False
    return [t for t, _, _ in toks], tags


sentence_rows = []; relation_rows = []; ner_rows = []
mk = Counter(); sig = Counter()
for path in sorted(DATA_IN.glob("*.txt")):
    phase = "kh2_instances"
    for si, sent in enumerate(L.sentence_split(path.read_text(encoding="utf-8")), 1):
        sid = f"{path.stem}_s{si:03d}"; split = L.assign_split(sent)
        mentions = all_mentions(sent)
        for m in mentions:
            mk[m["kind"]] += 1
        cands = infer_candidates(sent, mentions)
        sentence_rows.append({"sentence_id": sid, "source_file": path.name, "phase": phase, "split": split,
            "text": sent, "mentions": mentions,
            "candidate_relations": [{"uri": c["candidate_relation_uri"], "short": c["candidate_relation"],
                "source": c["weak_signal"], "triple_mode": "ontology_assertion",
                "subject_uri": c["subject_uri"], "subject_short": c["subject_short"],
                "object_uri": c["object_uri"], "object_short": c["object_short"]} for c in cands]})
        for c in cands:
            sig[c["weak_signal"]] += 1
            relation_rows.append({"sentence_id": sid, "source_file": path.name, "phase": phase, "split": split,
                "text": c["marked"], "sentence": sent, "candidate_relation": c["candidate_relation"],
                "candidate_relation_uri": c["candidate_relation_uri"], "weak_signal": c["weak_signal"],
                "subject_uri": c["subject_uri"], "object_uri": c["object_uri"], "mentions": mentions})
        tokens, tags = bio_tags(sent, mentions)
        if tokens:
            ner_rows.append({"sentence_id": sid, "split": split, "tokens": tokens, "ner_tags": tags})


def wj(p, rows):
    with open(p, "w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


wj(GEN / "kh2_sentences_all.jsonl", sentence_rows)
wj(GEN / "kh2_relation_candidates_all.jsonl", relation_rows)
for sp in ("train", "val", "test"):
    wj(GEN / f"kh2_relation_candidates_{sp}.jsonl", [r for r in relation_rows if r["split"] == sp])
    wj(GEN / f"kh2_ner_{sp}.jsonl", [r for r in ner_rows if r["split"] == sp])
ner_labels = sorted({t for r in ner_rows for t in r["ner_tags"]})
stats = {"sentences": len(sentence_rows), "sentences_with_mentions": sum(1 for r in sentence_rows if r["mentions"]),
    "sentences_with_candidate_relations": sum(1 for r in sentence_rows if r["candidate_relations"]),
    "relation_candidate_rows": len(relation_rows), "ner_rows": len(ner_rows),
    "split_distribution": dict(Counter(r["split"] for r in sentence_rows)),
    "mention_kind_distribution": dict(mk), "weak_signal_distribution": dict(sig),
    "candidate_relation_distribution": dict(Counter(r["candidate_relation"] for r in relation_rows)),
    "ner_labels": ner_labels}
(GEN / "kh2_weak_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
print("sentences:", len(sentence_rows), "| relation candidates:", len(relation_rows), "| ner rows:", len(ner_rows))
print("candidate relation distribution:", dict(Counter(r["candidate_relation"] for r in relation_rows)))
print("ner labels:", ner_labels)
