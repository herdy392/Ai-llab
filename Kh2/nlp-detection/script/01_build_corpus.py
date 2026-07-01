# -*- coding: utf-8 -*-
"""STAGE 0+2 (data generation, ONTOLOGY-DRIVEN). Dall'ontologia Kingdom Hearts 2 produce:
  - generated/ontology_interface.json        (entity dictionary + relation vocab + triggers)
  - generated/synthetic_phase_corpus/...      (corpus schema/TBox sintetico + annotazioni)
  - generated/kh2_verifier_{train,val,test,all}.jsonl  (relation verifier VALID/INVALID)
  - data-input/kh2_corpus.txt                 (testo ISTANZA/ABox, nomi reali)
  - generated/kh2_instance_triples.jsonl      (gold instance triples per la valutazione)

Tutto e' derivato dall'ontologia (classi, domini/codomini, sottoclassi, ABox): NON
contiene schemi cablati specifici di un gioco. Cambiando l'ontologia, cambiano i dati.
Run:  python script/01_build_corpus.py
"""
from __future__ import annotations
import json, random
from collections import Counter
from pathlib import Path
import kh2_lib as L

BASE = Path(__file__).resolve().parents[1]
OWX = BASE / "ontology" / "kingdom_hearts2.owx"
GEN = BASE / "generated"
SYN = GEN / "synthetic_phase_corpus"
DOCS = SYN / "documents"
DATA_IN = BASE / "data-input"
for d in (GEN, SYN, DOCS, DATA_IN):
    d.mkdir(parents=True, exist_ok=True)
RNG = random.Random(20260623)
onto = L.parse_ontology(OWX)


def words(x):
    return L.camel_to_words(x)


def wj(p, rows):
    with open(p, "w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# 1) SCHEMA TRIPLES derivati dall'ontologia (TBox)
# --------------------------------------------------------------------------- #
OBJ_T = [(onto["odomain"][p], p, onto["orange"][p]) for p in sorted(onto["objprops"])
         if onto["odomain"].get(p) and onto["orange"].get(p)]
DATA_T = [(onto["ddomain"][p], p, "LiteralValue") for p in sorted(onto["dataprops"])
          if onto["ddomain"].get(p)]
SUB_T = [(c, "rdfs:subClassOf", sup) for c in sorted(onto["subclass"]) for sup in onto["subclass"][c]]
ALL_T = OBJ_T + DATA_T + SUB_T

# proprieta'-oggetto realmente utilizzabili (con dominio+codominio) -> usate dal verifier
OBJ_PREDS = [p for p in sorted(onto["objprops"]) if onto["odomain"].get(p) and onto["orange"].get(p)]


def templates_for(s, p, o):
    """Frasi-schema generiche per una tripla TBox (superfici naturali)."""
    if p == "rdfs:subClassOf":
        return [f"A {words(s)} is a kind of {words(o)}.",
                f"Every {words(s)} is also a {words(o)}."]
    if o == "LiteralValue":          # data property
        return [f"A {words(s)} {L.relation_phrase(p)} value.",
                f"Each {words(s)} records its {words(p)}."]
    ph = L.relation_phrase(p)
    return [f"A {words(s)} {ph} a {words(o)}.",
            f"In general a {words(s)} {ph} some {words(o)}.",
            f"Typically a {words(s)} {ph} a given {words(o)}.",
            f"Each {words(s)} {ph} a {words(o)}.",
            f"Often a {words(s)} {ph} another {words(o)}."]


# ======== ontology_interface.json (+ trigger per la weak detection) ========
# I trigger di relazione sono le verbalizzazioni delle object property.
TRIGGERS = {p: sorted({L.relation_phrase(p), words(p)}) for p in onto["objprops"]}
ed = L.build_entity_dictionary(onto)
ed = {k: list(v) for k, v in ed.items()}
for pred, phrases in TRIGGERS.items():
    entry_t = {"uri": "#" + pred, "short": pred, "kind": "object_property",
               "canonical_label": words(pred)}
    for ph in phrases:
        key = ph.lower()
        if entry_t not in ed.setdefault(key, []):
            ed[key].append(entry_t)
relation_vocabulary = (
    [{"uri": "#" + p, "short": p, "label": words(p), "kind": "object_property"} for p in sorted(onto["objprops"])]
    + [{"uri": "#" + p, "short": p, "label": words(p), "kind": "datatype_property"} for p in sorted(onto["dataprops"])])
# gruppi semantici = coppie inverse dichiarate (KH2 non ne ha -> vuoto)
semantic_groups = {a: [b] for a, b in onto["inverses"]}
interface = {"ontology": "ontology/kingdom_hearts2.owx", "entity_dictionary": ed,
             "relation_vocabulary": relation_vocabulary, "semantic_groups": semantic_groups,
             "triggers": TRIGGERS}
(GEN / "ontology_interface.json").write_text(json.dumps(interface, indent=2, ensure_ascii=False), encoding="utf-8")

# ======== corpus schema sintetico (documenti TBox) ========
def split_doc(i):
    return "train" if i < 80 else ("val" if i < 90 else "test")


docs = []; srows = []
N_DOCS = 100
for di in range(N_DOCS):
    did = f"kingdom_hearts2_doc_{di+1:03d}"; sp = split_doc(di)
    k = RNG.randint(9, 12)
    chosen = RNG.sample(ALL_T, k=min(k, len(ALL_T)))
    sa = []; texts = []
    for si, tr in enumerate(chosen, 1):
        s, p, o = tr
        txt = RNG.choice(templates_for(s, p, o)); sid = f"{did}_s{si:03d}"
        trip = [{"subject": s, "predicate": p, "object": o}]
        is_obj = (p in OBJ_PREDS)
        sa.append({"sentence_id": sid, "text": txt, "triples": trip, "is_object_relation": is_obj})
        srows.append({"document_id": did, "sentence_id": sid, "split": sp, "text": txt,
                      "triples": trip, "is_object_relation": is_obj})
        texts.append(txt)
    docs.append({"document_id": did, "split": sp, "text": "\n".join(texts), "sentences": sa})

for d in docs:
    (DOCS / f"{d['document_id']}.txt").write_text(d["text"], encoding="utf-8")
wj(SYN / "synthetic_documents_all.jsonl", docs)
wj(SYN / "synthetic_sentence_annotations_all.jsonl", srows)
for sp in ("train", "val", "test"):
    wj(SYN / f"synthetic_documents_{sp}.jsonl", [d for d in docs if d["split"] == sp])
    wj(SYN / f"synthetic_sentence_annotations_{sp}.jsonl", [r for r in srows if r["split"] == sp])
(SYN / "synthetic_corpus_stats.json").write_text(json.dumps({
    "documents": len(docs), "sentences": len(srows),
    "document_split_distribution": {sp: sum(1 for d in docs if d["split"] == sp) for sp in ("train", "val", "test")},
    "sentence_split_distribution": {sp: sum(1 for r in srows if r["split"] == sp) for sp in ("train", "val", "test")},
    "supported_triples": [{"subject": s, "predicate": p, "object": o} for s, p, o in ALL_T]},
    indent=2, ensure_ascii=False), encoding="utf-8")

# ======== schema verifier (VALID/INVALID) — solo proprieta'-oggetto ========
def ent_uri(n): return L.NS + n
def rel_uri(n): return L.RDFS_NS + "subClassOf" if n == "rdfs:subClassOf" else L.NS + n
def clean(s, p, o): return f"A {words(s)} {L.relation_phrase(p)} a {words(o)}."
def entry(sent_pred, cand_pred, s, o, label, neg_type):
    marked = f"A [E1]{words(s)}[/E1] {L.relation_phrase(sent_pred)} a [E2]{words(o)}[/E2]."
    return {"text": f"[REL] {cand_pred} [/REL] {marked}", "label": label,
            "candidate_relation": rel_uri(cand_pred), "subject": ent_uri(s), "object": ent_uri(o),
            "tier": "tier_1", "neg_type": neg_type, "sentence": clean(s, sent_pred, o)}

# NEGATIVI DIFFICILI per tipo (vedi commento storico): type_identical / partial_overlap /
# type_incompatible, calcolati su dominio/codominio reali dall'ontologia.
_DR = {p: (onto["odomain"].get(p), onto["orange"].get(p)) for p in OBJ_PREDS}
def hard_negatives(p):
    dp, rp = _DR.get(p, (None, None))
    identical = [q for q in OBJ_PREDS if q != p and _DR.get(q) == (dp, rp) and dp and rp]
    partial = [q for q in OBJ_PREDS if q != p and _DR.get(q) != (dp, rp)
               and (_DR.get(q, (None, None))[0] == dp or _DR.get(q, (None, None))[1] == rp)]
    incompat = [q for q in OBJ_PREDS if q != p and _DR.get(q, (None, None))[0] != dp
                and _DR.get(q, (None, None))[1] != rp]
    negs = []
    if identical: negs.append((RNG.choice(identical), "type_identical"))
    if partial:   negs.append((RNG.choice(partial), "partial_overlap"))
    if incompat:  negs.append((RNG.choice(incompat), "type_incompatible"))
    if not negs and len(OBJ_PREDS) > 1:
        negs.append((RNG.choice([c for c in OBJ_PREDS if c != p]), "type_incompatible"))
    return negs


# Verifier a livello di INDIVIDUO (ABox): aumenta MOLTO le frasi DISTINTE usando
# i nomi reali degli individui (es. "Sora owns the keyblade Kingdom Chain"), non solo
# le classi. Le copie identiche finiscono nello stesso split (01b, hash) -> niente leakage.
INST_LEADS = ["", "In Kingdom Hearts 2, "]
def entry_inst(sent_pred, cand_pred, s, o, label, neg_type, lead=""):
    ds, do = L.display_name(onto, s), L.display_name(onto, o)
    marked = f"{lead}[E1]{ds}[/E1] {L.relation_phrase(sent_pred)} [E2]{do}[/E2]."
    return {"text": f"[REL] {cand_pred} [/REL] {marked}", "label": label,
            "candidate_relation": rel_uri(cand_pred), "subject": ent_uri(s), "object": ent_uri(o),
            "tier": "tier_1", "neg_type": neg_type,
            "sentence": f"{lead}{ds} {L.relation_phrase(sent_pred)} {do}."}


grp = {"train": [], "val": [], "test": []}; lab = Counter(); negc = Counter()
# (a) livello CLASSE: dalle frasi-schema (generalizzazione dei tipi)
for r in srows:
    for tr in r["triples"]:
        s, p, o = tr["subject"], tr["predicate"], tr["object"]
        if p not in OBJ_PREDS:          # verifier = validita' di RELAZIONI (object property)
            continue
        grp[r["split"]].append(entry(p, p, s, o, "VALID", "positive")); lab["VALID"] += 1; negc["positive"] += 1
        for w, nt in hard_negatives(p):
            grp[r["split"]].append(entry(p, w, s, o, "INVALID", nt)); lab["INVALID"] += 1; negc[nt] += 1
# (b) livello INDIVIDUO: per ogni asserzione ABox, piu' superfici (lead diversi) ->
# tante frasi DISTINTE. Positiva + negativi difficili per ciascuna superficie.
for p, s, o in onto["obj_assertions"]:
    if p not in OBJ_PREDS:
        continue
    sp = L.assign_split(f"{s}|{p}|{o}")
    for lead in INST_LEADS:
        grp[sp].append(entry_inst(p, p, s, o, "VALID", "positive", lead)); lab["VALID"] += 1; negc["positive"] += 1
        for w, nt in hard_negatives(p):
            grp[sp].append(entry_inst(p, w, s, o, "INVALID", nt, lead)); lab["INVALID"] += 1; negc[nt] += 1
for sp, it in grp.items():
    wj(GEN / f"kh2_verifier_{sp}.jsonl", it)
wj(GEN / "kh2_verifier_all.jsonl", grp["train"] + grp["val"] + grp["test"])
(GEN / "kh2_verifier_stats.json").write_text(json.dumps({
    "source_sentences": len(srows), "verifier_examples": sum(len(v) for v in grp.values()),
    "split_distribution": {k: len(v) for k, v in grp.items()},
    "label_distribution": dict(lab), "negative_type_distribution": dict(negc),
    "candidate_relations": OBJ_PREDS}, indent=2, ensure_ascii=False), encoding="utf-8")

# ======== INSTANCE source text (ABox) + gold triples ========
# Genera una frase naturale per ogni asserzione ABox di una object property con
# dominio+codominio (cosi' e' estraibile dalla pipeline a regola). Gold = quelle triple.
SYMMETRIC = {"hasFriend", "isEnemyOf"}     # dedup simmetrico (non ha verso canonico)
inst_assertions = [(p, s, o) for (p, s, o) in onto["obj_assertions"] if p in OBJ_PREDS]
sent_records = []; seen_sym = set()
for p, s, o in inst_assertions:
    if p in SYMMETRIC:
        key = tuple(sorted([s, o])) + (p,)
        if key in seen_sym:
            continue
        seen_sym.add(key)
    sd, od = L.display_name(onto, s), L.display_name(onto, o)
    txt = f"{sd} {L.relation_phrase(p)} {od}."
    sent_records.append({"text": txt, "subject": s, "predicate": p, "object": o})
RNG.shuffle(sent_records)

inst_docs = []; gold = []; i = 0; doc_i = 0
while i < len(sent_records):
    n = RNG.randint(8, 12); chunk = sent_records[i:i + n]; i += n; doc_i += 1
    did = f"kh2_inst_doc_{doc_i:03d}"
    text = "\n".join(r["text"] for r in chunk)
    inst_docs.append((did, text))
    for j, r in enumerate(chunk, 1):
        gold.append({"document_id": did, "sentence_id": f"{did}_s{j:03d}", "text": r["text"],
                     "triple": {"subject": r["subject"], "predicate": r["predicate"], "object": r["object"]}})
corpus_text = "\n\n".join(text for _, text in inst_docs)
(DATA_IN / "kh2_corpus.txt").write_text(corpus_text, encoding="utf-8")
wj(GEN / "kh2_instance_triples.jsonl", gold)

print("interface surface forms:", len(ed))
print("schema corpus sentences:", len(srows), "| schema verifier:", sum(len(v) for v in grp.values()))
print("object relations (with domain/range):", len(OBJ_PREDS))
print("instance assertions:", len(inst_assertions), "| instance sentences:", len(sent_records),
      "| instance docs:", len(inst_docs), "| gold triples:", len(gold))
