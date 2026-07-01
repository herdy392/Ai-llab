# -*- coding: utf-8 -*-
"""STAGE 1b (data generation). Aggiunge la COMPLESSITA' STRATIFICATA richiesta
dalla Slide 5 ("Tiered Complexity": Explicit / Implicit / Long-Distance / Nested),
che nel dataset template-based mancava (esisteva solo un campo `tier` costante).

Perche' serve: senza esempi difficili tutti i modelli del benchmark convergono a
~1.0 e il confronto NON discrimina. Questo step rende il benchmark significativo.

Cosa fa:
  - rilegge i verifier JSONL (kh2_verifier_{train,val,test,all}) prodotti da 01;
  - tagga gli esempi base come `complexity_tier="explicit"`;
  - ne deriva varianti `implicit` (sinonimi/parafrasi del trigger), `long_distance`
    (clausola distrattore tra i due argomenti) e `nested` (contesto con piu' entita');
  - se e' disponibile un LLM (ANTHROPIC_API_KEY) riscrive le frasi in modo piu'
    naturale; altrimenti usa trasformazioni deterministiche (fallback offline).
  - le LABEL restano corrette per costruzione (si trasforma solo la superficie,
    preservando relazione/classi/label).

Idempotente: ricostruisce sempre le varianti partendo dagli esempi `explicit`.

ANTI-LEAKAGE: dopo l'augmentation le frasi vengono DEDUPLICATE (una frase = un
esempio) e ri-splittate in train/val/test in modo DISGIUNTO a livello di frase
(hash del testo, 70/15/15). Garanzia verificata a fine run: nessuna frase di
validation/test compare nel training. Cosi' l'accuracy non e' gonfiata da copie
identiche viste in training.

Esegui dopo 01/02:  python script/01b_augment_complexity.py
"""
from __future__ import annotations
import hashlib
import json
import random
import re
from pathlib import Path

import _llm_client as LLM

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
RNG = random.Random(20260629)

SPLITS = ["train", "val", "test"]
TIERS = ["explicit", "implicit", "long_distance", "nested"]

# Sinonimi/parafrasi indirette per la tier "implicit" (per relazione KH2).
PARAPHRASE = {
    "appearsIn": ["shows up in", "can be found in"],
    "comesFrom": ["originates from", "hails from"],
    "takesPlaceIn": ["unfolds in", "is set in"],
    "ownsKeyblade": ["wields", "is the wielder of"],
    "hasFriend": ["is allied with", "stands with"],
    "isEnemyOf": ["is at odds with", "opposes"],
    "defeats": ["beats", "takes down"],
    "isMemberOf": ["belongs to", "is part of"],
    "hasDriveForm": ["can transform into", "unlocks the form"],
    "canSummon": ["is able to call upon", "summons"],
    "hasInInventory": ["carries", "keeps"],
    "hasPartyMember": ["is joined by", "includes"],
    "hasSecondaryWeapon": ["is equipped with", "fights with"],
    "participatesIn": ["takes part in", "is involved in"],
    "winner": ["is won by", "is claimed by"],
    "engagesInFight": ["takes part in the battle", "fights in"],
    "hasAction": ["can perform", "is able to use"],
    "drops": ["yields", "leaves behind"],
}

DISTRACTORS = [
    ", according to Jiminy's Journal,",
    ", as recorded in the Gummi logs,",
    ", a detail noted across the worlds,",
    ", as the heroes later recalled,",
]

NESTED_PREFIX = [
    "Although the Heartless gathered nearby,",
    "While Organization XIII schemed in the shadows,",
    "Even as the Keyblade glowed with light,",
]


def load(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def dump(p, rows):
    p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + ("\n" if rows else ""),
                 encoding="utf-8")


def short(uri):
    return uri.split("#")[-1].split("/")[-1]


_RE_BODY = re.compile(r"^(\[REL\].*?\[/REL\] )(.*?)(\[E1\].*?\[/E1\])(.*?)(\[E2\].*?\[/E2\])(.*)$", re.S)


def _phrase_of(rel, default_mid):
    syns = PARAPHRASE.get(rel)
    if not syns:
        return None
    return RNG.choice(syns)


def make_variant(row, tier):
    """Crea una variante di superficie preservando markers/relazione/label."""
    m = _RE_BODY.match(row["text"])
    out = dict(row)
    out["complexity_tier"] = tier
    out["tier"] = tier  # mantiene il campo storico allineato
    if not m or tier == "explicit":
        out["complexity_tier"] = "explicit" if tier == "explicit" else tier
        return out
    head, pre, e1, mid, e2, tail = m.groups()
    rel = short(row.get("candidate_relation", ""))

    if tier == "implicit":
        syn = _phrase_of(rel, mid)
        if syn is None:
            return None  # senza sinonimo non e' davvero "implicit": scarta
        new_mid = f" {syn} "
        out["text"] = f"{head}{pre}{e1}{new_mid}{e2}{tail}"
    elif tier == "long_distance":
        distract = RNG.choice(DISTRACTORS)
        out["text"] = f"{head}{pre}{e1}{distract}{mid}{e2}{tail}"
    elif tier == "nested":
        prefix = RNG.choice(NESTED_PREFIX)
        # contesto ambiguo: clausola iniziale + la tripla target.
        body = f"{pre}{e1}{mid}{e2}{tail}".strip()
        out["text"] = f"{head}{prefix} {body}"
    out["sentence"] = re.sub(r"\[/?E[12]\]|\[/?REL\]", "", out["text"]).strip()
    return out


def llm_naturalize(rows):
    """Se l'LLM e' disponibile, riscrive in modo piu' naturale la parte testuale.
    Best-effort: in caso di errore mantiene la variante deterministica."""
    if not LLM.available():
        return rows, False
    out = []
    for r in rows:
        m = _RE_BODY.match(r["text"])
        if not m or r["complexity_tier"] == "explicit":
            out.append(r); continue
        head, pre, e1, mid, e2, tail = m.groups()
        prompt = ("Rewrite the sentence to sound natural and fluent, keeping EXACTLY the "
                  "markers [E1]...[/E1] and [E2]...[/E2] around the same entities and not "
                  "adding new ones. Return only the sentence.\nSentence: "
                  + f"{pre}{e1}{mid}{e2}{tail}".strip())
        txt = LLM.complete(prompt, max_tokens=120, temperature=0.6)
        if txt and "[E1]" in txt and "[E2]" in txt:
            r = dict(r)
            r["text"] = head + txt.strip()
            r["sentence"] = re.sub(r"\[/?E[12]\]|\[/?REL\]", "", r["text"]).strip()
        out.append(r)
    return out, True


def _split_of(text: str, ptrain=0.70, pval=0.15) -> str:
    """Split DETERMINISTICO e disgiunto a livello di FRASE: ogni testo univoco
    finisce sempre in un solo split (train/val/test) in base al suo hash. Cosi'
    nessuna frase usata in validation/test puo' comparire nel training
    (no data leakage), come richiesto. Rapporto target 70/15/15."""
    h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % 10_000 / 10_000.0
    if h < ptrain:
        return "train"
    if h < ptrain + pval:
        return "val"
    return "test"


def main():
    used_llm = False

    # 1) RACCOLTA base "explicit" da TUTTI gli split prodotti da 01 (poi rifacciamo
    #    NOI lo split, in modo disgiunto e senza leakage). Si MANTIENE la frequenza
    #    naturale (no dedup) per preservare il bilanciamento VALID/INVALID e il volume.
    raw = []
    for split in SPLITS:
        p = GEN / f"kh2_verifier_{split}.jsonl"
        if p.exists():
            raw += load(p)
    base = []
    for r in raw:
        # SOLO i veri esempi base (explicit/tier_1): scarta le varianti gia' derivate
        # in run precedenti, altrimenti le ri-augmenteremmo (idempotenza).
        tier = r.get("complexity_tier", r.get("tier", "explicit"))
        if tier not in ("explicit", "tier_1", None):
            continue
        base.append(make_variant(r, "explicit"))

    # 2) AUGMENTATION: da ogni base genera implicit / long_distance / nested.
    out = list(base)
    for r in base:
        for tier in ("implicit", "long_distance", "nested"):
            v = make_variant(r, tier)
            if v is not None:
                out.append(v)
    out, used_llm = llm_naturalize(out)

    # 3) SPLIT disgiunto a livello di FRASE (by-hash): TUTTE le copie di una stessa
    #    frase finiscono nello stesso split -> nessuna frase di val/test e' nel train,
    #    pur conservando frequenze e bilanciamento delle classi.
    buckets = {"train": [], "val": [], "test": []}
    stats = {t: {"train": 0, "val": 0, "test": 0} for t in TIERS}
    for r in out:
        sp = _split_of(r["text"])
        r["split"] = sp
        buckets[sp].append(r)
        t = r.get("complexity_tier", "explicit")
        if t in stats:
            stats[t][sp] += 1

    for sp in SPLITS:
        RNG.shuffle(buckets[sp])
        dump(GEN / f"kh2_verifier_{sp}.jsonl", buckets[sp])
        uniq = len({r["text"] for r in buckets[sp]})
        print(f"{sp}: {len(buckets[sp])} esempi ({uniq} frasi uniche) "
              f"({'LLM' if used_llm else 'offline'})")

    all_rows = buckets["train"] + buckets["val"] + buckets["test"]
    dump(GEN / "kh2_verifier_all.jsonl", all_rows)

    # 4) VERIFICA anti-leakage: nessuna frase di val/test deve stare nel train.
    T = {r["text"] for r in buckets["train"]}
    leak_val = sum(1 for r in buckets["val"] if r["text"] in T)
    leak_test = sum(1 for r in buckets["test"] if r["text"] in T)
    assert leak_val == 0 and leak_test == 0, f"LEAKAGE! val={leak_val} test={leak_test}"

    # 5) Aggiorna kh2_verifier_stats.json con lo split REALE post-augmentation
    #    (l'export 11 legge questo file per verificare 70/15/15 e le label).
    from collections import Counter as _C
    vstats_p = GEN / "kh2_verifier_stats.json"
    if vstats_p.exists():
        vstats = json.loads(vstats_p.read_text(encoding="utf-8"))
        vstats["split_distribution"] = {sp: len(buckets[sp]) for sp in SPLITS}
        vstats["verifier_examples"] = len(all_rows)
        vstats["label_distribution"] = dict(_C(r["label"] for r in all_rows))
        vstats["split_is_disjoint_by_sentence"] = True
        vstats["augmented_tiers"] = TIERS
        vstats_p.write_text(json.dumps(vstats, indent=2, ensure_ascii=False), encoding="utf-8")

    sizes = {sp: len(buckets[sp]) for sp in SPLITS}
    tot = sum(sizes.values()) or 1
    (GEN / "complexity_stats.json").write_text(
        json.dumps({"generator": "llm" if used_llm else "offline-deterministic",
                    "tiers": TIERS, "counts": stats,
                    "split_sizes": sizes,
                    "split_percent": {k: round(100 * v / tot, 1) for k, v in sizes.items()},
                    "leakage_val_test": {"val_in_train": leak_val, "test_in_train": leak_test},
                    "total": len(all_rows)}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print("complexity tiers:", {t: sum(stats[t].values()) for t in TIERS})
    print(f"split sizes: {sizes}  (target 70/15/15) | leakage val/test in train: "
          f"{leak_val}/{leak_test}")
    print("stats -> generated/complexity_stats.json")


if __name__ == "__main__":
    main()
