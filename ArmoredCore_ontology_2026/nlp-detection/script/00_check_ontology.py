# -*- coding: utf-8 -*-
"""STAGE 0 (consistency check, LOCAL). Verifies that the ontology is coherent:
counts of classes / object properties / data properties / individuals, presence of
domain & range on every property, consistency of inverse-property pairs (swapped
domain/range), conformity of every ABox object assertion to its property's
domain/range, and presence of an rdfs:comment description on classes and properties.

  python script\\00_check_ontology.py
Exit code is non-zero if any hard inconsistency (missing domain/range, or an ABox
assertion violating domain/range) is found.
"""
from __future__ import annotations
import sys
from pathlib import Path
import ac6_lib as L

BASE = Path(__file__).resolve().parents[1]
onto = L.parse_ontology(BASE / "ontology" / "armoredcore.owx")

INVERSES = [("hasAlias", "isAliasOf"), ("hasRank", "isRankOf"), ("participatesIn", "involvesEntity"),
            ("isBossOf", "hasBoss"), ("takesPlaceIn", "hostsEvent"), ("pilots", "isPilotedBy"),
            ("kills", "isKilledBy"), ("worksFor", "employs")]


def closure(short):
    out = set(onto["ind_classes"].get(short, []))
    for c in list(out):
        out |= onto["ancestors"](c)
    return out


def main():
    hard = []   # blocking inconsistencies
    soft = []   # warnings (e.g. missing description)

    print("=" * 64)
    print("ONTOLOGY COHERENCE REPORT — armoredcore.owx")
    print("=" * 64)
    print(f"classes (classi).................. {len(onto['classes'])}")
    print(f"object properties (relazioni)..... {len(onto['objprops'])}")
    print(f"data properties (attributi)....... {len(onto['dataprops'])}")
    print(f"individuals (individui)........... {len(onto['individuals'])}")
    print()

    # 1) every object property has domain and range
    print("[1] object property domain/range")
    for p in sorted(onto["objprops"]):
        d, r = onto["odomain"].get(p), onto["orange"].get(p)
        if not d or not r:
            hard.append(f"object property {p} missing domain/range (domain={d}, range={r})")
        else:
            print(f"    OK  {p}: {d} -> {r}")
    print()

    # 2) every data property has domain and a (boolean/text) range
    print("[2] data property domain/range")
    for p in sorted(onto["dataprops"]):
        d = onto["ddomain"].get(p)
        if not d:
            hard.append(f"data property {p} missing domain")
        else:
            print(f"    OK  {p}: domain {d}")
    print()

    # 3) inverse pairs must have swapped domain/range
    print("[3] inverse-property consistency")
    for a, b in INVERSES:
        if a in onto["objprops"] and b in onto["objprops"]:
            if onto["odomain"].get(a) == onto["orange"].get(b) and onto["orange"].get(a) == onto["odomain"].get(b):
                print(f"    OK  {a} <-> {b}")
            else:
                hard.append(f"inverse pair {a}/{b} has inconsistent domain/range")
    print()

    # 4) every ABox object assertion respects domain/range
    print("[4] ABox assertions vs domain/range")
    bad = 0
    for p, s, o in onto["obj_assertions"]:
        d, r = onto["odomain"].get(p), onto["orange"].get(p)
        if d and d not in closure(s):
            bad += 1; hard.append(f"{s} {p} {o}: subject not a {d}")
        if r and r not in closure(o):
            bad += 1; hard.append(f"{s} {p} {o}: object not a {r}")
    print(f"    checked {len(onto['obj_assertions'])} object assertions, violations: {bad}")
    print()

    # 5) descriptions (rdfs:comment) on classes and properties
    print("[5] descriptions (rdfs:comment)")
    described = set(onto["labels"])  # labels exist; comments tracked separately below
    import re, html
    t = (BASE / "ontology" / "armoredcore.owx").read_text(encoding="utf-8")
    commented = set(re.findall(r'<AnnotationProperty abbreviatedIRI="rdfs:comment"/><IRI>#([^<]+)</IRI>', t))
    for kind, items in (("class", onto["classes"]), ("object property", onto["objprops"]), ("data property", onto["dataprops"])):
        missing = [x for x in items if x not in commented]
        print(f"    {kind}: {len(items) - len(missing)}/{len(items)} have a description"
              + (f"  (missing: {', '.join(sorted(missing))})" if missing else ""))
        for m in missing:
            soft.append(f"{kind} {m} has no rdfs:comment")
    print()

    # summary
    print("=" * 64)
    if hard:
        print(f"HARD INCONSISTENCIES: {len(hard)}")
        for h in hard:
            print("  X", h)
    else:
        print("HARD INCONSISTENCIES: none — domains, ranges, inverses and ABox are coherent.")
    if soft:
        print(f"WARNINGS (descriptions): {len(soft)}")
        for s in soft:
            print("  !", s)
    print("=" * 64)
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
