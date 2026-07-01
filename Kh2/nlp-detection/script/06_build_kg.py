# -*- coding: utf-8 -*-
"""STAGE 6 (KG building, LOCAL). Turns extracted triples into an RDF knowledge graph
(Turtle), validating each triple against the ontology domain/range and de-duplicating.
Adds rdf:type and rdfs:label for every individual involved.
  python script/06_build_kg.py
Writes generated/knowledge_graph.ttl
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
import kh2_lib as L

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
EX = Namespace(L.NS)
onto = L.parse_ontology(BASE / "ontology" / "kingdom_hearts2.owx")


def closure(short):
    out = set(onto["ind_classes"].get(short, []))
    for c in list(out):
        out |= onto["ancestors"](c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--triples", default=str(GEN / "extracted_triples.jsonl"))
    ap.add_argument("--out", default=str(GEN / "knowledge_graph.ttl"))
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.triples, encoding="utf-8") if l.strip()]
    g = Graph(); g.bind("ex", EX); g.bind("rdfs", RDFS)
    used = set(); added = 0; skipped = 0
    for r in rows:
        p = r["predicate"]; s = r["subject_short"]; o = r["object_short"]
        dom = onto["odomain"].get(p); rng = onto["orange"].get(p)
        if dom and dom not in closure(s):
            skipped += 1; continue
        if rng and rng not in closure(o):
            skipped += 1; continue
        g.add((EX[s], EX[p], EX[o])); added += 1
        used.update([s, o])
    type_classes = set()
    for ind in sorted(used):
        # materialise the full rdf:type closure (direct classes + their ancestors) so
        # that queries like "?x a ac:Character" also match Human/AI/Boss individuals
        # without needing an external reasoner.
        for c in closure(ind):
            g.add((EX[ind], RDF.type, EX[c])); type_classes.add(c)
        lab = onto["labels"].get(ind)
        if lab:
            g.add((EX[ind], RDFS.label, Literal(lab)))
    # also keep the relevant rdfs:subClassOf axioms in the graph (documentation / reasoners)
    for c in sorted(type_classes):
        for sup in onto["subclass"].get(c, ()):
            g.add((EX[c], RDFS.subClassOf, EX[sup]))
    # enrich the entities with their asserted data-property values (e.g. element, hitPoints)
    # so attributes are queryable in SPARQL alongside the extracted relations.
    data_added = 0
    for prop, ind, val in onto["data_assertions"]:
        if ind not in used:
            continue
        v = val.strip()
        if v.lower() in ("true", "false"):
            lit = Literal(v.lower() == "true", datatype=XSD.boolean)
        elif re.fullmatch(r"-?\d+", v):
            lit = Literal(int(v), datatype=XSD.integer)
        else:
            lit = Literal(v)
        g.add((EX[ind], EX[prop], lit)); data_added += 1
    g.serialize(destination=args.out, format="turtle")
    print(f"KG written -> {args.out}  (triples added={added}, skipped={skipped}, individuals={len(used)}, total RDF triples={len(g)})")


if __name__ == "__main__":
    main()
