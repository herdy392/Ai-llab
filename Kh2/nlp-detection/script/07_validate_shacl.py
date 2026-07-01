# -*- coding: utf-8 -*-
"""STAGE 7 (validation, LOCAL). Generates SHACL shapes from the ontology
domain/range and validates the knowledge graph with pySHACL.
  python script/07_validate_shacl.py
Writes generated/shapes.ttl and prints the conformance report.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from rdflib import Graph, Namespace, BNode, Literal
from rdflib.namespace import RDF, RDFS, SH, XSD
import kh2_lib as L

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
EX = Namespace(L.NS)
onto = L.parse_ontology(BASE / "ontology" / "kingdom_hearts2.owx")


def build_shapes():
    g = Graph(); g.bind("sh", SH); g.bind("ex", EX)
    by_domain = {}
    for p in onto["objprops"]:
        d = onto["odomain"].get(p); r = onto["orange"].get(p)
        if d and r:
            by_domain.setdefault(d, []).append((p, r))
    for d, props in by_domain.items():
        shape = EX[d + "Shape"]
        g.add((shape, RDF.type, SH.NodeShape)); g.add((shape, SH.targetClass, EX[d]))
        for p, r in props:
            pn = BNode()
            g.add((shape, SH.property, pn))
            g.add((pn, SH.path, EX[p]))
            g.add((pn, SH["class"], EX[r]))
            g.add((pn, SH.message, Literal(f"{p} must point to a {r}")))
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kg", default=str(GEN / "knowledge_graph.ttl"))
    ap.add_argument("--shapes-out", default=str(GEN / "shapes.ttl"))
    args = ap.parse_args()
    shapes = build_shapes(); shapes.serialize(destination=args.shapes_out, format="turtle")
    print("shapes written ->", args.shapes_out, f"({len(shapes)} triples)")
    try:
        from pyshacl import validate
    except ImportError:
        print("pyshacl not installed; run: pip install pyshacl. Shapes were still generated."); return
    data = Graph(); data.parse(args.kg, format="turtle")
    # type-closure so subclass instances satisfy targetClass (sh:class needs rdfs entailment)
    conforms, _, text = validate(data, shacl_graph=shapes, ont_graph=None, inference="rdfs",
                                 advanced=True, abort_on_first=False)
    print("conforms:", conforms)
    print(text[:4000])


if __name__ == "__main__":
    main()
