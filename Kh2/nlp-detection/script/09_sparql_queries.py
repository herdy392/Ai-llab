# -*- coding: utf-8 -*-
"""STAGE 9 (SPARQL, LOCAL). Runs 3 SPARQL queries over the knowledge graph produced
by 06_build_kg.py (generated/knowledge_graph.ttl), using rdflib. The 3 queries
intentionally showcase FILTER and OPTIONAL:

  Q1  FILTER              - friend pairs, excluding self-loops (FILTER ?a != ?b)
  Q2  OPTIONAL            - every character with keyblade/drive form/world IF present
  Q3  OPTIONAL + FILTER   - fights and their world, with the winning party (optional)

The .rq files are also written to generated/queries/ so you can open them in
Protege / GraphDB / Apache Jena (Fuseki) and run them there too.

  python script/09_sparql_queries.py
  python script/09_sparql_queries.py --query 2          # run only Q2
  python script/09_sparql_queries.py --graph generated/knowledge_graph.ttl
"""
from __future__ import annotations
import argparse
from pathlib import Path
from rdflib import Graph

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
PREFIX = "PREFIX kh: <http://www.semanticweb.org/andrea/ontologies/2026/4/Kingdom_hearts_2#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"

Q1 = PREFIX + """
# Q1 - FILTER: pairs of entities that are friends, dropping self-loops
SELECT ?a ?b WHERE {
  ?a kh:hasFriend ?b .
  FILTER (?a != ?b)
}
ORDER BY ?a ?b
"""

Q2 = PREFIX + """
# Q2 - OPTIONAL: every Character, with the keyblade, drive form and home world only IF asserted
SELECT ?char ?keyblade ?driveform ?world WHERE {
  ?char a kh:Character .
  OPTIONAL { ?char kh:ownsKeyblade ?keyblade . }
  OPTIONAL { ?char kh:hasDriveForm ?driveform . }
  OPTIONAL { ?char kh:comesFrom    ?world . }
}
ORDER BY ?char
"""

Q3 = PREFIX + """
# Q3 - OPTIONAL + FILTER: fights and the world they take place in, with the winning
# party (optional); keep only fights whose name does not contain "Final".
SELECT ?fight ?world ?winner WHERE {
  ?fight kh:takesPlaceIn ?world .
  OPTIONAL { ?fight kh:winner ?winner . }
  FILTER (!regex(str(?fight), "Final"))
}
ORDER BY ?fight ?world
"""

QUERIES = {1: ("Q1 (FILTER) - friendships, no self-loops", Q1),
           2: ("Q2 (OPTIONAL) - characters with optional keyblade, drive form and world", Q2),
           3: ("Q3 (OPTIONAL+FILTER) - fights, their world and optional winner", Q3)}


def short(term):
    s = str(term)
    return s.split("#")[-1] if "#" in s else s


def _edges_for_query(num, rows):
    """Trasforma le righe di una query in archi RDF (subject, predicate, object)
    da visualizzare nel grafo (stile esempio_grapho_RDF)."""
    edges = []
    for r in rows:
        if num == 1:
            edges.append((short(r["a"]), "hasFriend", short(r["b"])))
        elif num == 2:
            c = short(r["char"])
            if r["keyblade"] is not None:
                edges.append((c, "ownsKeyblade", short(r["keyblade"])))
            if r["driveform"] is not None:
                edges.append((c, "hasDriveForm", short(r["driveform"])))
            if r["world"] is not None:
                edges.append((c, "comesFrom", short(r["world"])))
        elif num == 3:
            fgt = short(r["fight"])
            if r["world"] is not None:
                edges.append((fgt, "takesPlaceIn", short(r["world"])))
            if r["winner"] is not None:
                edges.append((fgt, "winner", short(r["winner"])))
    return edges


def draw_rdf_graph(edges, title, out_png, max_edges=60):
    """Disegna le triple come grafo a nodi/archi (nodi azzurri, archi grigi,
    etichetta-relazione in rosso), come nell'esempio fornito. Degrada con grazia
    se networkx/matplotlib non sono installati."""
    if not edges:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        print("networkx/matplotlib non installati: salto il grafo RDF."); return False

    shown = edges[:max_edges]
    truncated = len(edges) > max_edges
    G = nx.DiGraph()
    for s, p, o in shown:
        G.add_edge(s, o, label=p)
    n = G.number_of_nodes()
    plt.figure(figsize=(13, 9), dpi=130)
    pos = nx.spring_layout(G, seed=42, k=(1.8 / (n ** 0.5) if n else 1.0))
    nx.draw_networkx_nodes(G, pos, node_color="#add8e6",
                           node_size=[600 + 120 * G.degree(v) for v in G.nodes()])
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True,
                           arrowstyle="-|>", arrowsize=12, width=1.3,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight="bold")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(s, o): p for s, p, o in shown},
                                 font_color="red", font_size=7, label_pos=0.5)
    ttl = title + (f"  (primi {max_edges} archi di {len(edges)})" if truncated else "")
    plt.title(ttl, fontsize=13)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(out_png); plt.close()
    print(f"grafo RDF -> {out_png.relative_to(out_png.parents[2])}")
    return True


def run(g, num, draw=True):
    title, q = QUERIES[num]
    print("=" * 72); print(title); print("-" * 72)
    rows = list(g.query(q))
    if not rows:
        print("(no results)")
    else:
        cols = [str(v) for v in rows[0].labels]
        print(" | ".join(cols))
        for r in rows:
            print(" | ".join(short(r[c]) if r[c] is not None else "—" for c in cols))
    print(f"({len(rows)} rows)\n")
    if draw and rows:
        draw_rdf_graph(_edges_for_query(num, rows),
                       f"Grafo RDF — {QUERIES[num][0].split(' - ')[0]}",
                       GEN / f"sparql_graph_q{num}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default=str(GEN / "knowledge_graph.ttl"))
    ap.add_argument("--query", type=int, choices=[1, 2, 3], default=None)
    args = ap.parse_args()

    qdir = GEN / "queries"; qdir.mkdir(parents=True, exist_ok=True)
    for n, (_, q) in QUERIES.items():
        (qdir / f"query{n}.rq").write_text(q.strip() + "\n", encoding="utf-8")

    gp = Path(args.graph)
    if not gp.exists():
        raise SystemExit(f"Knowledge graph not found: {gp}\nRun  python script/06_build_kg.py  first.")
    g = Graph(); g.parse(gp, format="turtle")
    print(f"loaded {len(g)} triples from {gp.name}\n")
    for n in ([args.query] if args.query else [1, 2, 3]):
        run(g, n)
    print("query files written to:", qdir)


if __name__ == "__main__":
    main()
