# -*- coding: utf-8 -*-
"""STAGE 13 (figura). Grafo KG Predicted vs Actual (Slide 11): confronta le triple
predette con quelle gold colorando gli archi:
  verde = True Positive · rosso = False Positive · arancione = False Negative.

Legge un file di triple predette (default: estrazione a regola) e il gold; per
leggibilita' disegna un sottoinsieme di story. Degrada: se manca networkx stampa
un avviso e non blocca la pipeline.

  python script/13_kg_figure.py
  python script/13_kg_figure.py --pred generated/spacy_baseline_triples.jsonl --out generated/kg_pred_vs_actual_spacy.png
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"


def _load(p):
    p = Path(p)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def _short(u):
    return str(u).split("#")[-1].split("/")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default=str(GEN / "kh2_instance_triples.jsonl"))
    ap.add_argument("--pred", default=str(GEN / "extracted_triples.jsonl"))
    ap.add_argument("--out", default=str(GEN / "kg_predicted_vs_actual.png"))
    ap.add_argument("--max-edges", type=int, default=45)
    args = ap.parse_args()

    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        print(f"[skip] figura KG: manca {e.name} (pip install networkx matplotlib)."); return

    gold = {(_short(r["triple"]["subject"]), r["triple"]["predicate"], _short(r["triple"]["object"]))
            for r in _load(args.gold)}
    pred = {(r.get("subject_short", _short(r.get("subject", ""))), r["predicate"],
             r.get("object_short", _short(r.get("object", "")))) for r in _load(args.pred)}

    tp = gold & pred; fp = pred - gold; fn = gold - pred
    # campiona per leggibilita': priorita' a errori (FP/FN) poi TP
    edges = ([(s, o, p, "FP") for s, p, o in fp] + [(s, o, p, "FN") for s, p, o in fn]
             + [(s, o, p, "TP") for s, p, o in tp])[:args.max_edges]
    if not edges:
        print("[skip] nessuna tripla da disegnare."); return

    color = {"TP": "#16a34a", "FP": "#dc2626", "FN": "#f59e0b"}
    G = nx.DiGraph()
    for s, o, p, kind in edges:
        G.add_edge(s, o, label=p, color=color[kind])
    pos = nx.spring_layout(G, k=0.9, seed=7)
    plt.figure(figsize=(13, 9), dpi=140)
    nx.draw_networkx_nodes(G, pos, node_size=520, node_color="#e5e7eb", edgecolors="#374151")
    nx.draw_networkx_labels(G, pos, font_size=7)
    ec = [G[u][v]["color"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color=ec, width=2, arrowsize=14, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): G[u][v]["label"] for u, v in G.edges()},
                                 font_size=6, label_pos=0.5)
    legend = [Line2D([0], [0], color=color["TP"], lw=2, label="True Positive"),
              Line2D([0], [0], color=color["FP"], lw=2, label="False Positive"),
              Line2D([0], [0], color=color["FN"], lw=2, label="False Negative")]
    plt.legend(handles=legend, loc="upper right")
    plt.title(f"Knowledge Graph — Predicted vs Actual  (TP={len(tp)} FP={len(fp)} FN={len(fn)})")
    plt.axis("off"); plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out); plt.close()
    print(f"figura KG -> {args.out}  (TP={len(tp)} FP={len(fp)} FN={len(fn)})")


if __name__ == "__main__":
    main()
