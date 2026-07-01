# -*- coding: utf-8 -*-
"""STAGE 8 (evaluation, LOCAL). Two modes:
  triples   (default) -> precision/recall/F1 of extracted_triples.jsonl vs the gold
                         ac6_instance_triples.jsonl, on (subject,predicate,object).
                         Also writes generated/triple_metrics.json and two PNG charts.
  labels    -> classification_report between a gold jsonl ('label') and a predictions
               jsonl ('pred'); useful to score verifier/NER predictions.

  python script/08_compute_metrics.py
  python script/08_compute_metrics.py --mode labels --gold g.jsonl --pred p.jsonl

This mode needs NO GPU and NO transformers: it gives you the precision chart directly.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def save_charts(P, R, F, per_rel):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts (pip install matplotlib)."); return
    # global precision / recall / F1
    plt.figure(figsize=(5, 4), dpi=130)
    bars = ["precision", "recall", "f1"]; vals = [P, R, F]
    cols = ["#1e3a8a", "#0f766e", "#9333ea"]
    plt.bar(bars, vals, color=cols)
    for i, v in enumerate(vals):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylim(0, 1.05); plt.title("End-to-end triple extraction"); plt.ylabel("score")
    plt.tight_layout(); plt.savefig(GEN / "triple_metrics.png"); plt.close()
    # per-relation recall
    rels = sorted(per_rel)
    plt.figure(figsize=(max(6, len(rels) * 0.7), 4), dpi=130)
    rec = [per_rel[r][0] / per_rel[r][1] if per_rel[r][1] else 0.0 for r in rels]
    plt.bar(range(len(rels)), rec, color="#1e3a8a")
    plt.xticks(range(len(rels)), rels, rotation=45, ha="right")
    plt.ylim(0, 1.05); plt.title("Per-relation recall"); plt.ylabel("recall")
    plt.tight_layout(); plt.savefig(GEN / "triple_recall_by_relation.png"); plt.close()
    print("saved charts -> generated/triple_metrics.png, generated/triple_recall_by_relation.png")


def triples_mode(gold_path, pred_path):
    gold = set()
    for r in load(gold_path):
        t = r["triple"]; gold.add((t["subject"], t["predicate"], t["object"]))
    pred = set((r["subject_short"], r["predicate"], r["object_short"]) for r in load(pred_path))
    tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    print(f"gold triples={len(gold)}  predicted={len(pred)}  TP={tp} FP={fp} FN={fn}")
    print(f"precision={P:.4f}  recall={R:.4f}  f1={F:.4f}")
    by = Counter(p for _, p, _ in gold)
    hit = Counter(p for tr in (gold & pred) for p in [tr[1]])
    per_rel = {rel: (hit[rel], by[rel]) for rel in by}
    print("per-relation recall:")
    for rel in sorted(by):
        print(f"  {rel}: {hit[rel]}/{by[rel]}")
    out = {"gold": len(gold), "predicted": len(pred), "tp": tp, "fp": fp, "fn": fn,
           "precision": P, "recall": R, "f1": F,
           "per_relation_recall": {rel: {"hit": hit[rel], "total": by[rel]} for rel in sorted(by)}}
    (GEN / "triple_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    save_charts(P, R, F, per_rel)
    print("metrics saved to:", GEN / "triple_metrics.json")


def labels_mode(gold_path, pred_path):
    from sklearn.metrics import classification_report
    gold = [r["label"] for r in load(gold_path)]
    pred = [r.get("pred", r.get("label")) for r in load(pred_path)]
    print(classification_report(gold, pred, digits=4))


def bytier_mode(pred_path):
    """Scorporo F1 per livello di complessita' (Slide 5). Il file deve avere per
    riga: complexity_tier, label (gold) e pred. Rende leggibile dove i modelli
    cedono (es. recall che crolla su 'implicit'/'nested', come da Slide 7)."""
    rows = load(pred_path)
    tiers = {}
    for r in rows:
        t = r.get("complexity_tier", r.get("tier", "explicit"))
        gv = r["label"] == "VALID"
        pv = r.get("pred", r["label"]) == "VALID"
        d = tiers.setdefault(t, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        d["tp"] += gv and pv; d["fn"] += gv and not pv
        d["fp"] += (not gv) and pv; d["tn"] += (not gv) and not pv
    print(f"{'tier':<14} {'n':>5} {'precision':>10} {'recall':>8} {'f1':>8}")
    for t in sorted(tiers):
        d = tiers[t]; n = sum(d.values())
        P = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        R = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        print(f"{t:<14} {n:>5} {P:>10.4f} {R:>8.4f} {F:>8.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["triples", "labels", "bytier"], default="triples")
    ap.add_argument("--gold", default=str(GEN / "ac6_instance_triples.jsonl"))
    ap.add_argument("--pred", default=str(GEN / "extracted_triples.jsonl"))
    args = ap.parse_args()
    if args.mode == "triples":
        triples_mode(args.gold, args.pred)
    elif args.mode == "labels":
        labels_mode(args.gold, args.pred)
    else:
        bytier_mode(args.pred)


if __name__ == "__main__":
    main()
