# -*- coding: utf-8 -*-
"""Standalone metrics helper (mirrors the professor's notebook/compute_metrics.py).
Computes precision / recall / F1 of the extracted triples against the gold instance
triples and prints a per-relation breakdown. Can be imported in a notebook or run as:

    python notebook/compute_metrics.py
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

NLP = Path(__file__).resolve().parents[1]
GEN = NLP / "generated"


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def evaluate(gold_path=GEN / "ac6_instance_triples.jsonl",
             pred_path=GEN / "extracted_triples.jsonl"):
    gold = set()
    for r in load(gold_path):
        t = r["triple"]; gold.add((t["subject"], t["predicate"], t["object"]))
    pred = set((r["subject_short"], r["predicate"], r["object_short"]) for r in load(pred_path))
    tp, fp, fn = len(gold & pred), len(pred - gold), len(gold - pred)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    by = Counter(p for _, p, _ in gold); hit = Counter(tr[1] for tr in (gold & pred))
    return {"precision": P, "recall": R, "f1": F, "tp": tp, "fp": fp, "fn": fn,
            "per_relation_recall": {r: (hit[r], by[r]) for r in sorted(by)}}


if __name__ == "__main__":
    m = evaluate()
    print(f"precision={m['precision']:.4f}  recall={m['recall']:.4f}  f1={m['f1']:.4f}  "
          f"(TP={m['tp']} FP={m['fp']} FN={m['fn']})")
    for rel, (h, tot) in m["per_relation_recall"].items():
        print(f"  {rel}: {h}/{tot}")
