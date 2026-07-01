# -*- coding: utf-8 -*-
"""Tabelle aggiuntive per il report finale (richieste dalla consegna):
  - per_story_table     : performance per singola frase/story (TP/FP/FN/P/R/F1)
  - dataset_split_table : conteggi POS/NEG, per-relazione e per neg_type, per split
  - ontology_structure  : tabelle strutturali dell'ontologia (classi/sottoclassi/proprieta')
Solo standard library + kh2_lib. Importato da 11_export_results_data.py.
"""
from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load(p):
    p = Path(p)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _short(uri):
    return str(uri).split("#")[-1].split("/")[-1]


# --------------------------------------------------------------------------- #
# 1) Performance per STORY (estrazione triple) — una riga per frase
# --------------------------------------------------------------------------- #
def per_story_table(gold_path, pred_path, system, out_csv, out_md=None):
    gold_by = defaultdict(set); text_of = {}
    for r in _load(gold_path):
        t = r["triple"]; key = r.get("sentence_id") or r["text"]
        text_of[key] = r["text"]
        gold_by[key].add((_short(t["subject"]), t["predicate"], _short(t["object"])))
    # i predetti non condividono lo stesso sentence_id del gold: si allineano per TESTO
    gold_by_text = defaultdict(set)
    for key, trips in gold_by.items():
        gold_by_text[text_of[key]] |= trips
    pred_by_text = defaultdict(set)
    for r in _load(pred_path):
        sent = r.get("sentence") or r.get("text") or ""
        pred_by_text[sent].add((r.get("subject_short", _short(r.get("subject", ""))),
                                r["predicate"],
                                r.get("object_short", _short(r.get("object", "")))))

    stories = sorted(set(gold_by_text) | set(pred_by_text))
    rows = []
    TP = FP = FN = 0
    f1s = []
    perfect = 0
    for i, sent in enumerate(stories, 1):
        g, p = gold_by_text.get(sent, set()), pred_by_text.get(sent, set())
        tp, fp, fn = len(g & p), len(p - g), len(g - p)
        TP += tp; FP += fp; FN += fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        f1s.append(f1); perfect += (fp == 0 and fn == 0)
        rows.append({"story": f"story_{i:03d}", "sentence": sent[:80],
                     "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4)})
    with open(out_csv, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["story", "sentence", "TP", "FP", "FN",
                                          "precision", "recall", "f1"])
        w.writeheader(); w.writerows(rows)
    micro_p = TP / (TP + FP) if TP + FP else 0.0
    micro_r = TP / (TP + FN) if TP + FN else 0.0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if micro_p + micro_r else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    summary = {"system": system, "stories": len(stories), "perfect_stories": perfect,
               "micro_precision": micro_p, "micro_recall": micro_r, "micro_f1": micro_f1,
               "macro_f1": macro_f1, "TP": TP, "FP": FP, "FN": FN}
    if out_md:
        head = "| Story | TP | FP | FN | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|\n"
        body = "\n".join(f"| {r['story']} | {r['TP']} | {r['FP']} | {r['FN']} | "
                         f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |" for r in rows[:60])
        Path(out_md).write_text(
            f"# Performance per story — {system}\n\n"
            f"Micro P/R/F1: {micro_p:.3f} / {micro_r:.3f} / {micro_f1:.3f}  ·  "
            f"Macro F1: {macro_f1:.3f}  ·  story perfette: {perfect}/{len(stories)}\n\n"
            f"{head}{body}\n\n(prime 60 story; tutte in {Path(out_csv).name})\n", encoding="utf-8")
    return summary


# --------------------------------------------------------------------------- #
# 2) Tabella DATASET SPLIT — POS/NEG + per-relazione + per neg_type, per split
# --------------------------------------------------------------------------- #
def dataset_split_table(gen_dir, out_csv, out_md=None):
    gen_dir = Path(gen_dir)
    rels = set()
    data = {}
    for split in ["train", "val", "test"]:
        rows = _load(gen_dir / f"kh2_verifier_{split}.jsonl")
        pos = sum(1 for r in rows if r["label"] == "VALID")
        neg = sum(1 for r in rows if r["label"] == "INVALID")
        per_rel = Counter(_short(r["candidate_relation"]) for r in rows if r["label"] == "VALID")
        per_neg = Counter(r.get("neg_type", "n/a") for r in rows if r["label"] == "INVALID")
        rels |= set(per_rel)
        data[split] = {"POS": pos, "NEG": neg, "per_rel": per_rel, "per_neg": per_neg, "tot": len(rows)}
    rels = sorted(rels)
    neg_types = ["type_identical", "partial_overlap", "type_incompatible"]
    with open(out_csv, "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["split", "total", "POS", "NEG"] + [f"NEG:{n}" for n in neg_types] + [f"POS:{r}" for r in rels])
        for split, d in data.items():
            w.writerow([split, d["tot"], d["POS"], d["NEG"]]
                       + [d["per_neg"].get(n, 0) for n in neg_types]
                       + [d["per_rel"].get(r, 0) for r in rels])
    if out_md:
        head = "| Split | Tot | POS | NEG | " + " | ".join(neg_types) + " |\n"
        head += "|---|---|---|---|" + "|".join("---" for _ in neg_types) + "|\n"
        body = "\n".join(
            f"| {s} | {d['tot']} | {d['POS']} | {d['NEG']} | "
            + " | ".join(str(d["per_neg"].get(n, 0)) for n in neg_types) + " |"
            for s, d in data.items())
        Path(out_md).write_text(
            "# Dataset split del verifier (per architettura)\n\n"
            "Conteggi per split, con i POSITIVI e i NEGATIVI scomposti per tipo di negativo "
            "(`type_identical` = stessi dominio/codominio del vero, non rifiutabili dalla regola).\n\n"
            f"{head}{body}\n\nConteggio per-relazione completo in " + Path(out_csv).name + "\n",
            encoding="utf-8")
    return data


def extraction_summaries(gen_dir, out_csv=None):
    """Per-story TP/FP/FN/P/R/F1 per ogni sistema di ESTRAZIONE disponibile
    (regola, SpaCy e ogni transformer che ha prodotto extracted_<slug>.jsonl).
    Scrive per_story_<sys>.csv/.md e model_summary_extraction.csv. Ritorna i riassunti."""
    gen_dir = Path(gen_dir)
    gold = gen_dir / "kh2_instance_triples.jsonl"
    systems = []
    if (gen_dir / "extracted_triples.jsonl").exists():
        systems.append(("rule", gen_dir / "extracted_triples.jsonl"))
    if (gen_dir / "spacy_baseline_triples.jsonl").exists():
        systems.append(("spacy", gen_dir / "spacy_baseline_triples.jsonl"))
    for f in sorted(gen_dir.glob("extracted_*.jsonl")):
        slug = f.stem.replace("extracted_", "")
        if slug != "triples":
            systems.append((slug, f))
    out = []
    for name, pred in systems:
        out.append(per_story_table(gold, pred, name, gen_dir / f"per_story_{name}.csv",
                                   gen_dir / f"per_story_{name}.md"))
    if out:
        path = out_csv or (gen_dir / "model_summary_extraction.csv")
        with open(path, "w", newline="", encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(["system", "stories", "perfect_stories", "micro_precision",
                        "micro_recall", "micro_f1", "macro_f1"])
            for s in out:
                w.writerow([s["system"], s["stories"], s["perfect_stories"],
                            f"{s['micro_precision']:.4f}", f"{s['micro_recall']:.4f}",
                            f"{s['micro_f1']:.4f}", f"{s['macro_f1']:.4f}"])
    return out


# --------------------------------------------------------------------------- #
# 3) Tabelle STRUTTURALI dell'ontologia
# --------------------------------------------------------------------------- #
def save_cm_png(matrix, labels, title, out_png):
    """Disegna una matrice di confusione 2x2 (o NxN) come PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    import numpy as _np
    cm = _np.array(matrix)
    plt.figure(figsize=(5, 4), dpi=130); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
    mx = cm.max() if cm.size else 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > mx / 2 else "black")
    plt.title(title); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(out_png); plt.close()
    return True


def verifier_summary_table(systems, out_csv, out_md=None):
    """Tabella riassuntiva del verifier: una riga per (modello, split) con
    Accuracy/Precision/Recall/F1. Usa il campo per_split di ogni sistema."""
    rows = []
    for s in systems:
        ps = s.get("per_split") or {}
        for split in ["train", "val", "test"]:
            d = ps.get(split)
            if not d:
                continue
            rows.append({"model": s.get("slug", s.get("system", "?")), "split": split,
                         "accuracy": d.get("accuracy", 0.0), "precision": d.get("precision", 0.0),
                         "recall": d.get("recall", 0.0), "f1": d.get("f1", 0.0)})
    if not rows:
        return rows
    with open(out_csv, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=["model", "split", "accuracy", "precision", "recall", "f1"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, **{k: round(r[k], 4) for k in ("accuracy", "precision", "recall", "f1")}})
    if out_md:
        head = "| Modello | Split | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|---|\n"
        body = "\n".join(f"| {r['model']} | {r['split']} | {r['accuracy']:.3f} | "
                         f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |" for r in rows)
        Path(out_md).write_text("# Riassunto verifier per split\n\n" + head + body + "\n", encoding="utf-8")
    return rows


def ontology_structure_md(onto, out_md):
    children = defaultdict(list)
    for child, parents in onto.get("subclass", {}).items():
        for par in parents:
            children[par].append(child)
    roots = [c for c in onto["classes"] if c not in onto.get("subclass", {})]

    lines = ["# Struttura dell'ontologia\n",
             f"Classi: {len(onto['classes'])} · object property: {len(onto['objprops'])} · "
             f"data property: {len(onto['dataprops'])} · individui: {len(onto['individuals'])}\n",
             "## Classi e sottoclassi\n",
             "| Classe | Sottoclassi |", "|---|---|"]
    for c in sorted(onto["classes"]):
        subs = ", ".join(sorted(children.get(c, []))) or "—"
        lines.append(f"| {c} | {subs} |")
    lines += ["", "## Object property (dominio → codominio)\n",
              "| Object property | Dominio | Codominio |", "|---|---|---|"]
    for p in sorted(onto["objprops"]):
        lines.append(f"| {p} | {onto['odomain'].get(p, '—')} | {onto['orange'].get(p, '—')} |")
    lines += ["", "## Data property (dominio)\n", "| Data property | Dominio |", "|---|---|"]
    for p in sorted(onto["dataprops"]):
        lines.append(f"| {p} | {onto['ddomain'].get(p, '—')} |")
    if roots:
        lines += ["", "Classi radice (senza superclasse dichiarata): " + ", ".join(sorted(roots))]
    Path(out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
