# -*- coding: utf-8 -*-
"""STAGE 12 (benchmark multi-modello, LOCALE). Realizza il confronto richiesto
dalla Slide 13 ("Benchmarking Requirements") e dal Deliverable #4: piu' famiglie
di Transformer come VERIFIER, confrontate su F1 e su velocita'/efficienza
(throughput di inferenza), piu' i baseline non-Transformer.

Cosa produce in generated/ (poi copiati in Results Data da 11):
  - verifier_metrics_<slug>.json   per ogni Transformer allenato
  - benchmark_models.json          tabella aggregata (tutti i sistemi)
  - benchmark_comparison.csv       stessa tabella in CSV
  - benchmark_model_comparison.png grafico a barre P/R/F1 per modello (leggibile)
  - benchmark_comparison_table.png tabella-immagine comparativa (incl. speed)

Degrada con grazia: se torch/transformers (o l'accesso ai modelli) non sono
disponibili, salta i Transformer ma emette comunque rule + SpaCy + grafico + nota,
cosi' lo step non rompe mai la pipeline.

Esempi:
  python script/12_benchmark_models.py                      # set CPU di default
  python script/12_benchmark_models.py --models roberta bert distilbert
  python script/12_benchmark_models.py --models all --epochs 3
"""
from __future__ import annotations
import argparse
import csv
import importlib.util
import json
import time
from pathlib import Path

import ac6_lib as L
import _models_config as REG
import _llm_client as LLM
import _train_utils as TU
import _report_extras as RX
import importlib.util as _ilu
import time

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
ONTO = L.parse_ontology(BASE / "ontology" / "armoredcore.owx")


def _load_03():
    spec = importlib.util.spec_from_file_location("verifier03", Path(__file__).with_name("03_train_verifier.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_jsonl(p):
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _short(uri):
    return uri.split("#")[-1].split("/")[-1]


def _cls_ok(sub_short, required):
    return required == sub_short or required in ONTO["ancestors"](sub_short)


# --------------------------------------------------------------------------- #
# Baseline 1: regola dominio/codominio (nessuna GPU, surface-invariant)
# --------------------------------------------------------------------------- #
def _rule_eval(rows):
    tp = fp = fn = tn = 0
    by_tier, by_neg = {}, {}
    t0 = time.perf_counter(); n = 0
    for r in rows:
        rel = _short(r.get("candidate_relation", ""))
        dom, rng = ONTO["odomain"].get(rel), ONTO["orange"].get(rel)
        if not dom or not rng:
            continue
        n += 1
        sub, obj = _short(r["subject"]), _short(r["object"])
        pred_valid = _cls_ok(sub, dom) and _cls_ok(obj, rng)
        gold_valid = (r["label"] == "VALID")
        tier = r.get("complexity_tier", r.get("tier", "explicit"))
        nt = r.get("neg_type", "positive")
        d = by_tier.setdefault(tier, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        if gold_valid and pred_valid: tp += 1; d["tp"] += 1
        elif gold_valid and not pred_valid: fn += 1; d["fn"] += 1
        elif (not gold_valid) and pred_valid: fp += 1; d["fp"] += 1
        else: tn += 1; d["tn"] += 1
        if not gold_valid:
            b = by_neg.setdefault(nt, {"rejected": 0, "total": 0})
            b["total"] += 1; b["rejected"] += (not pred_valid)
    dt = max(1e-6, time.perf_counter() - t0)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    tier_f1 = {}
    for t, d in by_tier.items():
        pr = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        rc = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        tier_f1[t] = {"precision": pr, "recall": rc,
                      "f1": (2 * pr * rc / (pr + rc)) if pr + rc else 0.0, "n": sum(d.values())}
    neg_acc = {nt: {"reject_accuracy": (b["rejected"] / b["total"] if b["total"] else 0.0), "n": b["total"]}
               for nt, b in by_neg.items()}
    return {"accuracy": acc, "precision": P, "recall": R, "f1": F,
            "confusion_matrix": [[tn, fp], [fn, tp]], "throughput": n / dt,
            "f1_by_complexity_tier": tier_f1, "reject_accuracy_by_neg_type": neg_acc}


def rule_baseline():
    per_split = {sp: _rule_eval(_load_jsonl(GEN / f"ac6_verifier_{sp}.jsonl"))
                 for sp in ["train", "val", "test"] if (GEN / f"ac6_verifier_{sp}.jsonl").exists()}
    te = per_split.get("test", _rule_eval(_load_jsonl(GEN / "ac6_verifier_test.jsonl")))
    return {"system": "rule (domain/range)", "slug": "rule", "level": "verifier",
            "arch": "symbolic", "f1": te["f1"], "precision": te["precision"], "recall": te["recall"],
            "throughput_examples_per_sec": te["throughput"],
            "f1_by_complexity_tier": te["f1_by_complexity_tier"],
            "reject_accuracy_by_neg_type": te["reject_accuracy_by_neg_type"],
            "per_split": {k: {kk: vv for kk, vv in v.items() if kk != "throughput"} for k, v in per_split.items()},
            "note": "surface-invariant: cade sui negativi type_identical"}


# --------------------------------------------------------------------------- #
# Baseline 2: SpaCy generator (livello tripla) — lo legge da 07b
# --------------------------------------------------------------------------- #
def spacy_baseline():
    p = GEN / "spacy_baseline_metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf-8"))
    return {"system": "SpaCy baseline (generator)", "slug": "spacy", "level": "triple",
            "arch": "rule/parse", "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "throughput_examples_per_sec": None,
            "note": "estrazione fragile, niente gate semantico (Slide 7)"}


# --------------------------------------------------------------------------- #
# Validatore LLM/SLM (Architettura C, Slide 9-10) — opzionale
# --------------------------------------------------------------------------- #
def llm_validator(sample=60):
    if not LLM.available():
        return None
    rows = _load_jsonl(GEN / "ac6_verifier_test.jsonl")[:sample]
    tp = fp = fn = tn = 0
    t0 = time.perf_counter()
    for r in rows:
        prompt = ("You are a strict ontology relation validator. Given a candidate triple "
                  "expressed with markers, answer with a single word VALID or INVALID.\n" + r["text"])
        out = (LLM.complete(prompt, max_tokens=4, temperature=0.0) or "").strip().upper()
        pred_valid = out.startswith("VALID")
        gold_valid = (r["label"] == "VALID")
        tp += gold_valid and pred_valid; fn += gold_valid and not pred_valid
        fp += (not gold_valid) and pred_valid; tn += (not gold_valid) and not pred_valid
    dt = max(1e-6, time.perf_counter() - t0)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"system": "LLM validator (Arch. C)", "slug": "llm", "level": "verifier",
            "arch": "llm", "f1": F, "precision": P, "recall": R,
            "throughput_examples_per_sec": len(rows) / dt,
            "note": f"valutato su {len(rows)} esempi (costoso)"}


def _load_07b():
    spec = _ilu.spec_from_file_location("spacy07b", Path(__file__).with_name("07b_spacy_baseline.py"))
    mod = _ilu.module_from_spec(spec); spec.loader.exec_module(mod); return mod


# --------------------------------------------------------------------------- #
# GLiNER — estrazione entita' zero-shot (Architettura A/B, Slide 9)
# --------------------------------------------------------------------------- #
def _bio_entities(tokens, tags):
    """Ricostruisce le entita' (testo_lower, label) da una sequenza BIO."""
    ents, cur, lab = [], None, None
    for tok, tg in zip(tokens, tags):
        if tg.startswith("B-"):
            if cur:
                ents.append((" ".join(cur).lower(), lab))
            cur, lab = [tok], tg[2:]
        elif tg.startswith("I-") and cur is not None:
            cur.append(tok)
        else:
            if cur:
                ents.append((" ".join(cur).lower(), lab))
            cur, lab = None, None
    if cur:
        ents.append((" ".join(cur).lower(), lab))
    return ents


def gliner_eval(hf_id="urchade/gliner_medium-v2.1", threshold=0.5):
    """Estrazione entita' zero-shot (livello entita': span+label) sul test NER.
    Degrada con grazia se il pacchetto 'gliner' o il modello non sono disponibili.
    Ritorna (row|None, nota|None)."""
    p = GEN / "ac6_ner_test.jsonl"
    if not p.exists():
        return None, "GLiNER saltato: manca generated/ac6_ner_test.jsonl."
    try:
        from gliner import GLiNER
    except Exception as e:
        return None, f"GLiNER saltato: pacchetto 'gliner' non installato ({e})."
    rows = _load_jsonl(p)
    labels = sorted({t[2:] for r in rows for t in r["ner_tags"] if t != "O"})
    if not labels:
        return None, "GLiNER saltato: nessuna label entita' nel test NER."
    try:
        model = GLiNER.from_pretrained(hf_id)
    except Exception as e:
        return None, f"GLiNER saltato: caricamento modello fallito ({e})."
    tp = fp = fn = 0
    t0 = time.perf_counter(); n = 0
    for r in rows:
        text = " ".join(r["tokens"])
        gold = set(_bio_entities(r["tokens"], r["ner_tags"]))
        try:
            preds = model.predict_entities(text, labels, threshold=threshold)
        except Exception:
            preds = []
        pset = {(d.get("text", "").lower(), d.get("label", "")) for d in preds}
        tp += len(gold & pset); fp += len(pset - gold); fn += len(gold - pset); n += 1
    dt = max(1e-6, time.perf_counter() - t0)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return ({"system": "GLiNER (zero-shot NER)", "slug": "gliner", "level": "entity",
             "arch": "gliner", "f1": F, "precision": P, "recall": R,
             "throughput_examples_per_sec": n / dt,
             "note": "estrazione entita' zero-shot (Arch. A/B), nessun training"}, None)


# --------------------------------------------------------------------------- #
# GLiREL — candidate generation relazioni zero-shot (Architettura A/B, Slide 9)
# --------------------------------------------------------------------------- #
def glirel_eval(hf_id="jackboyla/glirel-large-v0", threshold=0.5, top_k=3):
    """Genera triple candidate zero-shot e le valuta a livello tripla vs gold.
    Le entita' vengono individuate col dizionario dell'ontologia; le label di
    relazione sono le object property in linguaggio naturale. Best-effort sul
    formato di output di GLiREL; degrada con grazia. Ritorna (row|None, nota)."""
    corpus = BASE / "data-input" / "ac6_corpus.txt"
    gold_p = GEN / "ac6_instance_triples.jsonl"
    iface = GEN / "ontology_interface.json"
    if not (corpus.exists() and gold_p.exists() and iface.exists()):
        return None, "GLiREL saltato: mancano corpus, gold triples o ontology_interface.json."
    try:
        from glirel import GLiREL
    except Exception as e:
        return None, f"GLiREL saltato: pacchetto 'glirel' non installato ({e})."
    ed = json.loads(iface.read_text(encoding="utf-8"))["entity_dictionary"]
    # label-relazione (linguaggio naturale) -> predicate ontologico
    rel_labels, label2pred = [], {}
    for pr in ONTO["objprops"]:
        ph = L.relation_phrase(pr)
        rel_labels.append(ph); label2pred[ph.lower()] = pr
    try:
        model = GLiREL.from_pretrained(hf_id)
    except Exception as e:
        return None, f"GLiREL saltato: caricamento modello fallito ({e})."

    def _short_of(txt):
        for m in L.detect_mentions(txt, ed):
            for e in m.get("entries", []):
                if e.get("kind") == "individual":
                    return e["short"]
            return m.get("short")
        return None

    pred = set()
    t0 = time.perf_counter(); n = 0
    for sent in L.sentence_split(corpus.read_text(encoding="utf-8")):
        tokens = sent.split()
        ner = []
        for m in L.detect_mentions(sent, ed):
            if m["kind"] not in ("individual", "class"):
                continue
            a = len(sent[:m["start"]].split())
            span_txt = sent[m["start"]:m["end"]]
            b = a + max(1, len(span_txt.split())) - 1
            ner.append([a, b, L.ner_type(ONTO, m.get("short", "")), span_txt])
        if len(ner) < 2:
            continue
        n += 1
        try:
            rels = model.predict_relations(tokens, rel_labels, threshold=threshold, ner=ner, top_k=top_k)
        except Exception:
            continue
        for rel in (rels or []):
            lab = (rel.get("label") or rel.get("relation") or "").lower()
            ht = rel.get("head_text") or rel.get("head") or ""
            tt = rel.get("tail_text") or rel.get("tail") or ""
            if isinstance(ht, list):
                ht = " ".join(map(str, ht))
            if isinstance(tt, list):
                tt = " ".join(map(str, tt))
            p_uri = label2pred.get(lab)
            s_short, o_short = _short_of(str(ht)), _short_of(str(tt))
            if p_uri and s_short and o_short:
                pred.add((s_short, p_uri, o_short))
    dt = max(1e-6, time.perf_counter() - t0)

    gold = set()
    for r in _load_jsonl(gold_p):
        t = r["triple"]
        gold.add((t["subject"], t["predicate"], t["object"]))
    tp = len(gold & pred); fp = len(pred - gold); fn = len(gold - pred)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return ({"system": "GLiREL (zero-shot relations)", "slug": "glirel", "level": "triple",
             "arch": "glirel", "f1": F, "precision": P, "recall": R,
             "throughput_examples_per_sec": (n / dt) if n else None,
             "note": f"candidate generation zero-shot (Arch. A/B): {len(pred)} candidati, nessun training"},
            None)


def extract_with_verifier(slug, model_dir, sentences):
    """Estrazione end-to-end con un verifier allenato come gate (Architettura A):
    07b genera i candidati, il modello li filtra VALID/INVALID. Scrive le triple tenute
    e ritorna il path. Richiede torch/transformers."""
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_dir)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    sp = _load_07b()
    kept = []
    for sid, sent in sentences:
        for cand in sp.extract(sent, sid):
            s, p, o = cand["subject_short"], cand["predicate"], cand["object_short"]
            text = (f"[REL] {p} [/REL] A [E1]{L.camel_to_words(s)}[/E1] "
                    f"{L.relation_phrase(p)} a [E2]{L.camel_to_words(o)}[/E2].")
            with torch.no_grad():
                logits = mdl(**tok(text, return_tensors="pt", truncation=True, max_length=192)).logits
            if mdl.config.id2label[int(logits.argmax(-1))] == "VALID":
                kept.append(cand)
    out = GEN / f"extracted_{slug}.jsonl"
    out.write_text("\n".join(json.dumps(t, ensure_ascii=False) for t in kept) + "\n", encoding="utf-8")
    return out


def model_comparison_chart(rows, out_png):
    """Grafico a barre raggruppate: Precision / Recall / F1 per ogni sistema,
    ordinati per F1. Il TEMPO non e' un asse del grafico (sta solo nella tabella).
    Leggibile anche con molti modelli."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib non installato; salto il grafico di confronto."); return
    srows = [r for r in rows if r.get("f1") is not None]
    srows = sorted(srows, key=lambda r: r["f1"], reverse=True)
    if not srows:
        return
    labels = [r["slug"] for r in srows]
    pr = [r["precision"] for r in srows]
    rc = [r["recall"] for r in srows]
    f1 = [r["f1"] for r in srows]
    x = np.arange(len(labels)); w = 0.26
    plt.figure(figsize=(max(8, 1.5 * len(labels)), 6), dpi=130)
    plt.bar(x - w, pr, w, label="Precision", color="#93c5fd")
    plt.bar(x,     rc, w, label="Recall",    color="#fbbf24")
    plt.bar(x + w, f1, w, label="F1",        color="#1e3a8a")
    for i, v in enumerate(f1):
        plt.text(x[i] + w, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 1.10); plt.ylabel("score"); plt.grid(axis="y", alpha=0.3)
    plt.title("Confronto modelli (verifier + baseline) — Precision / Recall / F1")
    plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(out_png); plt.close()
    print(f"grafico confronto -> {out_png.name}")


def model_comparison_table(rows, out_png):
    """Tabella-immagine leggibile con tutti i sistemi: F1/P/R + livello, arch,
    velocita' (qui sì, ma come colonna numerica, non come asse del grafico)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    srows = sorted([r for r in rows if r.get("f1") is not None],
                   key=lambda r: r["f1"], reverse=True)
    if not srows:
        return
    cols = ["Sistema", "Livello", "Arch", "F1", "Precision", "Recall", "Speed (ex/s)"]
    data = []
    for r in srows:
        sp = "—" if r.get("throughput_examples_per_sec") is None else f"{r['throughput_examples_per_sec']:.0f}"
        data.append([r["system"], r.get("level", ""), r.get("arch", ""),
                     f"{r['f1']:.3f}", f"{r['precision']:.3f}", f"{r['recall']:.3f}", sp])
    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.45 * len(data)), dpi=130)
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)
    for j in range(len(cols)):
        c = tbl[0, j]; c.set_facecolor("#1e3a8a"); c.set_text_props(color="white", fontweight="bold")
    # evidenzia il miglior F1
    for j in range(len(cols)):
        tbl[1, j].set_facecolor("#dbeafe")
    ax.set_title("Benchmark modelli — tabella comparativa (ordinata per F1)", pad=12)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight"); plt.close()
    print(f"tabella confronto -> {out_png.name}")


def tier_breakdown_chart(rows, out_png):
    """Quanto e' bravo OGNI modello a riconoscere le frasi di OGNI livello di
    complessita': barre raggruppate di F1 per tier (explicit/implicit/long_distance/
    nested), un gruppo per modello. Risponde a 'statistiche per categoria di frase'."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib non installato; salto il breakdown per tier."); return
    tiers = ["explicit", "implicit", "long_distance", "nested"]
    colors = {"explicit": "#1e3a8a", "implicit": "#3b82f6",
              "long_distance": "#f59e0b", "nested": "#ef4444"}
    srows = [r for r in rows if r.get("f1_by_complexity_tier")]
    if not srows:
        print("nessun dato per-tier; salto il breakdown."); return
    srows = sorted(srows, key=lambda r: r.get("f1", 0), reverse=True)
    labels = [r["slug"] for r in srows]
    x = np.arange(len(labels)); w = 0.2
    plt.figure(figsize=(max(8, 1.7 * len(labels)), 6), dpi=130)
    for i, t in enumerate(tiers):
        vals = [r["f1_by_complexity_tier"].get(t, {}).get("f1", 0.0) for r in srows]
        plt.bar(x + (i - 1.5) * w, vals, w, label=t, color=colors[t])
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylim(0, 1.10); plt.ylabel("F1 (classe VALID)")
    plt.title("Riconoscimento per categoria di frase — F1 per livello di complessita'")
    plt.legend(title="tier", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()
    print(f"breakdown per tier -> {out_png.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=None,
                    help="slug del registry, oppure 'all', oppure hf_id arbitrari")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--retrain", "--reitrain", dest="retrain", action="store_true",
                    help="forza il riaddestramento di TUTTI i modelli; default: riusa i "
                         "training precedenti se presenti")
    args = ap.parse_args()

    if args.models == ["all"]:
        wanted = REG.transformer_slugs()
    else:
        wanted = args.models
    models = [m for m in REG.resolve(wanted) if m["arch"] in ("encoder", "seq2seq")]

    rows = []
    notes = []

    # 1) Transformer verifier (richiede torch/transformers + accesso ai modelli)
    try:
        mod03 = _load_03()
        import torch  # noqa: F401
        transformers_ok = True
    except Exception as e:
        transformers_ok = False
        notes.append(f"Transformer saltati: torch/transformers non disponibili ({e}).")

    if transformers_ok:
        n_models = len(models)
        bench_start = time.perf_counter()
        for i, m in enumerate(models, 1):
            elapsed_all = time.perf_counter() - bench_start
            print("\n" + "=" * 64)
            print(f"[modello {i}/{n_models}] {m['hf_id']}  "
                  f"(trascorso totale benchmark: {TU.format_time(elapsed_all)})")
            print("=" * 64)
            try:
                met = mod03.train_verifier(
                    m["hf_id"], epochs=args.epochs, batch_size=args.batch_size,
                    out_dir=BASE / "models" / f"ac6_verifier_{m['slug']}",
                    metrics_path=GEN / f"verifier_metrics_{m['slug']}.json",
                    seq2seq=(m["arch"] == "seq2seq"), retrain=args.retrain)
                if met.get("reused"):
                    notes.append(f"{m['slug']}: riusato training precedente (--retrain per riaddestrare).")
                rows.append({"system": m["hf_id"], "slug": m["slug"], "level": "verifier",
                             "arch": m["arch"], "f1": met["f1"], "precision": met["precision"],
                             "recall": met["recall"],
                             "throughput_examples_per_sec": met["throughput_examples_per_sec"],
                             "f1_by_complexity_tier": met.get("f1_by_complexity_tier", {}),
                             "note": m["note"]})
                # estrazione end-to-end con questo verifier come gate (per le tabelle per-story).
                # Default: riusa l'estrazione precedente; con --retrain la rigenera.
                try:
                    extracted_path = GEN / f"extracted_{m['slug']}.jsonl"
                    if not args.retrain and extracted_path.exists():
                        print(f"[reuse] {m['slug']}: estrazione per-story gia' presente -> {extracted_path.name}")
                    else:
                        sents = [(f"ac6_corpus_s{i:03d}", s) for i, s in enumerate(
                            L.sentence_split((BASE / "data-input" / "ac6_corpus.txt").read_text(encoding="utf-8")), 1)]
                        extract_with_verifier(m["slug"], BASE / "models" / f"ac6_verifier_{m['slug']}" / "best_model", sents)
                except Exception as e:
                    notes.append(f"{m['slug']}: estrazione per-story saltata -> {e}")
            except Exception as e:
                notes.append(f"{m['slug']} ({m['hf_id']}): training/eval fallito -> {e}")
                print(f"[skip] {m['slug']}: {e}")
        print(f"\nTutti i Transformer completati in {TU.format_time(time.perf_counter() - bench_start)}.")

    # 2) baseline regola (sempre)
    rows.append(rule_baseline())
    # 3) baseline SpaCy (se 07b ha girato)
    sb = spacy_baseline()
    if sb:
        rows.append(sb)
    # 4) validatore LLM (se credenziali presenti)
    lv = llm_validator()
    if lv:
        rows.append(lv)
    # 4b) GLiNER (estrazione entita' zero-shot, Arch. A/B) — se 'gliner' e' installato
    gn_row, gn_note = gliner_eval()
    if gn_row:
        rows.append(gn_row)
    elif gn_note:
        notes.append(gn_note)
    # 4c) GLiREL (candidate generation relazioni zero-shot, Arch. A/B) — se 'glirel' e' installato
    gr_row, gr_note = glirel_eval()
    if gr_row:
        rows.append(gr_row)
    elif gr_note:
        notes.append(gr_note)

    # 5) tabelle per-story dell'ESTRAZIONE (regola + SpaCy + transformer disponibili)
    extraction = RX.extraction_summaries(GEN)

    agg = {"systems": rows, "extraction_per_story": extraction, "notes": notes,
           "axes": {"x": "throughput_examples_per_sec", "y": "f1"},
           "transformer_count": sum(1 for r in rows if r["arch"] in ("encoder", "seq2seq"))}
    (GEN / "benchmark_models.json").write_text(
        json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    with open(GEN / "benchmark_comparison.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["system", "slug", "level", "arch", "f1", "precision", "recall",
                    "throughput_examples_per_sec", "note"])
        for r in rows:
            w.writerow([r["system"], r["slug"], r["level"], r["arch"],
                        f"{r['f1']:.4f}", f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                        ("" if r.get("throughput_examples_per_sec") is None
                         else f"{r['throughput_examples_per_sec']:.1f}"), r.get("note", "")])

    model_comparison_chart(rows, GEN / "benchmark_model_comparison.png")
    model_comparison_table(rows, GEN / "benchmark_comparison_table.png")
    tier_breakdown_chart(rows, GEN / "benchmark_tier_f1.png")
    print("\nbenchmark -> generated/benchmark_models.json, benchmark_comparison.csv")
    for r in rows:
        sp = "" if r.get("throughput_examples_per_sec") is None else f"{r['throughput_examples_per_sec']:.0f} ex/s"
        print(f"  {r['system']:<34} F1={r['f1']:.3f}  {sp}")
    if notes:
        print("\nNote:")
        for n in notes:
            print("  -", n)


if __name__ == "__main__":
    main()
