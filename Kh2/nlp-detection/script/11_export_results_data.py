# -*- coding: utf-8 -*-
"""STAGE 11 (consolidamento risultati, LOCALE).

Raccoglie in un'unica cartella "Results Data" (alla radice del progetto) TUTTI i
dati richiesti alla fine del progetto ontologia, secondo la Project Description
del professore (Block 14 — "Deliverables & Milestones"):

  1) Custom Domain Ontology  -> ontologia OWL + report di coerenza (domini/codomini,
                                inverse, ABox, descrizioni) + interfaccia ontologia.
  2) Synthetic Annotated Dataset (JSONL, split 70/15/15 Train/Val/Test, complessita'
                                a livelli) + statistiche del dataset.
  3) Working Software Pipeline -> KG RDF (.ttl), shapes SHACL, esito SHACL Conforms,
                                triple estratte, risultato di inferenza.
  4) Benchmarking Report -> metriche Precision/Recall/F1 a livello tripla (globali e
                                per-relazione), matrice di confusione del verifier
                                (regola dominio/codominio, e modello se presente),
                                risultati SPARQL (FILTER/OPTIONAL) e una relazione .md.

NESSUNA GPU richiesta: in modalita' regola produce comunque tutte le metriche.
I path sono SEMPRE relativi (risolti rispetto alla posizione di questo file); un
eventuale override assoluto si imposta in paths_config.py alla radice del progetto.

Uso:
  python script/11_export_results_data.py
"""
from __future__ import annotations
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import kh2_lib as L
import _report_extras as RX

# ---- Path resolution (relativi, robusti) -----------------------------------
BASE = Path(__file__).resolve().parents[1]          # -> nlp-detection/
REPO = Path(__file__).resolve().parents[2]          # -> project root (Kh2)
GEN = BASE / "generated"
ONTO_FILE = BASE / "ontology" / "kingdom_hearts2.owx"

# Override opzionale della cartella di output via paths_config.py (vedi istruzioni).
RESULTS = REPO / "Results Data"
try:
    sys.path.insert(0, str(REPO))
    import paths_config as _cfg          # type: ignore
    if getattr(_cfg, "RESULTS_DIR", ""):
        RESULTS = Path(_cfg.RESULTS_DIR)
except Exception:
    pass

onto = L.parse_ontology(ONTO_FILE)


def _load_jsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def _short(iri: str) -> str:
    return iri.split("#")[-1].split("/")[-1]


def _cls_compatible(sub_short: str, required_class: str) -> bool:
    """required_class deve essere antenato-o-uguale del tipo di sub_short.
    Vale sia per nomi di CLASSE sia per INDIVIDUI (chiude su classi asserite+antenati)."""
    types = {sub_short} | onto["ancestors"](sub_short)
    for c in onto["ind_classes"].get(sub_short, []):
        types.add(c); types |= onto["ancestors"](c)
    return required_class in types


def _safe_div(n, d):
    return n / d if d else 0.0


def _copy(src: Path, dst_dir: Path):
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)
        return True
    return False


# ============================================================================
# 1) ONTOLOGY — report di coerenza + interfaccia
# ============================================================================
INVERSES = list(onto["inverses"])   # coppie inverse lette dall'ontologia (KH2 puo' non averne)


def _closure(short):
    out = set(onto["ind_classes"].get(short, []))
    for c in list(out):
        out |= onto["ancestors"](c)
    return out


def export_ontology(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy(ONTO_FILE, out_dir)
    _copy(GEN / "ontology_interface.json", out_dir)

    import re
    txt = ONTO_FILE.read_text(encoding="utf-8")
    commented = set(re.findall(
        r'<AnnotationProperty abbreviatedIRI="rdfs:comment"/><IRI>#([^<]+)</IRI>', txt))

    # inverse pairs coerenti
    inv_ok = sum(1 for a, b in INVERSES
                 if a in onto["objprops"] and b in onto["objprops"]
                 and onto["odomain"].get(a) == onto["orange"].get(b)
                 and onto["orange"].get(a) == onto["odomain"].get(b))
    inv_total = sum(1 for a, b in INVERSES
                    if a in onto["objprops"] and b in onto["objprops"])

    # ABox vs domain/range
    abox_bad = 0
    for p, s, o in onto["obj_assertions"]:
        d, r = onto["odomain"].get(p), onto["orange"].get(p)
        if d and d not in _closure(s):
            abox_bad += 1
        if r and r not in _closure(o):
            abox_bad += 1

    report = {
        "ontology_file": "nlp-detection/ontology/kingdom_hearts2.owx",
        "counts": {
            "classes": len(onto["classes"]),
            "object_properties": len(onto["objprops"]),
            "data_properties": len(onto["dataprops"]),
            "individuals": len(onto["individuals"]),
        },
        "object_properties_with_domain_and_range":
            f"{sum(1 for p in onto['objprops'] if onto['odomain'].get(p) and onto['orange'].get(p))}"
            f"/{len(onto['objprops'])}",
        "inverse_pairs_consistent": f"{inv_ok}/{inv_total}",
        "abox_object_assertions_checked": len(onto["obj_assertions"]),
        "abox_violations": abox_bad,
        "descriptions_rdfs_comment": {
            "classes": f"{sum(1 for x in onto['classes'] if x in commented)}/{len(onto['classes'])}",
            "object_properties": f"{sum(1 for x in onto['objprops'] if x in commented)}/{len(onto['objprops'])}",
            "data_properties": f"{sum(1 for x in onto['dataprops'] if x in commented)}/{len(onto['dataprops'])}",
        },
        "hard_inconsistencies": abox_bad + sum(
            1 for p in onto["objprops"] if not (onto["odomain"].get(p) and onto["orange"].get(p))),
    }
    (out_dir / "ontology_coherence_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["ONTOLOGY COHERENCE REPORT - kingdom_hearts2.owx", "=" * 56,
             f"classes ............ {report['counts']['classes']}",
             f"object properties .. {report['counts']['object_properties']}",
             f"data properties .... {report['counts']['data_properties']}",
             f"individuals ........ {report['counts']['individuals']}", "",
             f"object props with domain+range ... {report['object_properties_with_domain_and_range']}",
             f"inverse pairs consistent ......... {report['inverse_pairs_consistent']}",
             f"ABox assertions checked .......... {report['abox_object_assertions_checked']}",
             f"ABox violations .................. {report['abox_violations']}",
             f"descriptions (classes) ........... {report['descriptions_rdfs_comment']['classes']}",
             f"descriptions (object props) ...... {report['descriptions_rdfs_comment']['object_properties']}",
             f"descriptions (data props) ........ {report['descriptions_rdfs_comment']['data_properties']}", "",
             ("HARD INCONSISTENCIES: none - coherent." if report["hard_inconsistencies"] == 0
              else f"HARD INCONSISTENCIES: {report['hard_inconsistencies']}")]
    (out_dir / "ontology_coherence_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


# ============================================================================
# 2) DATASET — split JSONL + statistiche (verifica 70/15/15)
# ============================================================================
def export_dataset(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = GEN / "synthetic_phase_corpus"

    # Synthetic annotated dataset (deliverable #2): documenti + annotazioni a livello frase
    for name in ["synthetic_documents_train.jsonl", "synthetic_documents_val.jsonl",
                 "synthetic_documents_test.jsonl", "synthetic_documents_all.jsonl",
                 "synthetic_sentence_annotations_train.jsonl",
                 "synthetic_sentence_annotations_val.jsonl",
                 "synthetic_sentence_annotations_test.jsonl",
                 "synthetic_sentence_annotations_all.jsonl",
                 "synthetic_corpus_stats.json"]:
        _copy(corpus_dir / name, out_dir / "synthetic_annotated_dataset")

    # Verifier dataset (special tokens [REL][E1][E2]) + NER + relation candidates
    for name in ["kh2_verifier_train.jsonl", "kh2_verifier_val.jsonl", "kh2_verifier_test.jsonl",
                 "kh2_verifier_all.jsonl", "kh2_verifier_stats.json",
                 "kh2_relation_candidates_train.jsonl", "kh2_relation_candidates_val.jsonl",
                 "kh2_relation_candidates_test.jsonl", "kh2_relation_candidates_all.jsonl",
                 "kh2_ner_train.jsonl", "kh2_ner_val.jsonl", "kh2_ner_test.jsonl",
                 "kh2_weak_stats.json", "kh2_instance_triples.jsonl"]:
        _copy(GEN / name, out_dir)

    # Statistiche consolidate con verifica esplicita dello split 70/15/15
    corpus_stats = json.loads((corpus_dir / "synthetic_corpus_stats.json").read_text(encoding="utf-8"))
    vstats = json.loads((GEN / "kh2_verifier_stats.json").read_text(encoding="utf-8"))

    def _pct(split: dict):
        tot = sum(split.values()) or 1
        return {k: round(100 * v / tot, 1) for k, v in split.items()}

    summary = {
        "deliverable": "Synthetic Annotated Dataset (JSONL, tiered complexity, Train/Val/Test)",
        "synthetic_corpus": {
            "documents": corpus_stats.get("documents"),
            "sentences": corpus_stats.get("sentences"),
            "document_split": corpus_stats.get("document_split_distribution"),
            "document_split_percent": _pct(corpus_stats.get("document_split_distribution", {})),
            "sentence_split": corpus_stats.get("sentence_split_distribution"),
            "sentence_split_percent": _pct(corpus_stats.get("sentence_split_distribution", {})),
        },
        "verifier_dataset": {
            "examples": vstats.get("verifier_examples"),
            "split": vstats.get("split_distribution"),
            "split_percent": _pct(vstats.get("split_distribution", {})),
            "label_distribution": vstats.get("label_distribution"),
        },
        "target_split": {"train": 70, "val": 15, "test": 15},
        "note": "Lo split documenti del corpus sintetico segue 70/15/15 (slide Phase 2).",
    }
    (out_dir / "dataset_statistics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


# ============================================================================
# 3) KNOWLEDGE GRAPH — TTL, SHACL, triple, inferenza
# ============================================================================
def export_kg(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ["knowledge_graph.ttl", "shapes.ttl", "extracted_triples.jsonl",
                 "last_inference_result.json"]:
        _copy(GEN / name, out_dir)

    # esito SHACL (Conforms)
    try:
        from pyshacl import validate
        from rdflib import Graph
        dg = Graph().parse(GEN / "knowledge_graph.ttl", format="turtle")
        sg = Graph().parse(GEN / "shapes.ttl", format="turtle")
        conforms, _, text = validate(dg, shacl_graph=sg, inference="rdfs")
        (out_dir / "shacl_validation.txt").write_text(
            f"Conforms: {conforms}\n\n{text}", encoding="utf-8")
        return {"conforms": bool(conforms)}
    except Exception as e:
        (out_dir / "shacl_validation.txt").write_text(f"SHACL non eseguito: {e}\n", encoding="utf-8")
        return {"conforms": None}


# ============================================================================
# 4) METRICS + SPARQL + BENCHMARK REPORT
# ============================================================================
def _rule_verifier_confusion(rows):
    """Classificatore = regola dominio/codominio. Applicato alle sole relazioni
    object-property (dove domain/range esistono). Ritorna [[TN,FP],[FN,TP]]."""
    tn = fp = fn = tp = 0
    applicable = skipped = 0
    for r in rows:
        rel = _short(r["candidate_relation"])
        dom, rng = onto["odomain"].get(rel), onto["orange"].get(rel)
        if not dom or not rng:
            skipped += 1
            continue
        applicable += 1
        sub, obj = _short(r["subject"]), _short(r["object"])
        pred_valid = _cls_compatible(sub, dom) and _cls_compatible(obj, rng)
        gold_valid = (r["label"] == "VALID")
        if gold_valid and pred_valid:
            tp += 1
        elif gold_valid and not pred_valid:
            fn += 1
        elif (not gold_valid) and pred_valid:
            fp += 1
        else:
            tn += 1
    return [[tn, fp], [fn, tp]], applicable, skipped


def _binary_metrics(cm):
    tn, fp = cm[0]
    fn, tp = cm[1]
    total = tn + fp + fn + tp
    import math
    prec = _safe_div(tp, tp + fp)
    rec = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    f1 = _safe_div(2 * prec * rec, prec + rec)
    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "total": total},
        "metrics": {
            "accuracy": _safe_div(tp + tn, total),
            "precision": prec,
            "recall_sensitivity": rec,
            "specificity": spec,
            "f1_score": f1,
            "negative_predictive_value": _safe_div(tn, tn + fn),
            "false_positive_rate": _safe_div(fp, fp + tn),
            "false_negative_rate": _safe_div(fn, fn + tp),
            "balanced_accuracy": (rec + spec) / 2,
            "matthews_corrcoef": _safe_div((tp * tn) - (fp * fn), mcc_den),
        },
        "task_type": "binary_classification",
        "expected_layout": "[[TN, FP], [FN, TP]]",
    }


def export_metrics(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # triple-level (gia' prodotte da 08): P/R/F1 + per-relazione + grafici
    for name in ["triple_metrics.json", "triple_metrics.png", "triple_recall_by_relation.png"]:
        _copy(GEN / name, out_dir)
    triple = json.loads((GEN / "triple_metrics.json").read_text(encoding="utf-8"))

    # matrice di confusione del verifier (regola dominio/codominio) sul TEST set
    test_rows = _load_jsonl(GEN / "kh2_verifier_test.jsonl")
    cm, appl, skip = _rule_verifier_confusion(test_rows)
    vm = _binary_metrics(cm)
    vm["evaluated_on"] = "kh2_verifier_test.jsonl"
    vm["classifier"] = "rule: domain/range (no GPU)"
    vm["applicable_object_property_rows"] = appl
    vm["skipped_rows_no_domain_range"] = skip
    (out_dir / "verifier_metrics_rule.json").write_text(
        json.dumps(vm, indent=2, ensure_ascii=False), encoding="utf-8")

    # metriche del verifier ALLENATO, se il modello esiste (parita' col professore)
    model_metrics = None
    model_dir = BASE / "models" / "kh2_verifier" / "best_model"
    try:
        import paths_config as _cfg  # type: ignore
        if getattr(_cfg, "VERIFIER_MODEL_DIR", ""):
            model_dir = Path(_cfg.VERIFIER_MODEL_DIR)
    except Exception:
        pass
    if model_dir.exists():
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tok = AutoTokenizer.from_pretrained(str(model_dir))
            mdl = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()
            tn = fp = fn = tp = 0
            for r in test_rows:
                with torch.no_grad():
                    logits = mdl(**tok(r["text"], return_tensors="pt",
                                       truncation=True, max_length=192)).logits
                pred_valid = mdl.config.id2label[int(logits.argmax(-1))] == "VALID"
                gold_valid = (r["label"] == "VALID")
                tp += gold_valid and pred_valid
                fn += gold_valid and not pred_valid
                fp += (not gold_valid) and pred_valid
                tn += (not gold_valid) and not pred_valid
            model_metrics = _binary_metrics([[tn, fp], [fn, tp]])
            model_metrics["evaluated_on"] = "kh2_verifier_test.jsonl"
            model_metrics["classifier"] = "trained BERT verifier"
            (out_dir / "verifier_metrics_model.json").write_text(
                json.dumps(model_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            (out_dir / "verifier_metrics_model.json").write_text(
                json.dumps({"note": f"modello presente ma valutazione non riuscita: {e}"},
                           indent=2, ensure_ascii=False), encoding="utf-8")

    # curve/matrici di training se presenti
    for name in ["verifier_training_curve.png", "verifier_confusion_matrix.png",
                 "ner_training_curve.png", "ner_confusion_matrix.png"]:
        _copy(GEN / name, out_dir)

    # --- benchmark multi-modello (step 12) + baseline SpaCy (step 07b) ---------
    bench = None
    if (GEN / "benchmark_models.json").exists():
        bench = json.loads((GEN / "benchmark_models.json").read_text(encoding="utf-8"))
    for name in ["benchmark_models.json", "benchmark_comparison.csv",
                 "benchmark_model_comparison.png", "benchmark_comparison_table.png",
                 "spacy_baseline_metrics.json",
                 "spacy_baseline_triples.jsonl", "complexity_stats.json"]:
        _copy(GEN / name, out_dir)
    # tutte le metriche/grafici per-modello prodotti dal benchmark
    for f in sorted(GEN.glob("verifier_metrics_*.json")):
        _copy(f, out_dir)
    for f in sorted(GEN.glob("verifier_training_curve_*.png")):
        _copy(f, out_dir)
    for f in sorted(GEN.glob("verifier_confusion_matrix_*.png")):
        _copy(f, out_dir)
    return triple, vm, model_metrics, bench


def export_sparql(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    from rdflib import Graph
    qdir = GEN / "queries"
    for rq in sorted(qdir.glob("*.rq")):
        _copy(rq, out_dir)
    g = Graph().parse(GEN / "knowledge_graph.ttl", format="turtle")

    results = {}
    txt_lines = []
    for rq in sorted(qdir.glob("*.rq")):
        q = rq.read_text(encoding="utf-8")
        qres = g.query(q)
        vars_ = [str(v) for v in (qres.vars or [])]
        body = []
        for row in qres:
            body.append({v: (_short(str(row[v])) if row[v] is not None else None)
                         for v in (qres.vars or [])})
        results[rq.stem] = {"rows": len(body), "data": body}
        txt_lines.append(f"===== {rq.stem} =====")
        txt_lines.append(f"({len(body)} rows)")
        for b in body:
            txt_lines.append(" | ".join(f"{k}={v}" for k, v in b.items()))
        txt_lines.append("")
    (out_dir / "sparql_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "sparql_results.txt").write_text("\n".join(txt_lines), encoding="utf-8")
    return results


def write_benchmark_report(out_dir: Path, triple, vm, model_metrics, bench, sparql, kg, onto_rep, ds):
    out_dir.mkdir(parents=True, exist_ok=True)
    pr = triple
    per_rel = pr.get("per_relation_recall", {})
    rel_lines = "\n".join(
        f"| {r} | {v['hit']}/{v['total']} | {(_safe_div(v['hit'], v['total'])):.3f} |"
        for r, v in sorted(per_rel.items()))
    m = vm["metrics"]

    # ---- Sezione 5: confronto multi-architettura (Slide 13) -------------------
    if bench and bench.get("systems"):
        rowsb = bench["systems"]
        header = ("| Sistema | Livello | Arch | F1 | Precision | Recall | Speed (ex/s) | Note |\n"
                  "|---|---|---|---|---|---|---|---|\n")
        body = "\n".join(
            "| {sys} | {lvl} | {arch} | {f1:.3f} | {p:.3f} | {r:.3f} | {sp} | {note} |".format(
                sys=s["system"], lvl=s["level"], arch=s["arch"], f1=s["f1"],
                p=s["precision"], r=s["recall"],
                sp=("—" if s.get("throughput_examples_per_sec") is None
                    else f"{s['throughput_examples_per_sec']:.0f}"),
                note=s.get("note", ""))
            for s in rowsb)
        n_tf = bench.get("transformer_count", 0)
        scatter = ""
        if (GEN / "benchmark_model_comparison.png").exists():
            scatter += "\n\n![Confronto modelli — P/R/F1](benchmark_model_comparison.png)\n"
        if (GEN / "benchmark_comparison_table.png").exists():
            scatter += "\n![Tabella comparativa](benchmark_comparison_table.png)\n"
        if (GEN / "benchmark_tier_f1.png").exists():
            scatter += "\n![F1 per categoria di frase (per modello)](benchmark_tier_f1.png)\n"
        # tabella per-tier (solo sistemi verifier che la espongono)
        tier_tbl = ""
        tier_rows = [s for s in rowsb if s.get("f1_by_complexity_tier")]
        if tier_rows:
            all_tiers = ["explicit", "implicit", "long_distance", "nested"]
            present = [t for t in all_tiers
                       if any(t in s["f1_by_complexity_tier"] for s in tier_rows)]
            th = "| Sistema | " + " | ".join(f"F1 {t}" for t in present) + " |\n"
            th += "|---|" + "|".join("---" for _ in present) + "|\n"
            tb = "\n".join(
                "| " + s["slug"] + " | " + " | ".join(
                    (f"{s['f1_by_complexity_tier'][t]['f1']:.3f}" if t in s["f1_by_complexity_tier"] else "—")
                    for t in present) + " |"
                for s in tier_rows)
            tier_tbl = ("\n\n### F1 per livello di complessita' (Slide 5)\n"
                        "Dove i modelli cedono: tipicamente recall in calo su `implicit`/`nested`.\n\n"
                        + th + tb + "\n")

        # tabella: accuratezza nel RICONOSCERE i negativi, per tipo (hard negatives)
        neg_tbl = ""
        neg_rows = [s for s in rowsb if s.get("reject_accuracy_by_neg_type")]
        if neg_rows:
            order = ["type_identical", "partial_overlap", "type_incompatible"]
            present = [n for n in order if any(n in s["reject_accuracy_by_neg_type"] for s in neg_rows)]
            th = "| Sistema | " + " | ".join(present) + " |\n|---|" + "|".join("---" for _ in present) + "|\n"
            tb = "\n".join(
                "| " + s["slug"] + " | " + " | ".join(
                    (f"{s['reject_accuracy_by_neg_type'][n]['reject_accuracy']:.3f}"
                     if n in s["reject_accuracy_by_neg_type"] else "—") for n in present) + " |"
                for s in neg_rows)
            neg_tbl = ("\n\n### Riconoscimento dei negativi per tipo (hard negatives)\n"
                       "Frazione di negativi correttamente respinti. I `type_identical` hanno gli stessi "
                       "dominio/codominio del vero: la **regola li sbaglia** (≈0), i modelli che leggono il "
                       "testo no — e' qui che il benchmark discrimina.\n\n" + th + tb + "\n")

        # tabella riassuntiva ESTRAZIONE per modello (per-story)
        extr_tbl = ""
        extr = bench.get("extraction_per_story") or []
        if extr:
            th = ("| Sistema | Story | Story perfette | Micro P | Micro R | Micro F1 | Macro F1 |\n"
                  "|---|---|---|---|---|---|---|\n")
            tb = "\n".join(
                f"| {e['system']} | {e['stories']} | {e['perfect_stories']} | "
                f"{e['micro_precision']:.3f} | {e['micro_recall']:.3f} | {e['micro_f1']:.3f} | "
                f"{e['macro_f1']:.3f} |" for e in extr)
            extr_tbl = ("\n\n### Estrazione triple — riassunto per modello (per-story)\n"
                        "Media su tutte le story; dettaglio riga-per-story in `per_story_<sistema>.csv`.\n\n"
                        + th + tb + "\n")

        refs = ("\n\n### Altri artefatti in Results Data\n"
                "- `02_dataset/dataset_split.md` — split train/val/test con POS/NEG e tipi di negativo.\n"
                "- `01_ontology/ontology_structure.md` — classi, sottoclassi, proprieta' (dominio→codominio).\n"
                "- `04_metrics/verifier_confusion_matrix_<modello>_{train,val,test}.png` — 3 matrici per verifier.\n"
                "- `03_knowledge_graph/kg_predicted_vs_actual*.png` — grafo TP/FP/FN (Slide 11).\n"
                "- `06_benchmark_report/model_summary_extraction.csv` — riassunto estrazione.\n")
        notes_txt = ""
        if bench.get("notes"):
            notes_txt = "\n\n> Note di esecuzione:\n" + "\n".join(f"> - {n}" for n in bench["notes"])
        bench_section = f"""## 5. Benchmark multi-architettura del verifier (Slide 13)
Confronto tra famiglie di Transformer (come validatore VALID/INVALID) e i baseline,
su **F1** e **velocita'/efficienza** (throughput di inferenza). Transformer allenati
in questa esecuzione: **{n_tf}**.

{header}{body}
{scatter}{tier_tbl}{neg_tbl}{extr_tbl}{notes_txt}
{refs}
I file: `benchmark_comparison.csv`, `benchmark_models.json`,
`benchmark_model_comparison.png`, `benchmark_comparison_table.png`,
e un `verifier_metrics_<slug>.json` per ogni Transformer.
"""
    else:
        bench_section = """## 5. Benchmark multi-architettura del verifier (Slide 13)
In questa esecuzione i Transformer non sono stati allenati (nessuna GPU / accesso ai
modelli in ambiente locale): sono riportati i baseline **regola** (sez. 3) e
**SpaCy** (sopra). Per popolare il confronto tra famiglie (RoBERTa, DeBERTa v3,
BigBird, XLNet, T5, DistilBERT) con F1 e throughput:

```
python script/01b_augment_complexity.py     # dati a complessita' stratificata
python script/07b_spacy_baseline.py          # baseline generatore SpaCy
python script/12_benchmark_models.py --models all --epochs 3
python script/11_export_results_data.py      # rigenera questo report
```
Lo step 12 produce la tabella comparativa e lo scatter F1-vs-Speed, che compaiono
automaticamente in questa sezione.
"""

    # ---- baseline SpaCy (livello tripla), se presente ------------------------
    spacy_block = ""
    sp_path = GEN / "spacy_baseline_metrics.json"
    if sp_path.exists():
        sp = json.loads(sp_path.read_text(encoding="utf-8"))
        spacy_block = (
            "\n### Baseline SpaCy (generator, livello tripla) — Slide 7\n"
            f"Estrazione fragile senza gate semantico: precision {sp['precision']:.3f} · "
            f"recall {sp['recall']:.3f} · F1 {sp['f1']:.3f} "
            f"(predette {sp['predicted']}, gold {sp['gold']}).\n")

    md = f"""# Benchmarking Report — Kingdom Hearts 2 NLP Pipeline

Deliverable #4 della Project Description (Block 14): *Comprehensive analysis of the
baseline vs. Transformer architectures*. Numeri **misurati** da questa esecuzione.

## 1. Setup
- Ontologia: {onto_rep['counts']['classes']} classi, {onto_rep['counts']['object_properties']} object property,
  {onto_rep['counts']['data_properties']} data property, {onto_rep['counts']['individuals']} individui.
  Coerenza: {onto_rep['abox_violations']} violazioni ABox.
- Dataset sintetico annotato: split documenti {ds['synthetic_corpus']['document_split']}
  ({ds['synthetic_corpus']['document_split_percent']} %), target 70/15/15, con complessita'
  stratificata (explicit/implicit/long_distance/nested) se 01b e' stato eseguito.
- KG RDF + validazione SHACL: **Conforms = {kg.get('conforms')}**.

## 2. Estrazione end-to-end a livello tripla (baseline deterministico regola/dizionario)
Confronto triple predette vs gold ({pr['gold']} triple gold).

| Metrica | Valore |
|---|---|
| TP / FP / FN | {pr['tp']} / {pr['fp']} / {pr['fn']} |
| Precision | {pr['precision']:.3f} |
| Recall | {pr['recall']:.3f} |
| F1 | {pr['f1']:.3f} |

### Recall per relazione
| Relazione | hit/total | recall |
|---|---|---|
{rel_lines}
{spacy_block}
## 3. Verifier — matrice di confusione (regola dominio/codominio, TEST set)
Classificatore binario VALID/INVALID applicato a {vm['applicable_object_property_rows']}
esempi object-property del test set.

Layout `[[TN, FP], [FN, TP]]` = `[[{vm['confusion_matrix']['tn']}, {vm['confusion_matrix']['fp']}], [{vm['confusion_matrix']['fn']}, {vm['confusion_matrix']['tp']}]]`

| Metrica | Valore |
|---|---|
| Accuracy | {m['accuracy']:.3f} |
| Precision | {m['precision']:.3f} |
| Recall (sensitivity) | {m['recall_sensitivity']:.3f} |
| Specificity | {m['specificity']:.3f} |
| F1 | {m['f1_score']:.3f} |
| Balanced accuracy | {m['balanced_accuracy']:.3f} |
| Matthews corrcoef | {m['matthews_corrcoef']:.3f} |

## 4. Interrogazione del KG (SPARQL, FILTER / OPTIONAL)
{chr(10).join(f"- **{k}**: {v['rows']} righe" for k, v in sparql.items())}

{bench_section}"""
    (out_dir / "benchmark_report.md").write_text(md, encoding="utf-8")
    # porta i grafici/CSV del benchmark accanto al report
    for name in ["benchmark_model_comparison.png", "benchmark_comparison_table.png",
                 "benchmark_comparison.csv"]:
        _copy(GEN / name, out_dir)


def write_index(triple, vm, kg, onto_rep, ds):
    idx = f"""# Results Data — indice

Tutti i dati richiesti alla fine del progetto ontologia (Project Description, Block 14),
generati automaticamente da `11_export_results_data.py`. Path tutti relativi.

## Mappa Deliverable -> cartella
| # | Deliverable (slide professore) | Cartella |
|---|---|---|
| 1 | Custom Domain Ontology (OWL, domain/range) | `01_ontology/` |
| 2 | Synthetic Annotated Dataset (JSONL, Train/Val/Test) | `02_dataset/` |
| 3 | Working Software Pipeline (RDF/KG, SHACL) | `03_knowledge_graph/` |
| 4 | Benchmarking Report (P/R/F1, confusion matrix, SPARQL) | `04_metrics/`, `05_sparql/`, `06_benchmark_report/` |

## Numeri chiave di questa esecuzione
- Ontologia: {onto_rep['counts']['classes']} classi · {onto_rep['counts']['object_properties']} object property ·
  {onto_rep['counts']['data_properties']} data property · {onto_rep['counts']['individuals']} individui ·
  violazioni ABox: {onto_rep['abox_violations']}.
- Dataset: split documenti {ds['synthetic_corpus']['document_split']} (target 70/15/15).
- Triple end-to-end: precision {triple['precision']:.3f} · recall {triple['recall']:.3f} · F1 {triple['f1']:.3f}.
- Verifier (regola, test): F1 {vm['metrics']['f1_score']:.3f} · accuracy {vm['metrics']['accuracy']:.3f}.
- SHACL Conforms: {kg.get('conforms')}.

## Contenuto cartelle
- `01_ontology/`: `kingdom_hearts2.owx`, `ontology_coherence_report.{{txt,json}}`, `ontology_interface.json`.
- `02_dataset/`: split verifier/relation/NER (`*_train/val/test.jsonl`), `synthetic_annotated_dataset/`,
  `kh2_instance_triples.jsonl` (gold), `dataset_statistics.json`, file `*_stats.json`.
- `03_knowledge_graph/`: `knowledge_graph.ttl`, `shapes.ttl`, `shacl_validation.txt`,
  `extracted_triples.jsonl`, `last_inference_result.json`.
- `04_metrics/`: `triple_metrics.json` + grafici PNG, `verifier_metrics_rule.json`
  (e `verifier_metrics_model.json` se il modello e' stato allenato).
- `05_sparql/`: `query1-3.rq`, `sparql_results.{{txt,json}}`.
- `06_benchmark_report/`: `benchmark_report.md`.
"""
    (RESULTS / "README_RESULTS.md").write_text(idx, encoding="utf-8")


def main():
    if RESULTS.exists():
        shutil.rmtree(RESULTS)
    RESULTS.mkdir(parents=True, exist_ok=True)

    onto_rep = export_ontology(RESULTS / "01_ontology")
    # tabelle strutturali dell'ontologia (classi/sottoclassi/proprieta')
    RX.ontology_structure_md(onto, RESULTS / "01_ontology" / "ontology_structure.md")

    ds = export_dataset(RESULTS / "02_dataset")
    # tabella dataset split (POS/NEG + per neg_type) per architettura
    RX.dataset_split_table(GEN, GEN / "dataset_split.csv", GEN / "dataset_split.md")
    for f in ["dataset_split.csv", "dataset_split.md"]:
        _copy(GEN / f, RESULTS / "02_dataset")

    kg = export_kg(RESULTS / "03_knowledge_graph")
    # figura KG Predicted vs Actual (se 13 ha girato)
    for f in sorted(GEN.glob("kg_predicted_vs_actual*.png")) + sorted(GEN.glob("kg_pred_vs_actual*.png")):
        _copy(f, RESULTS / "03_knowledge_graph")

    triple, vm, model_metrics, bench = export_metrics(RESULTS / "04_metrics")
    # tabelle performance per-story (estrazione): regola + SpaCy (+ transformer se presenti)
    RX.extraction_summaries(GEN)
    for f in sorted(GEN.glob("per_story_*.csv")) + sorted(GEN.glob("per_story_*.md")):
        _copy(f, RESULTS / "04_metrics")
    _copy(GEN / "model_summary_extraction.csv", RESULTS / "06_benchmark_report")
    # 3 matrici di confusione (train/val/test) per ogni verifier transformer
    for f in sorted(GEN.glob("verifier_confusion_matrix_*_*.png")):
        _copy(f, RESULTS / "04_metrics")

    sparql = export_sparql(RESULTS / "05_sparql")
    # grafi RDF dei risultati SPARQL (stile esempio_grapho_RDF) prodotti da 09
    for f in sorted(GEN.glob("sparql_graph_*.png")):
        _copy(f, RESULTS / "05_sparql")
    write_benchmark_report(RESULTS / "06_benchmark_report", triple, vm, model_metrics,
                           bench, sparql, kg, onto_rep, ds)

    # matrici di confusione 3-split + riassunto verifier per split (regola e modelli presenti)
    bsys = (bench or {}).get("systems", [])
    for s in bsys:
        ps = s.get("per_split") or {}
        for split, d in ps.items():
            cmimg = RESULTS / "04_metrics" / f"verifier_confusion_matrix_{s.get('slug','rule')}_{split}.png"
            if not cmimg.exists() and d.get("confusion_matrix"):
                RX.save_cm_png(d["confusion_matrix"], ["INVALID", "VALID"],
                               f"Verifier ({s.get('slug')}) — {split} confusion matrix", cmimg)
    RX.verifier_summary_table(bsys, RESULTS / "06_benchmark_report" / "verifier_summary.csv",
                              RESULTS / "06_benchmark_report" / "verifier_summary.md")

    # CATCH-ALL: nessun grafico deve restare solo in generated/. Ogni PNG prodotto
    # viene instradato nella sottocartella tematica di Results Data (Q5).
    routing = [("kg_", "03_knowledge_graph"), ("sparql_graph_", "05_sparql"),
               ("benchmark_", "06_benchmark_report"), ("triple_", "04_metrics"),
               ("verifier_", "04_metrics"), ("ner_", "04_metrics")]
    already = {p.name for p in RESULTS.rglob("*.png")}
    swept = 0
    for png in sorted(GEN.glob("*.png")):
        if png.name in already:
            continue
        sub = next((d for pre, d in routing if png.name.startswith(pre)), "04_metrics")
        _copy(png, RESULTS / sub)
        swept += 1
    if swept:
        print(f"  grafici aggiuntivi spostati in Results Data: {swept}")

    write_index(triple, vm, kg, onto_rep, ds)

    rel = RESULTS.relative_to(REPO) if str(RESULTS).startswith(str(REPO)) else RESULTS
    print("=" * 60)
    print(f"Results Data esportati in: {rel}/")
    print(f"  triple P/R/F1 ... {triple['precision']:.3f}/{triple['recall']:.3f}/{triple['f1']:.3f}")
    print(f"  verifier (regola) F1 ... {vm['metrics']['f1_score']:.3f}")
    print(f"  SHACL Conforms .. {kg.get('conforms')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
