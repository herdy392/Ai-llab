# -*- coding: utf-8 -*-
"""Launcher unico, multipiattaforma (Windows / Linux / Mac).

Sostituisce i file .bat: gli .bat funzionano SOLO su Windows, questo file gira
ovunque ci sia Python. Esegue gli script nello stesso ordine, fermandosi al
primo errore, usando l'interprete Python corrente (sys.executable) e path
relativi alla posizione di questo file.

Esempi:
  python run_all.py                # pipeline completa CPU + esporta "Results Data"
  python run_all.py --with-training  # come sopra + training verifier e NER (serve torch)
  python run_all.py --only 05 06 07  # esegue solo alcuni step
  python run_all.py --list           # elenca gli step disponibili
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
SCRIPTS = REPO / "nlp-detection" / "script"

# Pipeline base, senza GPU. Ogni voce: (codice, file, descrizione)
CPU_STEPS = [
    ("00", "00_check_ontology.py",      "Verifica coerenza ontologia"),
    ("01", "01_build_corpus.py",        "Costruisce corpus + dataset verifier (schema)"),
    ("02", "02_build_weak_dataset.py",  "Dataset debole (relation candidates + NER)"),
    ("01b","01b_augment_complexity.py", "Complessita' stratificata (explicit/implicit/long/nested)"),
    ("05", "05_run_end_to_end.py",      "Estrazione end-to-end (triple)"),
    ("06", "06_build_kg.py",            "Costruisce il Knowledge Graph RDF"),
    ("07", "07_validate_shacl.py",      "Validazione SHACL (Conforms)"),
    ("07b","07b_spacy_baseline.py",     "Baseline SpaCy relation extraction (Slide 7)"),
    ("08", "08_compute_metrics.py",     "Metriche tripla P/R/F1 + grafici"),
    ("09", "09_sparql_queries.py",      "Query SPARQL (FILTER/OPTIONAL)"),
    ("13", "13_kg_figure.py",           "Figura KG Predicted vs Actual (TP/FP/FN)"),
    ("11", "11_export_results_data.py", "Esporta tutto in 'Results Data'"),
]

TRAINING_STEPS = [
    ("03", "03_train_verifier.py", "Training verifier singolo (serve torch)"),
    ("04", "04_train_ner.py",      "Training NER (serve torch)"),
]

# Benchmark multi-architettura: allena/valuta piu' Transformer + baseline (Slide 13).
BENCHMARK_STEPS = [
    ("12", "12_benchmark_models.py", "Benchmark multi-modello (verifier) + scatter F1/speed"),
]


LOG_PATH = REPO / "latest.log"


def _tee(line, log):
    """Scrive una riga sia in console sia nel file di log latest.log.
    Robusto rispetto alla code page della console Windows (cp1252)."""
    try:
        sys.stdout.write(line)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", "ascii") or "ascii"
        sys.stdout.write(line.encode(enc, errors="replace").decode(enc, errors="replace"))
    sys.stdout.flush()
    if log is not None:
        log.write(line)
        log.flush()


def run_step(code, fname, desc, extra_args=None, log=None):
    script = SCRIPTS / fname
    header = "\n" + "=" * 64 + f"\n[{code}] {desc}\n" + "=" * 64 + "\n"
    _tee(header, log)
    cmd = [sys.executable, "-u", str(script)] + list(extra_args or [])
    # Cattura stdout+stderr combinati e li ritrasmette in tempo reale a console + log.
    proc = subprocess.Popen(cmd, cwd=str(SCRIPTS.parent), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", bufsize=1)
    for line in proc.stdout:
        _tee(line, log)
    proc.wait()
    if proc.returncode != 0:
        _tee(f"\n*** ERRORE nello step {code} ({fname}). Esecuzione interrotta. ***\n", log)
        if log is not None:
            log.close()
        sys.exit(proc.returncode)


def main():
    # Console UTF-8 dove possibile (evita crash di encoding su Windows cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Launcher pipeline Kingdom Hearts 2 NLP.")
    ap.add_argument("--with-training", action="store_true",
                    help="esegue anche 03 (verifier singolo) e 04 (NER) prima dell'estrazione")
    ap.add_argument("--with-benchmark", action="store_true",
                    help="esegue il benchmark multi-modello (12) prima dell'export (serve torch per i Transformer)")
    ap.add_argument("--retrain", "--reitrain", dest="retrain", action="store_true",
                    help="forza il riaddestramento negli step 03/04/12 (default: riusa i training precedenti)")
    ap.add_argument("--only", nargs="+", metavar="CODE",
                    help="esegue solo gli step con questi codici (es. 05 06 07)")
    ap.add_argument("--list", action="store_true", help="elenca gli step e termina")
    args = ap.parse_args()

    all_steps = CPU_STEPS[:]
    if args.with_training:
        # training dopo il dataset (02/01b), prima dell'estrazione (05)
        idx = next(i for i, s in enumerate(all_steps) if s[0] == "05")
        all_steps = all_steps[:idx] + TRAINING_STEPS + all_steps[idx:]
    if args.with_benchmark:
        # benchmark prima dell'export (11), cosi' il report ne raccoglie i risultati
        idx = next(i for i, s in enumerate(all_steps) if s[0] == "11")
        all_steps = all_steps[:idx] + BENCHMARK_STEPS + all_steps[idx:]

    if args.list:
        for code, fname, desc in all_steps:
            print(f"  {code}  {fname:32s} {desc}")
        return

    steps = all_steps
    if args.only:
        wanted = set(args.only)
        steps = [s for s in all_steps if s[0] in wanted]
        if not steps:
            print("Nessuno step corrisponde a", args.only)
            sys.exit(2)

    # Step che supportano --retrain (riusano i risultati salvati se non forzati).
    retrain_codes = {"03", "04", "07b", "12"}

    def extra_for(code):
        ex = []
        # Il benchmark deve confrontare TUTTE le famiglie nominate, non solo il set CPU.
        if code == "12":
            ex += ["--models", "all"]
        if args.retrain and code in retrain_codes:
            ex += ["--retrain"]
        return ex or None

    log = open(LOG_PATH, "w", encoding="utf-8")
    try:
        _tee(f"LOG: {LOG_PATH}\nComando: {' '.join(sys.argv)}\n", log)
        for code, fname, desc in steps:
            run_step(code, fname, desc, extra_for(code), log=log)

        _tee("\n" + "=" * 64 + "\n", log)
        _tee("PIPELINE OK. Risultati in: 'Results Data/'  e  nlp-detection/generated/\n", log)
        _tee(f"Log completo salvato in: {LOG_PATH}\n", log)
    finally:
        log.close()


if __name__ == "__main__":
    main()
