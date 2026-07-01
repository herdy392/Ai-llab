# Ai-llab — NLP Pipeline su Ontologia (Armored Core 6 / Kingdom Hearts 2) README ITALIANO

Questo repo contiene **due istanze dello stesso progetto**: una pipeline NLP che, a partire da
testo in linguaggio naturale, costruisce un Knowledge Graph (KG) basato su un'ontologia OWL,
lo valida con SHACL, lo interroga in SPARQL e confronta diverse architetture di modelli
(regola, BERT/RoBERTa/DistilBERT/T5, GLiNER/GLiREL, LLM) per l'estrazione e la validazione
delle relazioni.

Le due cartelle condividono **codice e struttura identici** — cambia solo l'ontologia di
dominio alla base:

- [`ArmoredCore_ontology_2026/`](ArmoredCore_ontology_2026) — ontologia di *Armored Core 6*
  (12 classi, 18 object property, 6 data property, 98 individui).
- [`Kh2/`](Kh2) — ontologia di *Kingdom Hearts 2*
  (35 classi, 18 object property, 5 data property, 148 individui).

## Pipeline (in ciascuna cartella)

Dal testo grezzo → individuo entità e relazioni dell'ontologia → le valido (regola, modello
BERT, o LLM) → costruisco un Knowledge Graph RDF → lo controllo con SHACL → lo interrogo in
SPARQL → confronto le architetture di modelli con un benchmark. Ogni passo è uno script
numerato in `nlp-detection/script/`, eseguito in ordine dal launcher `run_all.py`.

| Step | Script | Cosa fa |
|---|---|---|
| 00 | `00_check_ontology.py` | Verifica coerenza ontologia (domini/range/inverse/ABox) |
| 01 | `01_build_corpus.py` | Genera corpus schema + dataset verifier + testo ABox + gold |
| 01b | `01b_augment_complexity.py` | Complessità stratificata + split disgiunto anti-leakage |
| 02 | `02_build_weak_dataset.py` | Weak supervision: candidati di relazione + dati NER |
| 03 | `03_train_verifier.py` | Training del verifier BERT (riusa se già allenato) |
| 04 | `04_train_ner.py` | Training del NER BIO (riusa se già allenato) |
| 05 | `05_run_end_to_end.py` | Estrazione triple dal testo (regola / modello / LLM) |
| 06 | `06_build_kg.py` | Costruisce il Knowledge Graph RDF (`knowledge_graph.ttl`) |
| 07 | `07_validate_shacl.py` | Validazione SHACL del KG (Conforms) |
| 07b | `07b_spacy_baseline.py` | Baseline SpaCy (fragile) per il confronto |
| 08 | `08_compute_metrics.py` | Precision/Recall/F1 a livello tripla + grafici |
| 09 | `09_sparql_queries.py` | 3 query SPARQL (FILTER/OPTIONAL) + grafi RDF |
| 11 | `11_export_results_data.py` | Raccoglie tutto in `Results Data` |
| 12 | `12_benchmark_models.py` | Benchmark multi-modello (verifier) + grafico/tabella |
| 13 | `13_kg_figure.py` | Figura KG Predicted vs Actual (TP/FP/FN) |

## Come si esegue (per ciascun progetto)


| Comando | Cosa fò |
|---|---|
| `python run_all.py` | Pipeline completa CPU + export in `Results Data` |
| `python run_all.py --with-training` | Aggiunge il training di verifier e NER |
| `python run_all.py --with-benchmark` | Aggiunge il benchmark multi-modello |
| `python run_all.py --with-benchmark --retrain` | Benchmark riaddestrando tutti i modelli da zero |
| `python run_all.py --only 05 06 07` | Esegue solo alcuni step |

Tutto l'output della console viene salvato automaticamente in `latest.log`, perchè un log fa sempre bene.

## Risultati

Ogni progetto raccoglie il deliverable finale in `Results Data/`, organizzato in sei
sottocartelle (`01_ontology`, `02_dataset`, `03_knowledge_graph`, `04_metrics`, `05_sparql`,
`06_benchmark_report`), con un `README_RESULTS.md` che fa da indice e un `README.md` di
benchmarking con le metriche misurate nell'ultima esecuzione.


# Ai-llab — NLP Pipeline su Ontologia (Armored Core 6 / Kingdom Hearts 2) README INGLESE

This repo contains **two instances of the same project**: an NLP pipeline that, starting
from natural language text, builds a Knowledge Graph (KG) based on an OWL ontology, validates
it with SHACL, queries it in SPARQL, and benchmarks different model architectures (rule,
BERT/RoBERTa/DistilBERT/T5, GLiNER/GLiREL, LLM) for relation extraction and validation.

The two folders share **identical code and structure** — only the domain ontology changes:

- [`ArmoredCore_ontology_2026/`](ArmoredCore_ontology_2026) — *Armored Core 6* ontology
  (12 classes, 18 object properties, 6 data properties, 98 individuals).
- [`Kh2/`](Kh2) — *Kingdom Hearts 2* ontology
  (35 classes, 18 object properties, 5 data properties, 148 individuals).

## Pipeline (in each folder)

From raw text → identify ontology entities and relations → validate them (rule, BERT model,
or LLM) → build an RDF Knowledge Graph → check it with SHACL → query it in SPARQL → compare
model architectures with a benchmark. Each step is a numbered script in
`nlp-detection/script/`, run in order by the `run_all.py` launcher.

| Step | Script | What it does |
|---|---|---|
| 00 | `00_check_ontology.py` | Checks ontology consistency (domains/ranges/inverses/ABox) |
| 01 | `01_build_corpus.py` | Generates schema corpus + verifier dataset + ABox text + gold |
| 01b | `01b_augment_complexity.py` | Stratified complexity + disjoint anti-leakage split |
| 02 | `02_build_weak_dataset.py` | Weak supervision: relation candidates + NER data |
| 03 | `03_train_verifier.py` | Trains the BERT verifier (reuses it if already trained) |
| 04 | `04_train_ner.py` | Trains the BIO NER model (reuses it if already trained) |
| 05 | `05_run_end_to_end.py` | Extracts triples from text (rule / model / LLM) |
| 06 | `06_build_kg.py` | Builds the RDF Knowledge Graph (`knowledge_graph.ttl`) |
| 07 | `07_validate_shacl.py` | SHACL validation of the KG (Conforms) |
| 07b | `07b_spacy_baseline.py` | SpaCy baseline (fragile) for comparison |
| 08 | `08_compute_metrics.py` | Precision/Recall/F1 at triple level + charts |
| 09 | `09_sparql_queries.py` | 3 SPARQL queries (FILTER/OPTIONAL) + RDF graphs |
| 11 | `11_export_results_data.py` | Collects everything into `Results Data` |
| 12 | `12_benchmark_models.py` | Multi-model benchmark (verifier) + chart/table |
| 13 | `13_kg_figure.py` | KG Predicted vs Actual figure (TP/FP/FN) |

## How to run it (per project)

| Command | What it does |
|---|---|
| `python run_all.py` | Full CPU pipeline + export to `Results Data` |
| `python run_all.py --with-training` | Adds verifier and NER training |
| `python run_all.py --with-benchmark` | Adds the multi-model benchmark |
| `python run_all.py --with-benchmark --retrain` | Benchmark retraining all models from scratch |
| `python run_all.py --only 05 06 07` | Runs only some steps |

All console output is automatically saved to `latest.log`, because a log never hurts.

## Results

Each project collects the final deliverable in `Results Data/`, organized into six
subfolders (`01_ontology`, `02_dataset`, `03_knowledge_graph`, `04_metrics`, `05_sparql`,
`06_benchmark_report`), with a `README_RESULTS.md` acting as an index and a `README.md`
with benchmarking metrics measured in the latest run.