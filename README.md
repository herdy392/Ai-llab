# Ai-llab — NLP Pipeline su Ontologia (Armored Core 6 / Kingdom Hearts 2)

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

Prerequisito: attivare il virtual environment (Python 3.12), che contiene già i
requirements e i modelli salvati.

```
.venv\Scripts\activate
```

| Comando | Cosa fa |
|---|---|
| `python run_all.py` | Pipeline completa CPU + export in `Results Data` |
| `python run_all.py --with-training` | Aggiunge il training di verifier e NER |
| `python run_all.py --with-benchmark` | Aggiunge il benchmark multi-modello |
| `python run_all.py --with-benchmark --retrain` | Benchmark riaddestrando tutti i modelli da zero |
| `python run_all.py --only 05 06 07` | Esegue solo alcuni step |

Tutto l'output della console viene salvato automaticamente in `latest.log`.

## Risultati

Ogni progetto raccoglie il deliverable finale in `Results Data/`, organizzato in sei
sottocartelle (`01_ontology`, `02_dataset`, `03_knowledge_graph`, `04_metrics`, `05_sparql`,
`06_benchmark_report`), con un `README_RESULTS.md` che fa da indice e un `README.md` di
benchmarking con le metriche misurate nell'ultima esecuzione.
