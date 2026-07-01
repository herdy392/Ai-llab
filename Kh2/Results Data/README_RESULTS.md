# Results Data — indice

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
- Ontologia: 35 classi · 18 object property ·
  5 data property · 148 individui ·
  violazioni ABox: 0.
- Dataset: split documenti {'train': 80, 'val': 10, 'test': 10} (target 70/15/15).
- Triple end-to-end: precision 1.000 · recall 1.000 · F1 1.000.
- Verifier (regola, test): F1 0.810 · accuracy 0.858.
- SHACL Conforms: True.

## Contenuto cartelle
- `01_ontology/`: `kingdom_hearts2.owx`, `ontology_coherence_report.{txt,json}`, `ontology_interface.json`.
- `02_dataset/`: split verifier/relation/NER (`*_train/val/test.jsonl`), `synthetic_annotated_dataset/`,
  `kh2_instance_triples.jsonl` (gold), `dataset_statistics.json`, file `*_stats.json`.
- `03_knowledge_graph/`: `knowledge_graph.ttl`, `shapes.ttl`, `shacl_validation.txt`,
  `extracted_triples.jsonl`, `last_inference_result.json`.
- `04_metrics/`: `triple_metrics.json` + grafici PNG, `verifier_metrics_rule.json`
  (e `verifier_metrics_model.json` se il modello e' stato allenato).
- `05_sparql/`: `query1-3.rq`, `sparql_results.{txt,json}`.
- `06_benchmark_report/`: `benchmark_report.md`.
