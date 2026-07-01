# Kingdom Hearts 2 — NLP Ontology Pipeline

Pipeline NLP end-to-end costruita sull'ontologia di **Kingdom Hearts 2**: a partire da testo
grezzo individua entità e relazioni dell'ontologia, le valida, costruisce un **knowledge
graph** RDF, lo controlla con **SHACL** e lo interroga in **SPARQL**. Include il training di
un **verifier di relazioni** (BERT) e di un **NER**, con i relativi grafici di accuratezza.

La struttura ricalca il progetto del professore (`Ontology logistic` → `nlp-detection/`).

## Struttura
```
Kh2/
├─ ontology/kingdom_hearts2.owx              copia "master" dell'ontologia
├─ requirements.txt
├─ run_all_cpu.bat                       pipeline completa senza GPU (1 click)
├─ run_dataprep.bat / run_inference.bat
├─ run_train_verifier.bat / run_train_ner.bat
└─ nlp-detection/
   ├─ data-input/kh2_corpus.txt          testo ABox di partenza
   ├─ ontology/kingdom_hearts2.owx           ontologia usata dalla pipeline
   ├─ script/                            01..10 + kh2_lib.py + _schema_templates.json
   ├─ notebook/                          2 notebook (locale + Colab) + compute_metrics.py
   ├─ generated/                         output: jsonl, ttl, json, PNG dei grafici
   └─ inference_demo/last_inference_result.json
```

## Avvio rapido (Windows 11/10, senza GPU)
```bat
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
run_all_cpu.bat
```
Output principali in `nlp-detection\generated\`:
`triple_metrics.png` (precision/recall/F1), `triple_recall_by_relation.png`,
`knowledge_graph.ttl`, `triple_metrics.json`, `queries\query{1,2,3}.rq`.

## Gli script (in `nlp-detection/script/`)
| # | Script | Cosa fa | GPU |
|---|--------|---------|-----|
| 00 | `00_check_ontology.py` | verifica coerenza ontologia (domini/range/inverse/ABox/descrizioni) | no |
| 01 | `01_build_corpus.py` | genera corpus schema + verifier + testo ABox + gold | no |
| 02 | `02_build_weak_dataset.py` | weak supervision: candidati relazione + dati NER | no |
| 03 | `03_train_verifier.py` | training verifier BERT (+ grafici) | sì* |
| 04 | `04_train_ner.py` | training NER BIO (+ grafici) | sì* |
| 05 | `05_run_end_to_end.py` | estrazione triple da testo (rule o verifier) | no/sì |
| 06 | `06_build_kg.py` | costruisce `knowledge_graph.ttl` (relazioni + attributi) | no |
| 07 | `07_validate_shacl.py` | validazione SHACL del KG | no |
| 08 | `08_compute_metrics.py` | precision/recall/F1 + grafici | no |
| 09 | `09_sparql_queries.py` | 3 query SPARQL (FILTER / OPTIONAL) | no |
| 10 | `10_query_model.py` | interroga verifier / NER trainati | sì |

\* funziona anche su CPU, solo più lento.

## Training + grafici
```bat
run_train_verifier.bat      :: oppure il notebook Colab per la GPU gratuita
```
Grafici in `generated\`: `verifier_training_curve.png`, `verifier_confusion_matrix.png`,
metriche in `verifier_metrics.json`.

## SPARQL (3 query con FILTER / OPTIONAL)
```bat
cd nlp-detection
python script\09_sparql_queries.py
```
Q1 usa **FILTER**, Q2 usa **OPTIONAL**, Q3 usa **OPTIONAL + FILTER**. I file `.rq` finiscono
in `generated\queries\` e sono apribili anche in Protégé / GraphDB / Fuseki.

## Interrogare il modello trainato
```bat
cd nlp-detection
python script\10_query_model.py verifier --rel ownsKeyblade --subject Character --object Keyblade
python script\10_query_model.py ner --text "Sora owns the keyblade Kingdom Chain in Agrabah."
```

Dettagli completi e risoluzione problemi nel file **MANUALE.md** (e nel PDF allegato).
