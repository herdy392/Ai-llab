# MANUALE D'UTILIZZO — Kingdom Hearts 2 NLP Ontology Pipeline

Guida passo-passo. Il progetto produce **tutti** i dati richiesti alla fine del
progetto ontologia (vedi la Project Description del professore) e li raccoglie
nella cartella **`Results Data/`**. Funziona **senza GPU**.

---

## 0. Perché `.py` e non `.bat`?

I file `.bat` sono script di Windows: si avviano con doppio click ma funzionano
**solo su Windows**. Per questo ho aggiunto **`run_all.py`**, un launcher in Python
che fa le stesse cose ma gira su **Windows, Linux e Mac** con lo stesso comando.
I `.bat` restano nel progetto (comodi su Windows), ma il modo consigliato e
multipiattaforma è `run_all.py`.

---

## 1. Requisiti (una volta sola)

Serve **Python 3.10–3.12** (consigliato 3.11). In una console (PowerShell/Terminale)
posizionati nella cartella del progetto ed esegui:

```bash
# Windows
py -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

```bash
# Linux / Mac
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Per la sola pipeline CPU (dati + KG + SHACL + metriche + SPARQL + Results Data)
bastano `rdflib, pyshacl, numpy, scikit-learn, matplotlib`. `torch`/`transformers`
servono **solo** per il training (step 03/04) e per `10_query_model.py`.

---

## 2. Esecuzione completa in un comando

```bash
python run_all.py
```

Esegue in ordine gli step `00,01,02,05,06,07,08,09` e infine `11` che crea
`Results Data/`. Si ferma al primo errore mostrando quale step è fallito.

Varianti utili:

```bash
python run_all.py --list             # elenca gli step
python run_all.py --only 05 06 07    # esegue solo alcuni step
python run_all.py --with-training    # aggiunge 03 (verifier) e 04 (NER) — serve torch
```

> Su Windows puoi anche fare doppio click su `run_all_cpu.bat` (equivalente, ora
> include lo step 11).

---

## 3. Cosa fa ogni step (in dettaglio)

| Step | File | Cosa produce |
|---|---|---|
| 00 | `00_check_ontology.py` | Controlla l'ontologia: conteggi classi/relazioni/attributi/individui, domini/codomini, inverse, ABox, descrizioni. |
| 01 | `01_build_corpus.py` | Genera il corpus schema e il dataset del **verifier** (token speciali `[REL][E1][E2]`) + le **triple gold** d'istanza. |
| 02 | `02_build_weak_dataset.py` | Dataset debole: **relation candidates** e **NER** (con split train/val/test). |
| 03 | `03_train_verifier.py` | *(opzionale, GPU)* Allena il verifier BERT → `models/kh2_verifier/best_model`. |
| 04 | `04_train_ner.py` | *(opzionale, GPU)* Allena il NER → `models/kh2_ner/best_model`. |
| 05 | `05_run_end_to_end.py` | Estrazione **end-to-end** delle triple dal testo (usa il verifier se presente, altrimenti la regola dominio/codominio). |
| 06 | `06_build_kg.py` | Costruisce il **Knowledge Graph RDF** (`knowledge_graph.ttl`). |
| 07 | `07_validate_shacl.py` | Validazione **SHACL** → `Conforms: True/False`. |
| 08 | `08_compute_metrics.py` | **Precision/Recall/F1** a livello tripla + grafici PNG. |
| 09 | `09_sparql_queries.py` | 3 query **SPARQL** (FILTER/OPTIONAL) sul KG; salva i `.rq`. |
| 11 | `11_export_results_data.py` | **Raccoglie tutto in `Results Data/`** (vedi sotto). |

---

## 4. Dove trovo i risultati: la cartella `Results Data/`

Dopo l'esecuzione trovi alla radice del progetto:

```
Results Data/
├── README_RESULTS.md            indice + numeri chiave
├── 01_ontology/                 ontologia OWL + report di coerenza + interfaccia
├── 02_dataset/                  dataset annotato JSONL (train/val/test) + statistiche split
├── 03_knowledge_graph/          KG .ttl + shapes SHACL + esito Conforms + triple + inferenza
├── 04_metrics/                  triple_metrics.json + grafici PNG + matrice di confusione verifier
├── 05_sparql/                   query .rq + risultati (txt/json)
└── 06_benchmark_report/         benchmark_report.md (relazione finale)
```

Ogni cartella corrisponde a uno dei 4 deliverable della Project Description
(Ontology, Synthetic Dataset, Working Pipeline, Benchmarking Report).

---

## 5. Training dei modelli (opzionale, per il benchmark Transformer)

Solo se vuoi i numeri dei **modelli allenati** (oltre alla baseline a regola):

```bash
python nlp-detection/script/03_train_verifier.py --epochs 4 --batch-size 16
python nlp-detection/script/04_train_ner.py --epochs 5 --batch-size 16
python run_all.py --only 05 08 11    # rigenera estrazione, metriche e Results Data col modello
```

Dopo il training, `Results Data/04_metrics/` conterrà anche
`verifier_metrics_model.json` e le curve/matrici di confusione PNG.

### Benchmark multi-modello (step 12) — riuso vs retraining

Il benchmark confronta più famiglie di Transformer come verifier
(DistilBERT, BERT, RoBERTa-base, XLNet, BigBird, T5) più i baseline non-Transformer
e i modelli **zero-shot** delle architetture avanzate (**GLiNER** per l'estrazione
entità, **GLiREL** per la candidate generation di relazioni — usati solo se i
pacchetti `gliner`/`glirel` sono installati, altrimenti vengono saltati con una nota).

**Per default riusa i training precedenti**: se in `models/kh2_verifier_<slug>/best_model`
esiste già un modello con le sue metriche, lo step non riaddestra e riprende i numeri salvati.
Per **forzare il riaddestramento da zero** di tutti i modelli usa `--retrain` (alias `--reitrain`):

```bash
python nlp-detection/script/12_benchmark_models.py                 # riusa i training esistenti (default)
python nlp-detection/script/12_benchmark_models.py --retrain       # forza il retraining di tutti i modelli
python run_all.py --with-benchmark                                 # benchmark riusando i training
python run_all.py --with-benchmark --retrain                       # benchmark + retraining forzato
```

Lo stesso vale per `03_train_verifier.py` e `04_train_ner.py`: per default riusano il
modello già salvato, con `--retrain` lo riaddestrano.

---

## 6. Path: relativi di default, assoluti opzionali

Tutti i path nel codice sono **relativi** e risolti rispetto alla posizione dei
file (`Path(__file__)...`): puoi spostare o rinominare la cartella del progetto e
tutto continua a funzionare. **Non compare nessun path tipo `/home/...`.**

Se ti serve forzare un path **assoluto** (es. un corpus o un modello salvato
**fuori** dal progetto), apri **`paths_config.py`** alla radice e compila solo le
voci che ti interessano (lascia `""` per usare il default):

| Variabile | A cosa serve | Esempio |
|---|---|---|
| `RESULTS_DIR` | Cartella di output di "Results Data" | `r"D:\consegne\Results Data"` |
| `CORPUS_PATH` | Corpus di input alternativo per lo step 05 | `r"D:\dati\mio_corpus.txt"` |
| `VERIFIER_MODEL_DIR` | Cartella del verifier allenato | `r"D:\modelli\kh2_verifier\best_model"` |
| `NER_MODEL_DIR` | Cartella del NER allenato | `r"D:\modelli\kh2_ner\best_model"` |

Regole per le stringhe:
- **Windows**: usa la forma raw con la `r` davanti → `r"C:\Users\nome\file.txt"`.
- **Linux/Mac**: `"/home/nome/file.txt"`.
- Se lasci `""`, viene usato il percorso relativo interno al progetto.

---

## 7. Risoluzione problemi

| Sintomo | Soluzione |
|---|---|
| `'python' non riconosciuto` | Usa `py`, o reinstalla Python con "Add Python to PATH". |
| `ModuleNotFoundError` | Attiva il venv ed esegui `pip install -r requirements.txt`. |
| SPARQL: "Knowledge graph not found" | Esegui prima `06_build_kg.py` (o `python run_all.py`). |
| Training lentissimo | Sei su CPU: usa il notebook Colab con GPU o riduci `--epochs`. |
| PowerShell blocca il venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| I PNG non si aggiornano | Installa `matplotlib`; i grafici si rigenerano a ogni run. |
