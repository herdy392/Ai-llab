# Benchmarking Report — Armored Core 6 NLP Pipeline

Deliverable #4 della Project Description (Block 14): *Comprehensive analysis of the
baseline vs. Transformer architectures*. Numeri **misurati** da questa esecuzione.

## 1. Setup
- Ontologia: 12 classi, 18 object property,
  6 data property, 98 individui.
  Coerenza: 0 violazioni ABox.
- Dataset sintetico annotato: split documenti {'train': 88, 'val': 11, 'test': 11}
  ({'train': 80.0, 'val': 10.0, 'test': 10.0} %), target 70/15/15, con complessita'
  stratificata (explicit/implicit/long_distance/nested) se 01b e' stato eseguito.
- KG RDF + validazione SHACL: **Conforms = True**.

## 2. Estrazione end-to-end a livello tripla (baseline deterministico regola/dizionario)
Confronto triple predette vs gold (160 triple gold).

| Metrica | Valore |
|---|---|
| TP / FP / FN | 160 / 0 / 0 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |

### Recall per relazione
| Relazione | hit/total | recall |
|---|---|---|
| hasAlias | 30/30 | 1.000 |
| hasRank | 6/6 | 1.000 |
| isBossOf | 11/11 | 1.000 |
| isFriendlyTowards | 10/10 | 1.000 |
| isHostileToward | 10/10 | 1.000 |
| kills | 5/5 | 1.000 |
| participatesIn | 45/45 | 1.000 |
| pilots | 17/17 | 1.000 |
| takesPlaceIn | 12/12 | 1.000 |
| worksFor | 14/14 | 1.000 |

### Baseline SpaCy (generator, livello tripla) — Slide 7
Estrazione fragile senza gate semantico: precision 0.844 · recall 0.844 · F1 0.844 (predette 160, gold 160).

## 3. Verifier — matrice di confusione (regola dominio/codominio, TEST set)
Classificatore binario VALID/INVALID applicato a 1404
esempi object-property del test set.

Layout `[[TN, FP], [FN, TP]]` = `[[898, 99], [0, 407]]`

| Metrica | Valore |
|---|---|
| Accuracy | 0.929 |
| Precision | 0.804 |
| Recall (sensitivity) | 1.000 |
| Specificity | 0.901 |
| F1 | 0.892 |
| Balanced accuracy | 0.950 |
| Matthews corrcoef | 0.851 |

## 4. Interrogazione del KG (SPARQL, FILTER / OPTIONAL)
- **query1**: 10 righe
- **query2**: 30 righe
- **query3**: 38 righe

## 5. Benchmark multi-architettura del verifier (Slide 13)
Confronto tra famiglie di Transformer (come validatore VALID/INVALID) e i baseline,
su **F1** e **velocita'/efficienza** (throughput di inferenza). Transformer allenati
in questa esecuzione: **4**.

| Sistema | Livello | Arch | F1 | Precision | Recall | Speed (ex/s) | Note |
|---|---|---|---|---|---|---|---|
| distilbert-base-uncased | verifier | encoder | 0.830 | 0.982 | 0.719 | 671 | leggero/veloce |
| bert-base-uncased | verifier | encoder | 0.831 | 0.985 | 0.719 | 341 | BERT 'ontology-compiled model' (generatore/baseline encoder) |
| roberta-base | verifier | encoder | 0.878 | 0.979 | 0.795 | 344 | Supervised RoBERTa (validatore binario VALID/INVALID) |
| t5-base | verifier | seq2seq | 0.869 | 0.884 | 0.856 | 131 | encoder-decoder (text-to-text) |
| rule (domain/range) | verifier | symbolic | 0.892 | 0.804 | 1.000 | 299061 | surface-invariant: cade sui negativi type_identical |
| SpaCy baseline (generator) | triple | rule/parse | 0.844 | 0.844 | 0.844 | — | estrazione fragile, niente gate semantico (Slide 7) |
| GLiNER (zero-shot NER) | entity | gliner | 0.342 | 0.329 | 0.355 | 7 | estrazione entita' zero-shot (Arch. A/B), nessun training |


![Confronto modelli — P/R/F1](benchmark_model_comparison.png)

![Tabella comparativa](benchmark_comparison_table.png)

![F1 per categoria di frase (per modello)](benchmark_tier_f1.png)


### F1 per livello di complessita' (Slide 5)
Dove i modelli cedono: tipicamente recall in calo su `implicit`/`nested`.

| Sistema | F1 explicit | F1 implicit | F1 long_distance | F1 nested |
|---|---|---|---|---|
| distilbert | 0.985 | 0.000 | 1.000 | 0.941 |
| bert | 0.990 | 0.000 | 0.997 | 0.944 |
| roberta | 1.000 | 0.310 | 1.000 | 1.000 |
| t5 | 0.937 | 0.526 | 0.956 | 0.993 |
| rule | 0.955 | 0.962 | 0.866 | 0.788 |


### Riconoscimento dei negativi per tipo (hard negatives)
Frazione di negativi correttamente respinti. I `type_identical` hanno gli stessi dominio/codominio del vero: la **regola li sbaglia** (≈0), i modelli che leggono il testo no — e' qui che il benchmark discrimina.

| Sistema | type_identical | partial_overlap | type_incompatible |
|---|---|---|---|
| rule | 0.000 | 1.000 | 0.960 |


### Estrazione triple — riassunto per modello (per-story)
Media su tutte le story; dettaglio riga-per-story in `per_story_<sistema>.csv`.

| Sistema | Story | Story perfette | Micro P | Micro R | Micro F1 | Macro F1 |
|---|---|---|---|---|---|---|
| rule | 320 | 160 | 1.000 | 0.500 | 0.667 | 0.500 |
| spacy | 320 | 135 | 0.844 | 0.422 | 0.562 | 0.422 |
| bert | 320 | 254 | 0.836 | 0.794 | 0.814 | 0.794 |
| distilbert | 320 | 230 | 0.821 | 0.719 | 0.767 | 0.719 |
| roberta | 320 | 270 | 0.844 | 0.844 | 0.844 | 0.844 |
| t5 | 320 | 270 | 0.844 | 0.844 | 0.844 | 0.844 |


> Note di esecuzione:
> - deberta-v3 (microsoft/deberta-v3-base): training/eval fallito -> Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
> - xlnet (xlnet-base-cased): training/eval fallito -> Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
> - bigbird (google/bigbird-roberta-base): training/eval fallito -> Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
> - GLiREL saltato: pacchetto 'glirel' non installato (No module named 'loguru').


### Altri artefatti in Results Data
- `02_dataset/dataset_split.md` — split train/val/test con POS/NEG e tipi di negativo.
- `01_ontology/ontology_structure.md` — classi, sottoclassi, proprieta' (dominio→codominio).
- `04_metrics/verifier_confusion_matrix_<modello>_{train,val,test}.png` — 3 matrici per verifier.
- `03_knowledge_graph/kg_predicted_vs_actual*.png` — grafo TP/FP/FN (Slide 11).
- `06_benchmark_report/model_summary_extraction.csv` — riassunto estrazione.

I file: `benchmark_comparison.csv`, `benchmark_models.json`,
`benchmark_model_comparison.png`, `benchmark_comparison_table.png`,
e un `verifier_metrics_<slug>.json` per ogni Transformer.
