# Benchmarking Report — Kingdom Hearts 2 NLP Pipeline

Deliverable #4 della Project Description (Block 14): *Comprehensive analysis of the
baseline vs. Transformer architectures*. Numeri **misurati** da questa esecuzione.

## 1. Setup
- Ontologia: 35 classi, 18 object property,
  5 data property, 148 individui.
  Coerenza: 0 violazioni ABox.
- Dataset sintetico annotato: split documenti {'train': 80, 'val': 10, 'test': 10}
  ({'train': 80.0, 'val': 10.0, 'test': 10.0} %), target 70/15/15, con complessita'
  stratificata (explicit/implicit/long_distance/nested) se 01b e' stato eseguito.
- KG RDF + validazione SHACL: **Conforms = True**.

## 2. Estrazione end-to-end a livello tripla (baseline deterministico regola/dizionario)
Confronto triple predette vs gold (254 triple gold).

| Metrica | Valore |
|---|---|
| TP / FP / FN | 254 / 0 / 0 |
| Precision | 1.000 |
| Recall | 1.000 |
| F1 | 1.000 |

### Recall per relazione
| Relazione | hit/total | recall |
|---|---|---|
| appearsIn | 45/45 | 1.000 |
| comesFrom | 13/13 | 1.000 |
| defeats | 27/27 | 1.000 |
| drops | 2/2 | 1.000 |
| engagesInFight | 28/28 | 1.000 |
| hasAction | 22/22 | 1.000 |
| hasDriveForm | 6/6 | 1.000 |
| hasFriend | 24/24 | 1.000 |
| hasInInventory | 9/9 | 1.000 |
| hasPartyMember | 19/19 | 1.000 |
| hasSecondaryWeapon | 2/2 | 1.000 |
| isMemberOf | 12/12 | 1.000 |
| ownsKeyblade | 17/17 | 1.000 |
| takesPlaceIn | 14/14 | 1.000 |
| winner | 14/14 | 1.000 |

### Baseline SpaCy (generator, livello tripla) — Slide 7
Estrazione fragile senza gate semantico: precision 1.000 · recall 1.000 · F1 1.000 (predette 254, gold 254).

## 3. Verifier — matrice di confusione (regola dominio/codominio, TEST set)
Classificatore binario VALID/INVALID applicato a 1711
esempi object-property del test set.

Layout `[[TN, FP], [FN, TP]]` = `[[949, 243], [0, 519]]`

| Metrica | Valore |
|---|---|
| Accuracy | 0.858 |
| Precision | 0.681 |
| Recall (sensitivity) | 1.000 |
| Specificity | 0.796 |
| F1 | 0.810 |
| Balanced accuracy | 0.898 |
| Matthews corrcoef | 0.736 |

## 4. Interrogazione del KG (SPARQL, FILTER / OPTIONAL)
- **query1**: 24 righe
- **query2**: 109 righe
- **query3**: 13 righe

## 5. Benchmark multi-architettura del verifier (Slide 13)
Confronto tra famiglie di Transformer (come validatore VALID/INVALID) e i baseline,
su **F1** e **velocita'/efficienza** (throughput di inferenza). Transformer allenati
in questa esecuzione: **4**.

| Sistema | Livello | Arch | F1 | Precision | Recall | Speed (ex/s) | Note |
|---|---|---|---|---|---|---|---|
| distilbert-base-uncased | verifier | encoder | 0.924 | 0.964 | 0.886 | 838 | leggero/veloce |
| bert-base-uncased | verifier | encoder | 0.931 | 0.950 | 0.913 | 422 | BERT 'ontology-compiled model' (generatore/baseline encoder) |
| roberta-base | verifier | encoder | 0.902 | 0.915 | 0.890 | 420 | Supervised RoBERTa (validatore binario VALID/INVALID) |
| t5-base | verifier | seq2seq | 0.872 | 0.915 | 0.832 | 55 | encoder-decoder (text-to-text) |
| rule (domain/range) | verifier | symbolic | 0.810 | 0.681 | 1.000 | 262315 | surface-invariant: cade sui negativi type_identical |
| SpaCy baseline (generator) | triple | rule/parse | 1.000 | 1.000 | 1.000 | — | estrazione fragile, niente gate semantico (Slide 7) |
| GLiNER (zero-shot NER) | entity | gliner | 0.280 | 0.212 | 0.412 | 11 | estrazione entita' zero-shot (Arch. A/B), nessun training |


![Confronto modelli — P/R/F1](benchmark_model_comparison.png)

![Tabella comparativa](benchmark_comparison_table.png)

![F1 per categoria di frase (per modello)](benchmark_tier_f1.png)


### F1 per livello di complessita' (Slide 5)
Dove i modelli cedono: tipicamente recall in calo su `implicit`/`nested`.

| Sistema | F1 explicit | F1 implicit | F1 long_distance | F1 nested |
|---|---|---|---|---|
| distilbert | 1.000 | 0.687 | 0.997 | 0.991 |
| bert | 1.000 | 0.729 | 1.000 | 1.000 |
| roberta | 0.979 | 0.638 | 0.994 | 1.000 |
| t5 | 0.996 | 0.454 | 0.997 | 1.000 |
| rule | 0.774 | 0.822 | 0.848 | 0.784 |


### Riconoscimento dei negativi per tipo (hard negatives)
Frazione di negativi correttamente respinti. I `type_identical` hanno gli stessi dominio/codominio del vero: la **regola li sbaglia** (≈0), i modelli che leggono il testo no — e' qui che il benchmark discrimina.

| Sistema | type_identical | partial_overlap | type_incompatible |
|---|---|---|---|
| rule | 0.000 | 0.932 | 0.972 |


### Estrazione triple — riassunto per modello (per-story)
Media su tutte le story; dettaglio riga-per-story in `per_story_<sistema>.csv`.

| Sistema | Story | Story perfette | Micro P | Micro R | Micro F1 | Macro F1 |
|---|---|---|---|---|---|---|
| rule | 254 | 254 | 1.000 | 1.000 | 1.000 | 1.000 |
| spacy | 254 | 254 | 1.000 | 1.000 | 1.000 | 1.000 |
| bert | 254 | 254 | 1.000 | 1.000 | 1.000 | 1.000 |
| distilbert | 254 | 254 | 1.000 | 1.000 | 1.000 | 1.000 |
| roberta | 254 | 254 | 1.000 | 1.000 | 1.000 | 1.000 |
| t5 | 254 | 254 | 1.000 | 1.000 | 1.000 | 1.000 |


> Note di esecuzione:
> - deberta-v3 (microsoft/deberta-v3-base): training/eval fallito -> Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
> - xlnet (xlnet-base-cased): training/eval fallito -> Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
> - bigbird (google/bigbird-roberta-base): training/eval fallito -> Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6 in order to use the function. This version restriction does not apply when loading files with safetensors.
See the vulnerability report here https://nvd.nist.gov/vuln/detail/CVE-2025-32434
> - GLiREL saltato: caricamento modello fallito (GLiREL._from_pretrained() missing 2 required keyword-only arguments: 'proxies' and 'resume_download').


### Altri artefatti in Results Data
- `02_dataset/dataset_split.md` — split train/val/test con POS/NEG e tipi di negativo.
- `01_ontology/ontology_structure.md` — classi, sottoclassi, proprieta' (dominio→codominio).
- `04_metrics/verifier_confusion_matrix_<modello>_{train,val,test}.png` — 3 matrici per verifier.
- `03_knowledge_graph/kg_predicted_vs_actual*.png` — grafo TP/FP/FN (Slide 11).
- `06_benchmark_report/model_summary_extraction.csv` — riassunto estrazione.

I file: `benchmark_comparison.csv`, `benchmark_models.json`,
`benchmark_model_comparison.png`, `benchmark_comparison_table.png`,
e un `verifier_metrics_<slug>.json` per ogni Transformer.
