# Dataset split del verifier (per architettura)

Conteggi per split, con i POSITIVI e i NEGATIVI scomposti per tipo di negativo (`type_identical` = stessi dominio/codominio del vero, non rifiutabili dalla regola).

| Split | Tot | POS | NEG | type_identical | partial_overlap | type_incompatible |
|---|---|---|---|---|---|---|
| train | 8064 | 2615 | 5449 | 435 | 1892 | 3122 |
| val | 2215 | 988 | 1227 | 153 | 432 | 642 |
| test | 1642 | 541 | 1101 | 80 | 440 | 581 |

Conteggio per-relazione completo in dataset_split.csv
