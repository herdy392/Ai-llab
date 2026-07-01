# Dataset split del verifier (per architettura)

Conteggi per split, con i POSITIVI e i NEGATIVI scomposti per tipo di negativo (`type_identical` = stessi dominio/codominio del vero, non rifiutabili dalla regola).

| Split | Tot | POS | NEG | type_identical | partial_overlap | type_incompatible |
|---|---|---|---|---|---|---|
| train | 8461 | 2595 | 5866 | 973 | 2355 | 2538 |
| val | 1660 | 474 | 1186 | 184 | 451 | 551 |
| test | 1711 | 519 | 1192 | 195 | 498 | 499 |

Conteggio per-relazione completo in dataset_split.csv
