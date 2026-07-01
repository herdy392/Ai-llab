# -*- coding: utf-8 -*-
# =====================================================================
#  CONFIGURAZIONE PATH (opzionale)
# =====================================================================
# Il progetto funziona SENZA toccare questo file: tutti i path sono gia'
# relativi e vengono risolti rispetto alla posizione degli script
# (Path(__file__)...), quindi puoi spostare/rinominare la cartella del
# progetto e tutto continua a funzionare.
#
# Imposta un valore QUI SOTTO solo se vuoi forzare un path ASSOLUTO,
# tipicamente perche' un file/cartella si trova FUORI dal progetto.
# Lascia la stringa vuota "" per usare il default relativo.
#
# Regole:
#   - Usa una stringa raw su Windows: r"C:\Users\tuonome\dati\corpus.txt"
#   - Su Linux/Mac: "/home/tuonome/dati/corpus.txt"
#   - Niente virgolette mancanti, niente barre rovesciate singole su Windows.
# =====================================================================

# Cartella dove scrivere "Results Data". Vuoto = <progetto>/Results Data
RESULTS_DIR = ""

# Corpus di input alternativo per l'estrazione end-to-end (script 05).
# Vuoto = nlp-detection/data-input/kh2_corpus.txt
CORPUS_PATH = ""

# Cartella del modello VERIFIER allenato (per metriche del modello in 11 e per 05/10).
# Vuoto = nlp-detection/models/kh2_verifier/best_model
VERIFIER_MODEL_DIR = ""

# Cartella del modello NER allenato.
# Vuoto = nlp-detection/models/kh2_ner/best_model
NER_MODEL_DIR = ""
