Cartella progetto Ontologia AILAB 25/26 CdL ACSAI (Anche se noi siamo CdL Informatica)

Il Github ha due folder, la prima con codice e ontologia Matteo (Armored Core 6), la seconda con codice e ontologia Andrea (Kingdom Hearts 2)

Il codice alla base della pipeline è lo stesso per entrambi, l'ontologia e il testo di training è ovviamente diverso

Il codice è stato fatto per runnare su Windows, richiede gli import manuali per alcuni dei transformer (verrà scritto in console se mancano o meno) e per runnare su GPU richiede python 3.12 con torch + cu121. 

Per l'avvio su dati già trainati è sufficiente runnare python runall con --with-benchmarks 
Per retrainare i modelli (o trainarli la prima volta) python runall con --with-benchmarks --retrain
