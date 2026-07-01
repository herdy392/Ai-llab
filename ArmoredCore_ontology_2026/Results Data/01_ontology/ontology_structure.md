# Struttura dell'ontologia

Classi: 12 · object property: 18 · data property: 6 · individui: 98

## Classi e sottoclassi

| Classe | Sottoclassi |
|---|---|
| AI | — |
| Alias | — |
| ArmoredCore | — |
| Boss | — |
| Character | AI, Boss, Human, Rubiconian |
| Entity | Character, Society |
| Human | — |
| MajorEvent | — |
| Place | — |
| Rank | — |
| Rubiconian | — |
| Society | — |

## Object property (dominio → codominio)

| Object property | Dominio | Codominio |
|---|---|---|
| employs | Society | Character |
| hasAlias | Character | Alias |
| hasBoss | MajorEvent | Boss |
| hasRank | Character | Rank |
| hostsEvent | Place | MajorEvent |
| involvesEntity | MajorEvent | Entity |
| isAliasOf | Alias | Character |
| isBossOf | Boss | MajorEvent |
| isFriendlyTowards | Entity | Entity |
| isHostileToward | Entity | Entity |
| isKilledBy | Character | Character |
| isPilotedBy | ArmoredCore | Character |
| isRankOf | Rank | Character |
| kills | Character | Character |
| participatesIn | Entity | MajorEvent |
| pilots | Character | ArmoredCore |
| takesPlaceIn | MajorEvent | Place |
| worksFor | Character | Society |

## Data property (dominio)

| Data property | Dominio |
|---|---|
| acName | ArmoredCore |
| chronologicalOrder | MajorEvent |
| isBuddy | Character |
| isSpecial | ArmoredCore |
| name | Society |
| originatesOnRubicon | Society |

Classi radice (senza superclasse dichiarata): Alias, ArmoredCore, Entity, MajorEvent, Place, Rank
