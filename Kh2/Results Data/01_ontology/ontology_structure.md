# Struttura dell'ontologia

Classi: 35 · object property: 18 · data property: 5 · individui: 148

## Classi e sottoclassi

| Classe | Sottoclassi |
|---|---|
| Action | CombatAction, DriveForm, ItemUse, MovementAction, Summon, Support |
| Boss | — |
| Character | NonPlayableCharacter, PlayableCharacter |
| CombatAction | MagicAttack, SwordAttack |
| CommonNobody | — |
| Consumable | Gems, Munny, Potion |
| DisneyAntagonist | — |
| DriveForm | — |
| Enemy | Boss, DisneyAntagonist, Heartless, Nobody |
| Entity | Character, Organization, Party |
| Fight | — |
| Gems | — |
| Heartless | — |
| ItemUse | — |
| Items | Consumable, NonConsumable |
| Keyblade | — |
| MagicAttack | — |
| MovementAction | — |
| Munny | — |
| NPC | — |
| Nobody | CommonNobody, Org13Member |
| NonConsumable | Keyblade, SecondaryWeapon |
| NonPlayableCharacter | Enemy, NPC |
| Org13Member | — |
| Organization | — |
| Party | — |
| PlayableCharacter | Protagonist, RecruitableAlly |
| Potion | — |
| Protagonist | — |
| RecruitableAlly | — |
| SecondaryWeapon | — |
| Summon | — |
| Support | — |
| SwordAttack | — |
| Worlds | — |

## Object property (dominio → codominio)

| Object property | Dominio | Codominio |
|---|---|---|
| appearsIn | Entity | Worlds |
| canSummon | Protagonist | Summon |
| comesFrom | Entity | Worlds |
| defeats | Entity | Entity |
| drops | Entity | Items |
| engagesInFight | Party | Fight |
| hasAction | Entity | Action |
| hasDriveForm | Protagonist | DriveForm |
| hasFriend | Entity | Entity |
| hasInInventory | Protagonist | Items |
| hasPartyMember | Party | Entity |
| hasSecondaryWeapon | RecruitableAlly | SecondaryWeapon |
| isEnemyOf | — | — |
| isMemberOf | Org13Member | Organization |
| ownsKeyblade | Entity | Keyblade |
| participatesIn | Entity | Party |
| takesPlaceIn | Fight | Worlds |
| winner | Fight | Party |

## Data property (dominio)

| Data property | Dominio |
|---|---|
| attackPower | Keyblade |
| description | — |
| element | — |
| hitPoints | Entity |
| name | — |

Classi radice (senza superclasse dichiarata): Action, Entity, Fight, Items, Worlds
