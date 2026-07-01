# -*- coding: utf-8 -*-
"""Shared utilities for the Kingdom Hearts 2 NLP pipeline (local, Windows-friendly).
Pure standard library: parses the OWL/XML ontology with regex, builds the entity
dictionary, verbalises relations, splits sentences and detects mentions.
Mirrors the conventions of the professor's logistic-ontology pipeline
(entity dictionary + weak supervision + relation verifier).
"""
from __future__ import annotations
import re, json, html, hashlib
from pathlib import Path
from collections import defaultdict

NS = "http://www.semanticweb.org/andrea/ontologies/2026/4/Kingdom_hearts_2#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"

# NER coarse types (Kingdom Hearts 2): un individuo viene assegnato alla PRIMA classe
# di questo ordine di priorita' presente fra le sue classi (o i loro antenati).
# Completamente guidato dall'ontologia: vale per qualsiasi individuo KH2.
NER_TYPE_ORDER = ["Keyblade", "Summon", "DriveForm", "Worlds", "Organization",
                  "Party", "Fight", "SecondaryWeapon", "Items", "Action",
                  "Enemy", "Character"]

# Verbalizzazione delle relazioni KH2 (per le frasi). Per qualsiasi proprieta' non
# elencata si ricade su camel_to_words (es. "ownsKeyblade" -> "owns keyblade").
REL_PHRASE = {"rdfs:subClassOf": "is a subclass of",
 "appearsIn": "appears in", "comesFrom": "comes from", "takesPlaceIn": "takes place in",
 "ownsKeyblade": "owns the keyblade", "hasFriend": "is friends with", "isEnemyOf": "is the enemy of",
 "defeats": "defeats", "isMemberOf": "is a member of", "hasDriveForm": "has the drive form",
 "canSummon": "can summon", "hasInInventory": "has in the inventory", "hasPartyMember": "has as party member",
 "hasSecondaryWeapon": "has the secondary weapon", "participatesIn": "participates in",
 "winner": "is won by", "engagesInFight": "engages in the fight", "hasAction": "can perform", "drops": "drops",
 "name": "has name", "description": "is described as", "element": "has element",
 "attackPower": "has attack power", "hitPoints": "has hit points"}


def camel_to_words(name: str) -> str:
    if name == "LiteralValue":
        return "literal value"
    if ":" in name:
        name = name.split(":", 1)[1]
    parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name).replace("_", " ").split()
    return " ".join(p.lower() for p in parts)


def relation_phrase(p: str) -> str:
    return REL_PHRASE.get(p, camel_to_words(p))


def parse_ontology(owx_path: str | Path) -> dict:
    t = Path(owx_path).read_text(encoding="utf-8")
    # Tollera l'OWL/XML "pretty-printed" (con a capo/indentazione fra i tag), come
    # quello esportato da Protege per Kingdom Hearts 2: collassa gli spazi fra i tag
    # cosi' le regex single-line qui sotto funzionano senza altre modifiche.
    t = re.sub(r">\s+<", "><", t)
    classes = re.findall(r'<Declaration><Class IRI="#([^"]+)"/></Declaration>', t)
    objprops = re.findall(r'<Declaration><ObjectProperty IRI="#([^"]+)"/></Declaration>', t)
    dataprops = re.findall(r'<Declaration><DataProperty IRI="#([^"]+)"/></Declaration>', t)
    individuals = re.findall(r'<Declaration><NamedIndividual IRI="#([^"]+)"/></Declaration>', t)
    labels = {}
    for m in re.finditer(r'<AnnotationAssertion><AnnotationProperty abbreviatedIRI="rdfs:label"/><IRI>#([^<]+)</IRI><Literal>([^<]*)</Literal>', t):
        labels[m.group(1)] = html.unescape(m.group(2))
    odomain, orange = {}, {}
    for m in re.finditer(r'<ObjectPropertyDomain><ObjectProperty IRI="#([^"]+)"/><Class IRI="#([^"]+)"/></ObjectPropertyDomain>', t):
        odomain[m.group(1)] = m.group(2)
    for m in re.finditer(r'<ObjectPropertyRange><ObjectProperty IRI="#([^"]+)"/><Class IRI="#([^"]+)"/></ObjectPropertyRange>', t):
        orange[m.group(1)] = m.group(2)
    ddomain = {}
    for m in re.finditer(r'<DataPropertyDomain><DataProperty IRI="#([^"]+)"/><Class IRI="#([^"]+)"/></DataPropertyDomain>', t):
        ddomain[m.group(1)] = m.group(2)
    subclass = defaultdict(set)
    for m in re.finditer(r'<SubClassOf><Class IRI="#([^"]+)"/><Class IRI="#([^"]+)"/></SubClassOf>', t):
        subclass[m.group(1)].add(m.group(2))
    ind_classes = defaultdict(list)
    for m in re.finditer(r'<ClassAssertion><Class IRI="#([^"]+)"/><NamedIndividual IRI="#([^"]+)"/></ClassAssertion>', t):
        ind_classes[m.group(2)].append(m.group(1))
    # asserted ABox triples
    obj_assertions = re.findall(r'<ObjectPropertyAssertion><ObjectProperty IRI="#([^"]+)"/><NamedIndividual IRI="#([^"]+)"/><NamedIndividual IRI="#([^"]+)"/></ObjectPropertyAssertion>', t)
    data_assertions = []
    for m in re.finditer(r'<DataPropertyAssertion><DataProperty IRI="#([^"]+)"/><NamedIndividual IRI="#([^"]+)"/><Literal[^>]*>([^<]*)</Literal></DataPropertyAssertion>', t):
        data_assertions.append((m.group(1), m.group(2), m.group(3)))
    # coppie inverse dichiarate (se presenti nell'ontologia)
    inverses = re.findall(r'<InverseObjectProperties><ObjectProperty IRI="#([^"]+)"/><ObjectProperty IRI="#([^"]+)"/></InverseObjectProperties>', t)

    def ancestors(cls, acc=None):
        acc = acc if acc is not None else set()
        for sup in subclass.get(cls, ()):
            if sup not in acc:
                acc.add(sup); ancestors(sup, acc)
        return acc

    def primary_nature(ind):
        """Classe piu' SPECIFICA asserita per l'individuo (quella con piu' antenati);
        usata per verbalizzare il tipo di soggetto/oggetto."""
        cs = ind_classes.get(ind, [])
        if not cs:
            return "Entity"
        return max(cs, key=lambda c: len(ancestors(c)))

    return {"classes": classes, "objprops": objprops, "dataprops": dataprops, "individuals": individuals,
            "labels": labels, "odomain": odomain, "orange": orange, "ddomain": ddomain,
            "subclass": {k: sorted(v) for k, v in subclass.items()}, "ind_classes": dict(ind_classes),
            "obj_assertions": obj_assertions, "data_assertions": data_assertions, "inverses": inverses,
            "ancestors": ancestors, "primary_nature": primary_nature}


def display_name(onto: dict, ind: str) -> str:
    lab = onto["labels"].get(ind, camel_to_words(ind))
    lab = re.sub(r"\s*\([^)]*\)", "", lab).strip()          # drop parentheticals
    lab = lab.strip('"').strip("\u201c\u201d").strip()       # drop surrounding quotes
    return lab or camel_to_words(ind)


def ner_type(onto: dict, ind: str) -> str:
    """Tipo NER coarse: prima classe in NER_TYPE_ORDER fra le classi dell'individuo
    (o i loro antenati). Ontology-driven per KH2."""
    cls = set(onto["ind_classes"].get(ind, []))
    allc = set(cls)
    for c in cls:
        allc |= onto["ancestors"](c)
    for t in NER_TYPE_ORDER:
        if t in allc:
            return t
    return "Entity"


def build_entity_dictionary(onto: dict) -> dict:
    ed = defaultdict(list)

    def add(short, kind, canonical, extra_surfaces=()):
        entry = {"uri": "#" + short, "short": short, "kind": kind, "canonical_label": canonical}
        surfaces = {canonical.lower(), short.lower(), camel_to_words(short)} | {s.lower() for s in extra_surfaces}
        for s in surfaces:
            s = s.strip()
            # single-letter surfaces (e.g. rank letters) are too noisy for dictionary
            # detection; rank objects are handled by a dedicated context rule instead.
            if len(s) < 2:
                continue
            if s and entry not in ed[s]:
                ed[s].append(entry)
    for c in onto["classes"]:
        add(c, "class", onto["labels"].get(c, camel_to_words(c)))
    for p in onto["objprops"]:
        add(p, "object_property", camel_to_words(p))
    for p in onto["dataprops"]:
        add(p, "datatype_property", camel_to_words(p))
    for i in onto["individuals"]:
        add(i, "individual", onto["labels"].get(i, camel_to_words(i)),
            extra_surfaces=(display_name(onto, i),))
    return dict(ed)


_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\u201c])')


def sentence_split(text: str) -> list[str]:
    out = []
    for para in text.splitlines():
        para = para.strip()
        if not para:
            continue
        for s in _SENT_SPLIT.split(para):
            s = s.strip()
            if s:
                out.append(s)
    return out


def assign_split(key: str) -> str:
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")


def _alias_pattern(alias: str) -> re.Pattern:
    return re.compile(r'(?<!\w)' + re.escape(alias) + r'(?!\w)', re.IGNORECASE)


def detect_mentions(text: str, ed: dict) -> list[dict]:
    """Longest-match, non-overlapping dictionary detection over the entity dictionary.
    Each mention keeps ALL dictionary entries that share its surface form (field
    'entries'), so a later step can pick the one whose type fits the relation it plays
    (e.g. the surface 'Balteus' can be a Character as subject or the AC as object)."""
    spans = []
    for surface, entries in ed.items():
        if len(surface) < 2:
            continue
        for m in _alias_pattern(surface).finditer(text):
            spans.append((m.start(), m.end(), surface, entries))
    spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))
    chosen, occupied = [], []
    for s, e, surface, entries in spans:
        if any(not (e <= os or s >= oe) for os, oe in occupied):
            continue
        occupied.append((s, e))
        primary = entries[0]
        chosen.append({"uri": primary["uri"], "short": primary["short"], "kind": primary["kind"],
                       "canonical_label": primary["canonical_label"], "matched_text": text[s:e],
                       "start": s, "end": e,
                       "entries": [{"uri": x["uri"], "short": x["short"], "kind": x["kind"]} for x in entries]})
    chosen.sort(key=lambda x: x["start"])
    return chosen


def detect_rank_mentions(text: str, onto: dict) -> list[dict]:
    """L'ontologia Kingdom Hearts 2 non usa "ranghi" a lettera singola: nessuna
    rilevazione speciale necessaria. Mantenuta per compatibilita' di firma con gli
    step che la chiamano (02/05/07b)."""
    return []


def _entry_types(onto: dict, short: str, kind: str) -> set:
    if kind == "class":
        return {short} | onto["ancestors"](short)
    out = set(onto["ind_classes"].get(short, []))
    for c in list(out):
        out |= onto["ancestors"](c)
    return out


def mention_entries(mention: dict) -> list:
    """All candidate dictionary entries for a mention (always at least one)."""
    return mention.get("entries") or [{"uri": mention["uri"], "short": mention["short"], "kind": mention["kind"]}]


def compat_role(onto: dict, mention: dict, target_class: str) -> bool:
    """True if ANY entry behind this mention is type-compatible with target_class."""
    return any(target_class in _entry_types(onto, e["short"], e["kind"]) for e in mention_entries(mention))


def ground_role(onto: dict, mention: dict, target_class: str):
    """Resolve the mention to the NamedIndividual whose type fits target_class.
    This is what disambiguates surfaces shared by two individuals (e.g. the character
    'Balteus' vs the Kingdom Hearts 2 'AAP07: BALTEUS' that shares the word 'Balteus')."""
    for e in mention_entries(mention):
        if e["kind"] == "individual" and target_class in _entry_types(onto, e["short"], "individual"):
            return {"uri": e["uri"], "short": e["short"]}
    for e in mention_entries(mention):           # fallback: any individual
        if e["kind"] == "individual":
            return {"uri": e["uri"], "short": e["short"]}
    return None
