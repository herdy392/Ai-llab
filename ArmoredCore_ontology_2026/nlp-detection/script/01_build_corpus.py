# -*- coding: utf-8 -*-
"""STAGE 0+2 (data generation). From the AC6 ontology produces:
  - generated/ontology_interface.json        (entity dictionary + relation vocab + triggers)
  - generated/synthetic_phase_corpus/...      (schema/TBox synthetic corpus + annotations)
  - generated/ac6_verifier_{train,val,test,all}.jsonl  (relation verifier VALID/INVALID)
  - data-input/ac6_corpus.txt                 (INSTANCE/ABox source text, real entity names)
  - generated/ac6_instance_triples.jsonl      (gold instance triples, for evaluation)
Run:  python script/01_build_corpus.py

Mirrors the professor's generate_synthetic_phase_corpus.py + generate_verifier_jsonl.py.
"""
from __future__ import annotations
import json, re, random
from collections import Counter, defaultdict
from pathlib import Path
import ac6_lib as L

BASE = Path(__file__).resolve().parents[1]          # -> nlp-detection/
OWX = BASE / "ontology" / "armoredcore.owx"
GEN = BASE / "generated"
SYN = GEN / "synthetic_phase_corpus"
DOCS = SYN / "documents"
DATA_IN = BASE / "data-input"
for d in (GEN, SYN, DOCS, DATA_IN):
    d.mkdir(parents=True, exist_ok=True)
RNG = random.Random(20260623)
onto = L.parse_ontology(OWX)

# ---------- schema triples (TBox) ----------
OBJ_T = [("Character","hasAlias","Alias"),("Alias","isAliasOf","Character"),("Character","hasRank","Rank"),
 ("Rank","isRankOf","Character"),("Entity","participatesIn","MajorEvent"),("MajorEvent","involvesEntity","Entity"),
 ("Boss","isBossOf","MajorEvent"),("MajorEvent","hasBoss","Boss"),("MajorEvent","takesPlaceIn","Place"),
 ("Place","hostsEvent","MajorEvent"),("Character","pilots","ArmoredCore"),("ArmoredCore","isPilotedBy","Character"),
 ("Character","kills","Character"),("Character","isKilledBy","Character"),("Character","worksFor","Society"),
 ("Society","employs","Character"),("Entity","isFriendlyTowards","Entity"),("Entity","isHostileToward","Entity")]
DATA_T = [("ArmoredCore","acName","LiteralValue"),("ArmoredCore","isSpecial","LiteralValue"),
 ("Society","name","LiteralValue"),("Society","originatesOnRubicon","LiteralValue"),
 ("MajorEvent","chronologicalOrder","LiteralValue"),("Character","isBuddy","LiteralValue")]
SUB_T = [("Boss","rdfs:subClassOf","Character"),("Character","rdfs:subClassOf","Entity"),
 ("Human","rdfs:subClassOf","Character"),("AI","rdfs:subClassOf","Character"),
 ("Rubiconian","rdfs:subClassOf","Character"),("Society","rdfs:subClassOf","Entity")]
ALL_T = OBJ_T + DATA_T + SUB_T
PHASE1={("Character","hasAlias","Alias"),("Character","hasRank","Rank"),("Character","pilots","ArmoredCore"),
 ("Character","worksFor","Society"),("ArmoredCore","acName","LiteralValue"),("ArmoredCore","isSpecial","LiteralValue"),
 ("Society","name","LiteralValue"),("Society","originatesOnRubicon","LiteralValue"),("Character","isBuddy","LiteralValue")}
PHASE2={("Entity","participatesIn","MajorEvent"),("Boss","isBossOf","MajorEvent"),("MajorEvent","takesPlaceIn","Place"),
 ("Character","kills","Character"),("MajorEvent","chronologicalOrder","LiteralValue")}
PHASE3={("Entity","isFriendlyTowards","Entity"),("Entity","isHostileToward","Entity")}
INV={("Alias","isAliasOf","Character"):("Character","hasAlias","Alias"),("Rank","isRankOf","Character"):("Character","hasRank","Rank"),
 ("MajorEvent","involvesEntity","Entity"):("Entity","participatesIn","MajorEvent"),("MajorEvent","hasBoss","Boss"):("Boss","isBossOf","MajorEvent"),
 ("Place","hostsEvent","MajorEvent"):("MajorEvent","takesPlaceIn","Place"),("ArmoredCore","isPilotedBy","Character"):("Character","pilots","ArmoredCore"),
 ("Character","isKilledBy","Character"):("Character","kills","Character"),("Society","employs","Character"):("Character","worksFor","Society")}
def phases_of(tr):
    if tr in PHASE1: return ["phase1"]
    if tr in PHASE2: return ["phase2"]
    if tr in PHASE3: return ["phase3"]
    if tr in INV: return phases_of(INV[tr])
    return ["phase1"]
TEMPLATES = json.loads(Path(__file__).with_name("_schema_templates.json").read_text(encoding="utf-8"))
TEMPLATES = {tuple(k.split("|")): v for k, v in TEMPLATES.items()}

# ======== ontology_interface.json (+ trigger synonyms for weak detection) ========
TRIGGERS = {
 "isFriendlyTowards":["is friendly towards","is an ally of","fights alongside"],
 "isHostileToward":["is hostile toward","is an enemy of","is opposed to"],
 "hasAlias":["is also known as","goes by the alias","is also called"],
 "hasRank":["holds the rank","is ranked","is rated"],
 "pilots":["pilots","sorties in","goes into battle in"],
 "worksFor":["works for","is affiliated with","serves"],
 "kills":["kills","destroys","eliminates"],
 "isBossOf":["is the boss of","is fought as the boss in","is the boss encountered in"],
 "participatesIn":["takes part in","participates in","is involved in"],
 "takesPlaceIn":["takes place in","unfolds at","is set in"]}
ed = L.build_entity_dictionary(onto)
ed = {k: list(v) for k, v in ed.items()}
for pred, phrases in TRIGGERS.items():
    entry = {"uri": "#" + pred, "short": pred, "kind": "object_property", "canonical_label": L.camel_to_words(pred)}
    for ph in phrases:
        if entry not in ed.setdefault(ph.lower(), []):
            ed[ph.lower()].append(entry)
relation_vocabulary = ([{"uri":"#"+p,"short":p,"label":L.camel_to_words(p),"kind":"object_property"} for p in sorted(onto["objprops"])]
 + [{"uri":"#"+p,"short":p,"label":L.camel_to_words(p),"kind":"datatype_property"} for p in sorted(onto["dataprops"])])
semantic_groups = {"isFriendlyTowards":["isHostileToward"],"participatesIn":["involvesEntity"],
 "isBossOf":["hasBoss"],"hasAlias":["isAliasOf"],"pilots":["isPilotedBy"],"worksFor":["employs"]}
interface = {"ontology":"ontology/armoredcore.owx","entity_dictionary":ed,
 "relation_vocabulary":relation_vocabulary,"semantic_groups":semantic_groups,"triggers":TRIGGERS}
(GEN/"ontology_interface.json").write_text(json.dumps(interface,indent=2,ensure_ascii=False),encoding="utf-8")

# ======== schema synthetic corpus ========
available={tr:set(phases_of(tr)) for tr in ALL_T}
p1=[t for t in ALL_T if t in PHASE1]; p2=[t for t in ALL_T if t in PHASE2]; p3=[t for t in ALL_T if t in PHASE3]
opt=[t for t in ALL_T if t not in PHASE1|PHASE2|PHASE3]
# N_DOCS controlla il volume del corpus schema (e quindi del dataset verifier).
# Portato a 110 per allineare il numero di frasi AC6 a quello attuale di KH2 (~11.8k),
# mantenendo i 4 livelli di complessita' (gestiti da 01b) e zero leakage (split per hash).
N_DOCS = 110
def split_doc(i): return "train" if i < int(0.8*N_DOCS) else ("val" if i < int(0.9*N_DOCS) else "test")
docs=[]; srows=[]
for di in range(N_DOCS):
    did=f"armoredcore_doc_{di+1:03d}"; sp=split_doc(di); chosen=[]
    chosen+=RNG.sample(p1,k=min(5,len(p1))); chosen+=RNG.sample(p2,k=min(2,len(p2))); chosen+=RNG.sample(p3,k=min(1,len(p3)))
    tgt=RNG.randint(9,12); pool=[t for t in opt+p1+p2 if t not in chosen]; RNG.shuffle(pool)
    for t in pool:
        if len(chosen)>=tgt: break
        chosen.append(t)
    RNG.shuffle(chosen); sa=[]; texts=[]
    for si,tr in enumerate(chosen,1):
        txt=TEMPLATES[tr][RNG.randrange(len(TEMPLATES[tr]))]; sid=f"{did}_s{si:03d}"
        trip=[{"subject":tr[0],"predicate":tr[1],"object":tr[2]}]
        sa.append({"sentence_id":sid,"text":txt,"triples":trip,"source_phases":sorted(available[tr])})
        srows.append({"document_id":did,"sentence_id":sid,"split":sp,"text":txt,"triples":trip,"source_phases":sorted(available[tr])})
        texts.append(txt)
    docs.append({"document_id":did,"split":sp,"text":"\n".join(texts),"sentences":sa})
def wj(p,rows):
    with open(p,"w",encoding="utf-8") as h:
        for r in rows: h.write(json.dumps(r,ensure_ascii=False)+"\n")
for d in docs: (DOCS/f"{d['document_id']}.txt").write_text(d["text"],encoding="utf-8")
wj(SYN/"synthetic_documents_all.jsonl",docs); wj(SYN/"synthetic_sentence_annotations_all.jsonl",srows)
for sp in ("train","val","test"):
    wj(SYN/f"synthetic_documents_{sp}.jsonl",[d for d in docs if d["split"]==sp])
    wj(SYN/f"synthetic_sentence_annotations_{sp}.jsonl",[r for r in srows if r["split"]==sp])
(SYN/"synthetic_corpus_stats.json").write_text(json.dumps({"documents":len(docs),"sentences":len(srows),
 "document_split_distribution":{sp:sum(1 for d in docs if d["split"]==sp) for sp in("train","val","test")},
 "sentence_split_distribution":{sp:sum(1 for r in srows if r["split"]==sp) for sp in("train","val","test")},
 "supported_triples":[{"subject":t[0],"predicate":t[1],"object":t[2],"phases":sorted(ph)} for t,ph in sorted(available.items())]},
 indent=2,ensure_ascii=False),encoding="utf-8")

# ======== schema verifier (VALID/INVALID) ========
preds=sorted({t[1] for t in ALL_T})
def ent_uri(n): return L.NS+n
def rel_uri(n): return L.RDFS_NS+"subClassOf" if n=="rdfs:subClassOf" else L.NS+n
def clean(s,p,o): return f"A {L.camel_to_words(s)} {L.relation_phrase(p)} a {L.camel_to_words(o)}."
def entry(sent_pred,cand_pred,s,o,label,neg_type):
    marked=f"A [E1]{L.camel_to_words(s)}[/E1] {L.relation_phrase(sent_pred)} a [E2]{L.camel_to_words(o)}[/E2]."
    return {"text":f"[REL] {cand_pred} [/REL] {marked}","label":label,"candidate_relation":rel_uri(cand_pred),
     "subject":ent_uri(s),"object":ent_uri(o),"tier":"tier_1","neg_type":neg_type,"sentence":clean(s,sent_pred,o)}

# Domini/codomini per generare NEGATIVI DIFFICILI (non piu' relazioni a caso):
#   type_identical    -> stesso (dominio,codominio) della relazione vera => la regola
#                        dominio/codominio NON puo' rifiutarli: il modello deve leggere il testo.
#   partial_overlap   -> condivide dominio O codominio (semi-difficile).
#   type_incompatible -> tipi diversi (facile, rifiutabile dalla regola).
# Generando un negativo per ciascun tipo disponibile otteniamo anche il breakdown per neg_type.
_DR={p:(onto["odomain"].get(p), onto["orange"].get(p)) for p in preds}
def hard_negatives(p):
    dp,rp=_DR.get(p,(None,None))
    identical=[q for q in preds if q!=p and _DR.get(q)==(dp,rp) and dp and rp]
    partial=[q for q in preds if q!=p and _DR.get(q)!=(dp,rp) and (_DR.get(q,(None,None))[0]==dp or _DR.get(q,(None,None))[1]==rp)]
    incompat=[q for q in preds if q!=p and _DR.get(q,(None,None))[0]!=dp and _DR.get(q,(None,None))[1]!=rp]
    negs=[]
    if identical: negs.append((RNG.choice(identical),"type_identical"))
    if partial:   negs.append((RNG.choice(partial),"partial_overlap"))
    if incompat:  negs.append((RNG.choice(incompat),"type_incompatible"))
    if not negs:
        negs.append((RNG.choice([c for c in preds if c!=p]),"type_incompatible"))
    return negs

grp={"train":[],"val":[],"test":[]}; lab=Counter(); negc=Counter()
for r in srows:
    for tr in r["triples"]:
        s,p,o=tr["subject"],tr["predicate"],tr["object"]
        grp[r["split"]].append(entry(p,p,s,o,"VALID","positive")); lab["VALID"]+=1; negc["positive"]+=1
        for w,nt in hard_negatives(p):
            grp[r["split"]].append(entry(p,w,s,o,"INVALID",nt)); lab["INVALID"]+=1; negc[nt]+=1
for sp,it in grp.items(): wj(GEN/f"ac6_verifier_{sp}.jsonl",it)
wj(GEN/"ac6_verifier_all.jsonl",grp["train"]+grp["val"]+grp["test"])
(GEN/"ac6_verifier_stats.json").write_text(json.dumps({"source_sentences":len(srows),
 "verifier_examples":sum(len(v) for v in grp.values()),"split_distribution":{k:len(v) for k,v in grp.items()},
 "label_distribution":dict(lab),"negative_type_distribution":dict(negc),"candidate_relations":preds},indent=2,ensure_ascii=False),encoding="utf-8")

# ======== INSTANCE source text (ABox) ========
INST_TPL = {
 "hasAlias":["{S} is also known as {O}.","{S} goes by the alias {O}."],
 "hasRank":["{S} holds the rank {O}.","{S} is ranked {O} in the Arena."],
 "pilots":["{S} pilots the {O}.","{S} goes into battle in the {O}."],
 "worksFor":["{S} works for {O}.","{S} is affiliated with {O}."],
 "kills":["{S} kills {O}.","{S} destroys {O} in battle."],
 "isBossOf":["{S} is the boss of {O}.","{S} is fought as the boss in {O}."],
 "participatesIn":["{S} takes part in {O}.","{S} is involved in {O}."],
 "takesPlaceIn":["{S} takes place in {O}.","{S} is set in {O}."],
 "isFriendlyTowards":["{S} is friendly towards {O}.","{S} fights alongside {O}."],
 "isHostileToward":["{S} is hostile toward {O}.","{S} is an enemy of {O}."]}
SYMMETRIC={"isFriendlyTowards","isHostileToward"}
inst_triples=[]; seen=set()
for pred,s,o in onto["obj_assertions"]:
    if pred not in INST_TPL: continue
    if pred in SYMMETRIC:
        key=tuple(sorted([s,o]))+(pred,)
        if key in seen: continue
        seen.add(key)
    inst_triples.append((s,pred,o))
# build instance documents (group sentences) with all template variants (augmentation)
sent_records=[]
for s,pred,o in inst_triples:
    sd,od=L.display_name(onto,s),L.display_name(onto,o)
    for tpl in INST_TPL[pred]:
        sent_records.append({"text":tpl.format(S=sd,O=od),"subject":s,"predicate":pred,"object":o})
RNG.shuffle(sent_records)
inst_docs=[]; gold=[]
i=0; doc_i=0
while i<len(sent_records):
    n=RNG.randint(8,12); chunk=sent_records[i:i+n]; i+=n; doc_i+=1
    did=f"ac6_inst_doc_{doc_i:03d}"
    text="\n".join(r["text"] for r in chunk)
    inst_docs.append((did,text))
    for j,r in enumerate(chunk,1):
        gold.append({"document_id":did,"sentence_id":f"{did}_s{j:03d}","text":r["text"],
                     "triple":{"subject":r["subject"],"predicate":r["predicate"],"object":r["object"]}})
corpus_text="\n\n".join(text for _,text in inst_docs)
(DATA_IN/"ac6_corpus.txt").write_text(corpus_text,encoding="utf-8")
wj(GEN/"ac6_instance_triples.jsonl",gold)

print("interface surface forms:",len(ed))
print("schema corpus sentences:",len(srows),"| schema verifier:",sum(len(v) for v in grp.values()))
print("instance triples:",len(inst_triples),"| instance sentences:",len(sent_records),"| instance docs:",len(inst_docs))
