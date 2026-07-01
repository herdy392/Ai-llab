# -*- coding: utf-8 -*-
"""STAGE 3 (training, LOCAL). Allena il VERIFIER di relazione: special tokens
[REL]/[/REL]/[E1]/[/E1]/[E2]/[/E2], classificazione binaria VALID/INVALID.

Caratteristiche:
  - `train_verifier(...)` riutilizzabile dal benchmark multi-modello (12);
  - encoder (BERT/RoBERTa/DeBERTa/DistilBERT/XLNet/BigBird) e seq2seq (T5);
  - DEVICE: di default usa la GPU (CUDA); se non c'e' passa alla CPU;
  - PARALLELISMO: multi-GPU via DataParallel, CPU via thread intra-op +
    DataLoader workers (sicuri per piattaforma);
  - CRONOMETRO: a ogni batch stampa tempo trascorso dall'inizio + ETA;
  - output SPECIFICI per modello (niente overwrite) + throughput di inferenza.

Esempi:
  python script/03_train_verifier.py --epochs 4 --model roberta-base
  python script/03_train_verifier.py --model t5-base --seq2seq
  python script/03_train_verifier.py --model bert-base-uncased --num-workers 4
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import _train_utils as TU

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"
SPECIAL = ["[REL]", "[/REL]", "[E1]", "[/E1]", "[E2]", "[/E2]"]


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def _slug(model_name):
    return model_name.replace("/", "-")


def _has_trained_model(out_dir: Path, metrics_path: Path) -> bool:
    """True se esiste gia' un training riutilizzabile: best_model salvato + metriche."""
    best = out_dir / "best_model"
    if not (metrics_path.exists() and best.is_dir()):
        return False
    return any(best.glob("*.safetensors")) or (best / "pytorch_model.bin").exists()


def train_verifier(model_name="bert-base-uncased", epochs=4, batch_size=16, lr=2e-5,
                   out_dir=None, metrics_path=None, plots=True, seq2seq=False,
                   num_workers=None, retrain=False):
    """Allena+valuta un verifier. Ritorna il dict delle metriche (con throughput).

    Per default RIUSA un training precedente: se in `out_dir/best_model` c'e' gia'
    un modello e il file metriche esiste, salta il training e ritorna le metriche
    salvate. Passare `retrain=True` per forzare il riaddestramento da zero.
    Lazy import di torch/transformers: il modulo resta importabile senza di essi."""
    slug = _slug(model_name)
    out_dir = Path(out_dir) if out_dir else BASE / "models" / f"kh2_verifier_{slug}"
    metrics_path = Path(metrics_path) if metrics_path else GEN / f"verifier_metrics_{slug}.json"

    # --- RIUSO: se esiste un training precedente e non si forza il retrain -----
    if not retrain and _has_trained_model(out_dir, metrics_path):
        cached = json.loads(metrics_path.read_text(encoding="utf-8"))
        cached["reused"] = True
        print(f"[reuse] {slug}: trovato training precedente in {out_dir / 'best_model'}; "
              f"salto il training (usa --retrain per forzare). "
              f"F1={cached.get('f1', float('nan')):.4f}")
        return cached

    import torch
    from torch.utils.data import Dataset, DataLoader
    from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                              get_linear_schedule_with_warmup)
    from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix

    # --- DEVICE: GPU-first con fallback CPU + parallelismo --------------------
    device, dev = TU.setup_device(prefer_gpu=True)
    nw = TU.resolve_num_workers(num_workers, dev["n_threads"])
    print(f"device: {device.type} | {dev['info']} | data-workers: {nw} | model: {model_name}")

    train = load_jsonl(GEN / "kh2_verifier_train.jsonl")
    val = load_jsonl(GEN / "kh2_verifier_val.jsonl")
    test = load_jsonl(GEN / "kh2_verifier_test.jsonl")
    labels = sorted({r["label"] for r in train + val + test})
    l2i = {l: i for i, l in enumerate(labels)}
    i2l = {i: l for l, i in l2i.items()}
    print("labels:", labels, "| sizes:", len(train), len(val), len(test))

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.add_special_tokens({"additional_special_tokens": SPECIAL})
    lens = [len(tok(r["text"])["input_ids"]) for r in train[:4000]]
    max_len = max(64, min(192, int(np.percentile(lens, 99)) if lens else 96))
    print("max_len:", max_len)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
    model.config.id2label = i2l
    model.config.label2id = l2i
    if getattr(model.config, "pad_token_id", None) is None and tok.pad_token_id is not None:
        model.config.pad_token_id = tok.pad_token_id
    model.resize_token_embeddings(len(tok))
    model.to(device)
    model, core = TU.parallelize(model, dev)   # DataParallel se piu' GPU

    class DS(Dataset):
        def __init__(self, rows):
            self.enc = tok([r["text"] for r in rows], truncation=True, padding="max_length",
                           max_length=max_len, return_tensors="pt")
            self.y = torch.tensor([l2i[r["label"]] for r in rows], dtype=torch.long)
            self.tiers = [r.get("complexity_tier", r.get("tier", "explicit")) for r in rows]
            self.negtypes = [r.get("neg_type", "positive") for r in rows]

        def __len__(self):
            return self.y.shape[0]

        def __getitem__(self, i):
            d = {k: v[i] for k, v in self.enc.items()}
            d["labels"] = self.y[i]
            return d

    tr_ds, va_ds, te_ds = DS(train), DS(val), DS(test)
    pin = dev["pin_memory"]
    tr = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
    va = DataLoader(va_ds, batch_size=64, num_workers=nw, pin_memory=pin)
    te = DataLoader(te_ds, batch_size=64, num_workers=nw, pin_memory=pin)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    total = len(tr) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    def to_dev(b):
        return {k: v.to(device, non_blocking=pin) for k, v in b.items()}

    def evaluate(loader):
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for b in loader:
                b = to_dev(b)
                logits = model(**{k: v for k, v in b.items() if k != "labels"}).logits
                ps += logits.argmax(-1).cpu().tolist(); ys += b["labels"].cpu().tolist()
        acc = float(np.mean([p == y for p, y in zip(ps, ys)])) if ys else 0.0
        return acc, ys, ps

    best = -1.0; (out_dir / "best_model").mkdir(parents=True, exist_ok=True)
    history = []
    timer = TU.BatchTimer(total_batches=len(tr) * epochs, label=f"[{slug}]")
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0
        for bi, b in enumerate(tr, 1):
            b = to_dev(b)
            opt.zero_grad()
            loss = TU.loss_reduce(model(**b).loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
            timer.step(ep, epochs, bi, len(tr), loss.item())   # messaggio per ogni batch
        timer.end_epoch()
        acc, _, _ = evaluate(va)
        history.append({"epoch": ep, "train_loss": tot / max(1, len(tr)), "val_acc": acc})
        print(f"  -> epoch {ep}/{epochs}: train_loss={tot/max(1,len(tr)):.4f}  val_acc={acc:.4f}  "
              f"(trascorso {TU.format_time(timer.elapsed())})")
        if acc >= best:
            best = acc; core.save_pretrained(out_dir / "best_model"); tok.save_pretrained(out_dir / "best_model")

    # --- valutazione finale + THROUGHPUT (esempi/sec) sull'asse 'speed' Slide 13
    import time as _t
    t0 = _t.perf_counter()
    acc, ys, ps = evaluate(te)
    dt = max(1e-6, _t.perf_counter() - t0)
    throughput = len(te_ds) / dt
    print(f"\nTEST accuracy: {acc:.4f} | throughput: {throughput:.1f} esempi/sec "
          f"| training totale: {TU.format_time(timer.elapsed())}")
    print(classification_report(ys, ps, target_names=labels, digits=4))

    pos = l2i.get("VALID", 1)
    P, R, F, _ = precision_recall_fscore_support(ys, ps, labels=[pos], average="binary",
                                                 pos_label=pos, zero_division=0)
    cm = confusion_matrix(ys, ps, labels=list(range(len(labels)))).tolist()

    def _split_metrics(y, p):
        a = float(np.mean([yy == pp for yy, pp in zip(y, p)])) if y else 0.0
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[pos], average="binary",
                                                        pos_label=pos, zero_division=0)
        c = confusion_matrix(y, p, labels=list(range(len(labels)))).tolist()
        return {"accuracy": a, "precision": float(pr), "recall": float(rc),
                "f1": float(f1), "confusion_matrix": c}

    # 3 matrici di confusione + metriche per split (Training/Validation/Test)
    per_split = {"test": _split_metrics(ys, ps)}
    _save_cm(slug, "test", labels, ys, ps)
    for nm, ds_ in [("train", tr_ds), ("val", va_ds)]:
        _a, _y, _p = evaluate(DataLoader(ds_, batch_size=64, num_workers=nw, pin_memory=pin))
        per_split[nm] = _split_metrics(_y, _p)
        _save_cm(slug, nm, labels, _y, _p)

    by_tier = {}
    for y, p, t in zip(ys, ps, te_ds.tiers):
        d = by_tier.setdefault(t, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
        gv, pv = (y == pos), (p == pos)
        d["tp"] += gv and pv; d["fn"] += gv and not pv
        d["fp"] += (not gv) and pv; d["tn"] += (not gv) and not pv
    tier_f1 = {}
    for t, d in by_tier.items():
        pr = d["tp"] / (d["tp"] + d["fp"]) if d["tp"] + d["fp"] else 0.0
        rc = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
        tier_f1[t] = {"precision": pr, "recall": rc,
                      "f1": (2 * pr * rc / (pr + rc)) if pr + rc else 0.0, "n": sum(d.values())}

    # accuratezza nel RICONOSCERE i negativi, per tipo di negativo (Slide: hard negatives)
    by_neg = {}
    for y, p, nt in zip(ys, ps, te_ds.negtypes):
        if y == pos:  # solo negativi
            continue
        d = by_neg.setdefault(nt, {"rejected": 0, "total": 0})
        d["total"] += 1; d["rejected"] += (p != pos)
    neg_acc = {nt: {"reject_accuracy": (d["rejected"] / d["total"] if d["total"] else 0.0),
                    "n": d["total"]} for nt, d in by_neg.items()}

    metrics = {"model": model_name, "slug": slug, "arch": "seq2seq" if seq2seq else "encoder",
               "labels": labels, "best_val_acc": best, "test_accuracy": acc,
               "precision": float(P), "recall": float(R), "f1": float(F),
               "throughput_examples_per_sec": throughput,
               "train_seconds": timer.elapsed(), "device": dev["info"],
               "confusion_matrix": cm, "per_split": per_split,
               "f1_by_complexity_tier": tier_f1, "reject_accuracy_by_neg_type": neg_acc,
               "history": history}
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if plots:
        _save_curve(slug, history)
    print("model saved to:", out_dir / "best_model")
    print("metrics saved to:", metrics_path)
    return metrics


def _save_curve(slug, history):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    plt.figure(figsize=(6, 4), dpi=130)
    xs = [h["epoch"] for h in history]; acc = [h["val_acc"] for h in history]
    plt.plot(xs, acc, marker="o", color="#1e3a8a")
    plt.title(f"Verifier ({slug}) — validation accuracy"); plt.xlabel("epoch"); plt.ylabel("accuracy")
    plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(GEN / f"verifier_training_curve_{slug}.png"); plt.close()


def _save_cm(slug, split, labels, ys, ps):
    """Salva la matrice di confusione di uno split (train/val/test)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
    except ImportError:
        return
    import numpy as _np
    cm = confusion_matrix(ys, ps, labels=list(range(len(labels))))
    plt.figure(figsize=(5, 4), dpi=130); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(labels)), labels); plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.title(f"Verifier ({slug}) — {split} confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(GEN / f"verifier_confusion_matrix_{slug}_{split}.png"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--metrics", default=None)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="worker DataLoader (default: auto per piattaforma)")
    ap.add_argument("--seq2seq", action="store_true", help="modello encoder-decoder (T5)")
    ap.add_argument("--retrain", "--reitrain", dest="retrain", action="store_true",
                    help="forza il riaddestramento anche se esiste gia' un modello (default: riusa)")
    args = ap.parse_args()
    out = args.out or str(BASE / "models" / "kh2_verifier")
    metrics = args.metrics or str(GEN / "verifier_metrics.json")
    train_verifier(args.model, args.epochs, args.batch_size, args.lr, out, metrics,
                   seq2seq=args.seq2seq, num_workers=args.num_workers, retrain=args.retrain)


if __name__ == "__main__":
    main()
