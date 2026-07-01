# -*- coding: utf-8 -*-
"""STAGE 1b (training, LOCAL). Trains a NER / mention detector (token classification,
BIO) on generated/kh2_ner_{train,val,test}.jsonl. Recognises Kingdom Hearts 2 entity
types (Character, Enemy, Worlds, Keyblade, Items, Action, Fight, Party, etc.).

Example (Windows):
  python script\\04_train_ner.py --epochs 5 --batch-size 16 --model bert-base-cased

Saves, besides the model:
  generated/ner_metrics.json
  generated/ner_training_curve.png
  generated/ner_confusion_matrix.png
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import _train_utils as TU

BASE = Path(__file__).resolve().parents[1]
GEN = BASE / "generated"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


class NerDS(Dataset):
    def __init__(self, rows, tok, l2i, max_len):
        self.items = []
        for r in rows:
            enc = tok(r["tokens"], is_split_into_words=True, truncation=True, padding="max_length",
                      max_length=max_len, return_tensors="pt")
            word_ids = enc.word_ids(0); labels = []
            prev = None
            for wid in word_ids:
                if wid is None:
                    labels.append(-100)
                elif wid != prev:
                    labels.append(l2i[r["ner_tags"][wid]])
                else:
                    labels.append(-100)
                prev = wid
            self.items.append({**{k: v[0] for k, v in enc.items()}, "labels": torch.tensor(labels)})
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        return self.items[i]


def save_plots(history, label_names, ys, ps):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots."); return
    plt.figure(figsize=(6, 4), dpi=130)
    xs = [h["epoch"] for h in history]; acc = [h["val_token_acc"] for h in history]
    plt.plot(xs, acc, marker="o", color="#0f766e")
    plt.title("NER — validation token accuracy"); plt.xlabel("epoch"); plt.ylabel("token accuracy")
    plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(GEN / "ner_training_curve.png"); plt.close()
    present = sorted(set(ys + ps))
    names = [label_names[i] for i in present]
    cm = confusion_matrix(ys, ps, labels=present)
    plt.figure(figsize=(max(5, len(present) * 0.7), max(4, len(present) * 0.6)), dpi=130)
    plt.imshow(cm, cmap="Blues")
    plt.xticks(range(len(present)), names, rotation=90); plt.yticks(range(len(present)), names)
    plt.title("NER — test confusion matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout(); plt.savefig(GEN / "ner_confusion_matrix.png"); plt.close()
    print("saved plots -> generated/ner_training_curve.png, generated/ner_confusion_matrix.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="bert-base-cased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--out", default=str(BASE / "models" / "kh2_ner"))
    ap.add_argument("--metrics", default=str(GEN / "ner_metrics.json"),
                    help="path del JSON di metriche (per evitare overwrite tra modelli)")
    ap.add_argument("--num-workers", type=int, default=None,
                    help="worker DataLoader (default: auto per piattaforma)")
    ap.add_argument("--retrain", "--reitrain", dest="retrain", action="store_true",
                    help="forza il riaddestramento anche se esiste gia' un NER allenato (default: riusa)")
    args = ap.parse_args()

    # RIUSO: se esiste gia' un NER allenato + metriche e non si forza il retrain, salta.
    out_best = Path(args.out) / "best_model"
    metrics_p = Path(args.metrics)
    has_model = out_best.is_dir() and (any(out_best.glob("*.safetensors"))
                                       or (out_best / "pytorch_model.bin").exists())
    if not args.retrain and has_model and metrics_p.exists():
        print(f"[reuse] NER: trovato training precedente in {out_best}; salto "
              f"(usa --retrain per forzare).")
        return

    device, dev = TU.setup_device(prefer_gpu=True)
    nw = TU.resolve_num_workers(args.num_workers, dev["n_threads"])
    pin = dev["pin_memory"]
    train = load_jsonl(GEN / "kh2_ner_train.jsonl"); val = load_jsonl(GEN / "kh2_ner_val.jsonl"); test = load_jsonl(GEN / "kh2_ner_test.jsonl")
    labels = sorted({t for r in train + val + test for t in r["ner_tags"]})
    if "O" in labels:
        labels.remove("O"); labels = ["O"] + labels
    l2i = {l: i for i, l in enumerate(labels)}; i2l = {i: l for l, i in l2i.items()}
    print(f"device: {device.type} | {dev['info']} | data-workers: {nw} | labels:", labels,
          "| sizes:", len(train), len(val), len(test))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model, num_labels=len(labels), id2label=i2l, label2id=l2i).to(device)
    model, core = TU.parallelize(model, dev)
    max_len = 64
    tr = DataLoader(NerDS(train, tok, l2i, max_len), batch_size=args.batch_size, shuffle=True, num_workers=nw, pin_memory=pin)
    va = DataLoader(NerDS(val, tok, l2i, max_len), batch_size=32, num_workers=nw, pin_memory=pin)
    te = DataLoader(NerDS(test, tok, l2i, max_len), batch_size=32, num_workers=nw, pin_memory=pin)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total = len(tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * total), total)

    def evaluate(loader):
        model.eval(); ys, ps = [], []
        with torch.no_grad():
            for b in loader:
                b = {k: v.to(device, non_blocking=pin) for k, v in b.items()}
                logits = model(**{k: v for k, v in b.items() if k != "labels"}).logits
                pred = logits.argmax(-1)
                for i in range(b["labels"].shape[0]):
                    for j in range(b["labels"].shape[1]):
                        if b["labels"][i, j].item() != -100:
                            ys.append(b["labels"][i, j].item()); ps.append(pred[i, j].item())
        return ys, ps

    best = -1.0; out = Path(args.out); (out / "best_model").mkdir(parents=True, exist_ok=True)
    history = []
    timer = TU.BatchTimer(total_batches=len(tr) * args.epochs, label="[ner]")
    for ep in range(1, args.epochs + 1):
        model.train(); tot = 0.0
        for bi, b in enumerate(tr, 1):
            b = {k: v.to(device, non_blocking=pin) for k, v in b.items()}
            opt.zero_grad(); loss = TU.loss_reduce(model(**b).loss); loss.backward()
            opt.step(); sched.step(); tot += loss.item()
            timer.step(ep, args.epochs, bi, len(tr), loss.item())   # messaggio per ogni batch
        timer.end_epoch()
        ys, ps = evaluate(va); acc = float(np.mean([y == p for y, p in zip(ys, ps)])) if ys else 0.0
        history.append({"epoch": ep, "train_loss": tot / max(1, len(tr)), "val_token_acc": acc})
        print(f"  -> epoch {ep}/{args.epochs}: train_loss={tot/max(1,len(tr)):.4f}  val_token_acc={acc:.4f}  "
              f"(trascorso {TU.format_time(timer.elapsed())})")
        if acc >= best:
            best = acc; core.save_pretrained(out / "best_model"); tok.save_pretrained(out / "best_model")
    ys, ps = evaluate(te)
    present = sorted(set(ys + ps)); names = [i2l[i] for i in present]
    print("\nTEST token-level report:")
    print(classification_report(ys, ps, labels=present, target_names=names, digits=4, zero_division=0))

    metrics = {"model": args.model, "labels": labels, "best_val_token_acc": best, "history": history}
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_plots(history, i2l, ys, ps)
    print("best model saved to:", out / "best_model")


if __name__ == "__main__":
    main()
