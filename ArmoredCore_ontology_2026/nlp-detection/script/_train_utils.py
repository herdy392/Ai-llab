# -*- coding: utf-8 -*-
"""Utility condivise per il training.

Fornisce:
  - setup_device(): di default prova la GPU (CUDA); se non c'e' usa la CPU.
    Imposta anche il parallelismo intra-op CPU (torch.set_num_threads).
  - resolve_num_workers(): numero di worker DataLoader sicuro per piattaforma
    (data-loading parallelo su Linux; 0 su Windows/macOS per evitare blocchi).
  - parallelize(): su piu' GPU avvolge il modello in DataParallel.
  - BatchTimer: cronometro globale con tempo trascorso dall'inizio + ETA,
    stampati a ogni batch su una riga che si aggiorna.

Torch e' importato in modo lazy: questo modulo resta importabile anche senza
torch (lo usa il benchmark in modalita' degradata).
Importato da: 03_train_verifier.py, 04_train_ner.py, 12_benchmark_models.py.
"""
from __future__ import annotations
import os
import platform
import sys
import time


def format_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def setup_device(prefer_gpu: bool = True):
    """Ritorna (device, info_dict). Default: GPU se disponibile, altrimenti CPU."""
    import torch
    n_threads = max(1, os.cpu_count() or 1)
    try:
        torch.set_num_threads(n_threads)
    except Exception:
        pass
    use_cuda = bool(prefer_gpu and torch.cuda.is_available())
    device = torch.device("cuda" if use_cuda else "cpu")
    n_gpu = torch.cuda.device_count() if use_cuda else 0
    if use_cuda:
        names = ", ".join(torch.cuda.get_device_name(i) for i in range(n_gpu))
        info = f"GPU x{n_gpu} ({names})"
    else:
        info = f"CPU ({n_threads} thread)"
    return device, {"use_cuda": use_cuda, "n_gpu": n_gpu, "n_threads": n_threads,
                    "pin_memory": use_cuda, "info": info}


def resolve_num_workers(requested, n_threads):
    """Worker DataLoader sicuri per piattaforma. Parallelizzazione del data-loading
    abilitata di default solo su Linux (fork); su Windows/macOS default 0 per
    evitare hang/errori con lo start-method 'spawn'."""
    is_linux = platform.system() == "Linux"
    if requested is None:
        return min(4, max(0, n_threads - 1)) if is_linux else 0
    if requested > 0 and not is_linux:
        print("[info] num_workers forzato a 0 (data-loading parallelo sicuro solo su Linux).")
        return 0
    return max(0, requested)


def parallelize(model, dev_info):
    """Su >1 GPU avvolge in DataParallel. Ritorna (model, core): `core` e' il modulo
    reale, da usare per .save_pretrained()/.config. Con DataParallel la loss e' un
    vettore (una per GPU) e va mediata: usare loss_reduce()."""
    import torch
    core = model
    if dev_info.get("n_gpu", 0) > 1:
        model = torch.nn.DataParallel(model)
        print(f"[parallel] DataParallel attivo su {dev_info['n_gpu']} GPU")
    return model, core


def loss_reduce(loss):
    """Media la loss se DataParallel l'ha restituita come vettore."""
    return loss.mean() if hasattr(loss, "dim") and loss.dim() > 0 else loss


class BatchTimer:
    """Cronometro globale: tempo dall'inizio + ETA, stampati a ogni batch."""
    def __init__(self, total_batches: int, label: str = ""):
        self.total = max(1, total_batches)
        self.label = label
        self.done = 0
        self.start = time.perf_counter()

    def step(self, epoch, n_epochs, batch_idx, n_batches, loss):
        self.done += 1
        elapsed = time.perf_counter() - self.start
        eta = elapsed / self.done * (self.total - self.done)
        sys.stdout.write(
            f"\r{self.label} epoch {epoch}/{n_epochs} batch {batch_idx}/{n_batches} | "
            f"loss {loss:.4f} | trascorso {format_time(elapsed)} | ETA {format_time(eta)}   ")
        sys.stdout.flush()

    def end_epoch(self):
        sys.stdout.write("\n")
        sys.stdout.flush()

    def elapsed(self):
        return time.perf_counter() - self.start
