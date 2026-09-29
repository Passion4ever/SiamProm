"""Inference helpers behind SiamProm_Colab.ipynb.

Keeps the notebook thin: model registry, input parsing and validation,
batched GPU/CPU prediction and result tables. Reuses predict.py for k-mer
encoding and FASTA reading so results match the command-line tool.
"""

import hashlib
import html
import json
import time
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from predict import encode_sequences, read_fasta
from srcs.model.siamprom import SiamProm

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
EXPECTED_LEN = 81
VALID_BASES = set("ATCG")


# ==================== Model registry ====================

def load_registry(path=WEIGHTS_DIR / "models.json"):
    """Read weights/models.json -> list of {name, file, description, threshold}."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("models.json must be a non-empty list")
    seen = set()
    for e in entries:
        for key in ("name", "file"):
            if key not in e:
                raise ValueError(f"models.json entry is missing '{key}': {e}")
        if e["name"] in seen:
            raise ValueError(f"duplicate model name in models.json: {e['name']}")
        seen.add(e["name"])
        e.setdefault("description", "")
        e.setdefault("threshold", 0.5)
        if not 0 < float(e["threshold"]) < 1:
            raise ValueError(f"threshold for '{e['name']}' must be between 0 and 1")
    return entries


def resolve_checkpoint(entry, weights_dir=WEIGHTS_DIR):
    """Return the local path of a registry entry, downloading it if `file` is a URL."""
    weights_dir = Path(weights_dir)
    f = entry["file"]
    if f.startswith(("http://", "https://")):
        digest = hashlib.sha1(f.encode()).hexdigest()[:8]
        dest = weights_dir / f"{digest}_{Path(f.split('?')[0]).name}"
        if not dest.exists():
            weights_dir.mkdir(parents=True, exist_ok=True)
            part = dest.with_name(dest.name + ".part")
            urllib.request.urlretrieve(f, part)
            part.replace(dest)
        return dest
    path = weights_dir / f
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint for '{entry['name']}' not found: {path}")
    return path


# ==================== Input parsing and validation ====================

def parse_sequences(text):
    """Parse FASTA text, or one bare sequence per line, into (names, seqs)."""
    text = text.lstrip("﻿").replace("\r", "")
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return [], []
    if not any(ln.startswith(">") for ln in lines):
        return (
            [f"seq_{i + 1}" for i in range(len(lines))],
            ["".join(ln.split()) for ln in lines],
        )
    names, seqs, parts = [], [], None
    for ln in lines:
        if ln.startswith(">"):
            if parts is not None:
                seqs.append("".join(parts))
            names.append(ln[1:].strip() or f"seq_{len(names) + 1}")
            parts = []
        else:
            if parts is None:  # sequence text before the first header
                names.append(f"seq_{len(names) + 1}")
                parts = []
            parts.append("".join(ln.split()))
    if parts is not None:
        seqs.append("".join(parts))
    return names, seqs


def validate_sequences(names, seqs):
    """Split inputs into predictable ones and skipped ones with a reason.

    Returns (valid_indices, skipped_df[name, sequence, reason]).
    """
    valid_idx, skipped = [], []
    for i, (name, raw) in enumerate(zip(names, seqs)):
        seq = "".join(raw.split()).upper()
        bad = sorted(set(seq) - VALID_BASES)
        if not seq:
            reason = "empty sequence"
        elif bad:
            reason = f"contains non-ATCG characters: {''.join(bad)}"
        elif len(seq) != EXPECTED_LEN:
            reason = (
                f"length {len(seq)} bp (expected {EXPECTED_LEN} bp; "
                "longer sequences are not truncated or scanned)"
            )
        else:
            valid_idx.append(i)
            continue
        skipped.append({"name": name, "sequence": raw, "reason": reason})
    return valid_idx, pd.DataFrame(skipped, columns=["name", "sequence", "reason"])


# ==================== Model loading ====================

def pick_device():
    """Use the GPU when present; otherwise warn loudly and fall back to CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    warnings.warn(
        "No GPU detected - running on CPU, which is slow for thousands of sequences. "
        "In Colab: Runtime > Change runtime type > T4 GPU, then re-run from the top."
    )
    return torch.device("cpu")


def load_model_safe(path, device, allow_pickle=False):
    """Load a SiamProm checkpoint, refusing arbitrary pickled objects by default."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic != b"PK" and magic[:1] != b"\x80":  # torch zip archive / legacy pickle
        raise RuntimeError(
            f"'{path}' is not a PyTorch checkpoint (a failed or interrupted download, "
            "e.g. an HTML page?). Delete the file and try again."
        )
    try:
        ckpt = torch.load(path, map_location=device, weights_only=not allow_pickle)
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{path}' with weights_only=True ({e}). If you trust this "
            'file, set "allow_pickle": true for it in weights/models.json.'
        ) from e
    model = SiamProm(**ckpt["arch"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ==================== Prediction ====================

@torch.no_grad()
def predict_probs(model, seqs, device, batch_size=1024, progress=None):
    """P(promoter) for already-validated 81 bp sequences, in input order.

    Encodes per batch to keep memory flat; halves the batch on CUDA OOM.
    """
    probs = np.empty(len(seqs), dtype=np.float64)
    i, bs = 0, max(1, int(batch_size))
    while i < len(seqs):
        chunk = seqs[i:i + bs]
        x, _ = encode_sequences(chunk)
        assert x is not None and x.shape[0] == len(chunk), "unvalidated sequence reached the encoder"
        try:
            logits = model.predict(x.to(device))
        except torch.cuda.OutOfMemoryError:
            if bs == 1:
                raise
            torch.cuda.empty_cache()
            bs = max(1, bs // 2)
            continue
        probs[i:i + len(chunk)] = F.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        i += len(chunk)
        if progress:
            progress(i, len(seqs))
    return probs


def build_results(names, seqs, probs, thresholds):
    """Result table. One model: same columns as predict.py. Several: per-model columns + consensus."""
    if len(probs) == 1:
        ((m, p),) = probs.items()
        thr = thresholds.get(m, 0.5)
        return pd.DataFrame({
            "name": names,
            "sequence": seqs,
            "prediction": np.where(p >= thr, "promoter", "non_promoter"),
            "confidence": np.where(p >= thr, p, 1 - p),
        })
    data = {"name": names, "sequence": seqs}
    calls = []
    for m, p in probs.items():
        call = np.where(p >= thresholds.get(m, 0.5), "promoter", "non_promoter")
        data[f"prob_{m}"] = p
        data[f"pred_{m}"] = call
        calls.append(call)
    stacked = np.vstack(calls)
    all_prom = (stacked == "promoter").all(axis=0)
    all_non = (stacked == "non_promoter").all(axis=0)
    data["consensus"] = np.where(all_prom, "promoter", np.where(all_non, "non_promoter", "disagree"))
    return pd.DataFrame(data)


def run_prediction(names, seqs, models, device, batch_size=1024, thresholds=None, progress=None):
    """Validate, predict with every model in `models`, and summarise.

    models: {model_name: loaded model}. progress(done, total) is cumulative over models.
    """
    thresholds = thresholds or {}
    valid_idx, skipped = validate_sequences(names, seqs)
    clean = ["".join(seqs[i].split()).upper() for i in valid_idx]
    valid_names = [names[i] for i in valid_idx]
    total = len(clean) * len(models)

    start = time.perf_counter()
    probs = {}
    for k, (mname, model) in enumerate(models.items()):
        cb = (lambda done, _t, off=k * len(clean): progress(off + done, total)) if progress else None
        probs[mname] = predict_probs(model, clean, device, batch_size, cb)
    seconds = time.perf_counter() - start

    stats = {
        "n_input": len(names),
        "n_valid": len(valid_idx),
        "n_skipped": len(skipped),
        "device": str(device),
        "seconds": seconds,
        "seq_per_sec": len(valid_idx) / seconds if seconds > 0 and valid_idx else 0,
        "promoter_counts": {m: int((p >= thresholds.get(m, 0.5)).sum()) for m, p in probs.items()},
    }
    return {
        "results": build_results(valid_names, clean, probs, thresholds),
        "skipped": skipped,
        "probs": probs,
        "stats": stats,
    }


# ==================== Examples and self-check ====================

def load_example(n=10):
    """First n promoters plus first n randomly generated non-promoters shipped in data/."""
    names, seqs = read_fasta(ROOT / "data" / "7120_cdhit.fasta")
    names, seqs = names[:n], seqs[:n]
    neg_names, neg_seqs = read_fasta(ROOT / "data" / "full_random_data.fasta")
    negatives = [(a, b) for a, b in zip(neg_names, neg_seqs) if "|non_promoter|" in a][:n]
    return names + [a for a, _ in negatives], seqs + [b for _, b in negatives]


def self_check(entry, device):
    """Load `entry`, predict the examples, and compare with predict.py's own code path."""
    import predict

    path = resolve_checkpoint(entry)
    names, seqs = load_example(10)
    probs = predict_probs(load_model_safe(path, device), seqs, device, batch_size=8)
    assert len(probs) == len(seqs) and np.isfinite(probs).all(), "non-finite probabilities"
    assert ((probs >= 0) & (probs <= 1)).all(), "probabilities outside [0, 1]"
    legacy = predict.predict(predict.load_model(path, device), seqs, device)
    np.testing.assert_allclose(probs, legacy["probabilities"], atol=1e-4)
    print(f"Self-check OK ({entry['name']}, {device}): matches predict.py on {len(seqs)} example sequences")
    return True


# ==================== Notebook presentation ====================

_CALL_COLOURS = {
    "promoter": "rgba(42, 127, 98, 0.30)",
    "non_promoter": "rgba(128, 128, 128, 0.18)",
    "disagree": "rgba(214, 158, 46, 0.30)",
}


def summary_html(stats):
    """Row of summary cards (theme-neutral colours, works in Colab light and dark)."""

    def card(label, value, sub=""):
        sub_html = f'<div style="font-size:0.8em;opacity:0.65">{sub}</div>' if sub else ""
        return (
            '<div style="flex:1 1 130px;padding:10px 16px;border-radius:8px;'
            'border:1px solid rgba(128,128,128,0.35);background:rgba(128,128,128,0.08)">'
            f'<div style="font-size:0.8em;opacity:0.7">{label}</div>'
            f'<div style="font-size:1.6em;font-weight:600">{value}</div>{sub_html}</div>'
        )

    n_valid = stats["n_valid"]
    cards = [
        card("Sequences in", stats["n_input"]),
        card("Predicted", n_valid),
        card("Skipped", stats["n_skipped"]),
        card(
            "Speed",
            f"{stats['seq_per_sec']:.0f} seq/s",
            f"{html.escape(str(stats['device']))} \u00b7 {stats['seconds']:.1f} s",
        ),
    ]
    for model, k in stats["promoter_counts"].items():
        frac = k / n_valid if n_valid else 0
        cards.append(card(f"Promoter \u00b7 {html.escape(model)}", f"{frac:.1%}", f"{k} of {n_valid}"))
    return '<div style="display:flex;flex-wrap:wrap;gap:10px;margin:8px 0">' + "".join(cards) + "</div>"


def style_results(df):
    """Result table styled for display: coloured calls, confidence bars, escaped text."""
    styler = df.style.format(escape="html").hide(axis="index")
    call_cols = [c for c in df.columns if c in ("prediction", "consensus") or c.startswith("pred_")]
    if call_cols:
        styler = styler.apply(
            lambda col: [f"background-color: {_CALL_COLOURS.get(v, '')}" for v in col],
            subset=call_cols,
        )
    prob_cols = [c for c in df.columns if c == "confidence" or c.startswith("prob_")]
    for c in prob_cols:
        styler = styler.bar(subset=[c], vmin=0, vmax=1, color="rgba(42, 127, 98, 0.35)")
    if prob_cols:
        styler = styler.format({c: "{:.3f}" for c in prob_cols}, escape="html")
    return styler
