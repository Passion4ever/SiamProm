#!/usr/bin/env python3
"""
SiamProm Inference Script — Predict whether input sequences are cyanobacterial promoters.

Usage:
    python predict.py --checkpoint path/to/ckpt.pth --fasta seqs.fasta --output results.csv
    python predict.py --checkpoint path/to/ckpt.pth --fasta seqs.fasta --output results.csv --device 1
"""

import argparse
import sys
import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from srcs.model.siamprom import SiamProm


# ==================== Sequence Encoding ====================

def encode_sequences(sequences, k=3, max_len=81):
    """Encode DNA sequences into k-mer integer tensors."""
    alphabet = list("ATCG")
    total_kmer = reduce(
        lambda x, y: [i + j for i in x for j in y], [alphabet] * k
    )
    kmer_map = dict(zip(total_kmer, range(len(total_kmer))))

    kmer_list = []
    valid_indices = []
    for i, seq in enumerate(sequences):
        seq = seq.upper().strip()
        if max_len is not None and len(seq) > max_len:
            seq = seq[:max_len]
        try:
            integer = [kmer_map[seq[j:j+k]] for j in range(len(seq) - k + 1)]
            kmer_list.append(integer)
            valid_indices.append(i)
        except KeyError:
            # 找到第一个非法 k-mer 的位置
            bad_chars = set(seq) - set("ATCG")
            warnings.warn(
                f"Skipped sequence {i}: contains non-ATCG characters {bad_chars}"
            )

    if not kmer_list:
        return None, valid_indices

    return torch.tensor(kmer_list), valid_indices


def read_fasta(fasta_path):
    """Read a FASTA file and return (names, sequences)."""
    names, sequences = [], []
    header = None
    seq_parts = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    names.append(header)
                    sequences.append("".join(seq_parts))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)

    if header is not None:
        names.append(header)
        sequences.append("".join(seq_parts))

    return names, sequences


# ==================== Model Loading ====================

def load_model(checkpoint_path, device='cpu'):
    """Load a SiamProm model from checkpoint.

    Checkpoint 格式: {'arch': {...模型架构参数}, 'model_state_dict': state_dict}
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = SiamProm(**ckpt['arch'])
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    return model


# ==================== Inference ====================

@torch.no_grad()
def predict(model, sequences, device='cpu', batch_size=256):
    """Run promoter prediction on a list of DNA sequences."""
    encoded, valid_indices = encode_sequences(sequences)

    if encoded is None:
        return {'probabilities': [], 'valid_indices': []}

    encoded = encoded.to(device)
    all_probs = []

    for i in range(0, len(encoded), batch_size):
        batch = encoded[i:i+batch_size]
        logits = model.predict(batch)
        probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)

    return {
        'probabilities': all_probs[:, 1].tolist(),
        'valid_indices': valid_indices,
    }


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='SiamProm: Cyanobacterial Promoter Prediction')
    parser.add_argument('--fasta', type=str, required=True, help='Input FASTA file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--device', type=int, default=0, help='GPU device index (default: 0)')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size (default: 256)')
    parser.add_argument('--threshold', type=float, default=0.5, help='Classification threshold (default: 0.5)')

    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')

    print(f"Loading model: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    names, sequences = read_fasta(args.fasta)
    print(f"Loaded {len(sequences)} sequences")

    results = predict(model, sequences, device, args.batch_size)

    valid_idx = results['valid_indices']
    n_valid = len(valid_idx)
    n_skipped = len(sequences) - n_valid

    df = pd.DataFrame({
        'name': [names[i] for i in valid_idx],
        'sequence': [sequences[i] for i in valid_idx],
        'prediction': ['promoter' if p >= args.threshold else 'non_promoter'
                       for p in results['probabilities']],
        'confidence': [p if p >= args.threshold else 1 - p
                       for p in results['probabilities']],
    })

    df.to_csv(args.output, index=False)

    n_promoter = (df['prediction'] == 'promoter').sum()
    n_non_promoter = n_valid - n_promoter

    print(f"\nResults:")
    print(f"  Valid sequences: {n_valid}/{len(sequences)}", end="")
    if n_skipped > 0:
        print(f" ({n_skipped} skipped due to non-ATCG characters)")
    else:
        print()
    print(f"  Promoter:     {n_promoter} ({n_promoter/max(1,n_valid)*100:.1f}%)")
    print(f"  Non-promoter: {n_non_promoter} ({n_non_promoter/max(1,n_valid)*100:.1f}%)")
    print(f"  Mean confidence: {df['confidence'].mean():.4f}")
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
