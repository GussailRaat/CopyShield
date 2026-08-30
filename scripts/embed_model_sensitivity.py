"""
Embedding-model sensitivity for the non-literal metric.

Re-flags the non-literal outputs with a SECOND sentence-transformer
(all-mpnet-base-v2) using its own alpha=0.001 calibrated threshold, and
compares method ranking + per-sample agreement against the paper's
all-MiniLM-L6-v2 flags.
"""
import glob
import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as cos_sim

MINILM_THR = json.load(open("outputs/calibration/thresholds.json"))["embedding_sim"]["threshold"]
MPNET_THR = json.load(open("outputs/calibration_mpnet/thresholds.json"))["embedding_sim"]["threshold"]

# (label, glob) — non-literal generation files per method
METHODS = [
    ("SFT baseline", "sft_baseline/nonliteral_outputs.json"),
    ("Contrastive l2", "contrastive/nonliteral_contrastive_lam2p0.json"),
    ("DPO", "dpo/nonliteral_outputs.json"),
    ("Activation", "activation/nonliteral_activation_thr0p5.json"),
]


def sims(model, gens, refs):
    eg = model.encode(gens, batch_size=64, show_progress_bar=False)
    er = model.encode(refs, batch_size=64, show_progress_bar=False)
    return np.array([float(cos_sim([eg[i]], [er[i]])[0][0]) for i in range(len(gens))])


def run(base, tag, model):
    print(f"\n===== {tag} =====")
    print(f"  MiniLM thr={MINILM_THR:.3f}   mpnet thr={MPNET_THR:.3f}")
    print(f"  {'method':16s} {'MiniLM flags':>12s} {'mpnet flags':>12s} {'sim corr r':>11s} {'flag agree%':>11s}")
    rank_minilm, rank_mpnet = [], []
    for name, rel in METHODS:
        g = glob.glob(os.path.join(base, rel))
        if not g:
            print(f"  {name:16s} (missing)")
            continue
        rows = json.load(open(g[0]))
        gens = [r["generated"] for r in rows]
        refs = [r["reference"] for r in rows]
        # MiniLM sims: prefer the stored per-sample values if present, else recompute
        minilm = None
        lk = glob.glob(os.path.join(base, os.path.dirname(rel), "leak_nonliteral*.json"))
        s_mini = sims(MINI, gens, refs)  # recompute for a clean apples-to-apples pairing
        s_mp = sims(model, gens, refs)
        f_mini = int((s_mini > MINILM_THR).sum())
        f_mp = int((s_mp > MPNET_THR).sum())
        r = float(np.corrcoef(s_mini, s_mp)[0, 1])
        agree = float(np.mean((s_mini > MINILM_THR) == (s_mp > MPNET_THR)) * 100)
        rank_minilm.append((name, f_mini))
        rank_mpnet.append((name, f_mp))
        print(f"  {name:16s} {f_mini:>12d} {f_mp:>12d} {r:>11.3f} {agree:>10.1f}%")
    order_mini = [n for n, _ in sorted(rank_minilm, key=lambda x: x[1])]
    order_mp = [n for n, _ in sorted(rank_mpnet, key=lambda x: x[1])]
    print(f"  ranking (fewest->most flags)  MiniLM: {order_mini}")
    print(f"                                 mpnet:  {order_mp}")
    print(f"  ranking identical: {order_mini == order_mp}")


if __name__ == "__main__":
    print("Loading models...")
    MINI = SentenceTransformer("all-MiniLM-L6-v2")
    MP = SentenceTransformer("all-mpnet-base-v2")
    run("outputs/llama3_8b_s42", "Llama-3.1-8B (redo)", MP)
    run("outputs/mistral7b_s42", "Mistral-7B seed 42", MP)
    run("outputs/mistral7b_s7", "Mistral-7B seed 7", MP)
