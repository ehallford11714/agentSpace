"""Compute semantic pivot tokens from any open-source model's embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is on the Python path so ``utils`` can be imported
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoTokenizer

from utils.robust_pca import robust_pca


def load_embeddings(model_name: str) -> Tuple[AutoTokenizer, np.ndarray]:
    """Load a tokenizer and embedding matrix for ``model_name``."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    emb_layer = model.get_input_embeddings()
    return tokenizer, emb_layer.weight.detach().cpu().numpy()


def compute_pivots(
    W: np.ndarray,
    tokenizer: AutoTokenizer,
    k: int = 64,
    verbose: bool = False,
) -> Tuple[List[str], List[int], np.ndarray, np.ndarray]:
    """Return pivot tokens and PCA components from robust PCA."""
    L, _ = robust_pca(W, verbose=verbose)

    pca = PCA(n_components=k)
    Lc = L - L.mean(axis=0)
    pca.fit(Lc)
    comps = pca.components_

    norms = np.linalg.norm(L, axis=1)
    pivots: List[str] = []
    pivot_ids: List[int] = []
    for comp in comps:
        sims = (L @ comp) / (norms * np.linalg.norm(comp))
        idx = int(np.argmax(np.abs(sims)))
        pivot_ids.append(idx)
        pivots.append(tokenizer.decode(idx))

    return pivots, pivot_ids, comps, L


def token_to_id(token: str, tokenizer: AutoTokenizer) -> int:
    try:
        return int(token)
    except ValueError:
        return tokenizer.encode(token, add_special_tokens=False)[0]


def project_token(
    token: Union[str, int],
    tokenizer: AutoTokenizer,
    L: np.ndarray,
    comps: np.ndarray,
    mean: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    idx = token_to_id(str(token), tokenizer)
    v = L[idx]
    coords = (v - mean) @ comps.T
    recon = coords @ comps + mean
    return coords, recon, idx


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compute semantic pivots from a given model"
    )
    parser.add_argument("--model", default="gpt2", help="Hugging Face model name")
    parser.add_argument("--k", type=int, default=64, help="number of principal components")
    parser.add_argument("--token", help="token (text or id) to project", default=None)
    parser.add_argument("--analogy", nargs=3, metavar=("A", "B", "C"),
                        help="compute analogy A - B + C")
    parser.add_argument("--plot", action="store_true", help="plot pivot tokens")
    args = parser.parse_args(argv)

    tokenizer, W = load_embeddings(args.model)
    pivots, pivot_ids, comps, L = compute_pivots(W, tokenizer, k=args.k)

    print("Semantic pivots:")
    for i, tok in enumerate(pivots, 1):
        print(f"{i:2d}: {tok}")

    if args.plot and args.k >= 2:
        coords = (L[pivot_ids] - L.mean(0)) @ comps[:2].T
        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], c="tab:blue")
        for t, (x, y) in zip(pivots, coords):
            plt.text(x, y, t, fontsize=8)
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.title("Pivot Tokens")
        plt.tight_layout()
        plt.show()

    if args.token:
        coords, _, idx = project_token(args.token, tokenizer, L, comps, L.mean(0))
        print(f"Coordinates for '{tokenizer.decode(idx)}': {coords}")

    if args.analogy:
        ids = [token_to_id(t, tokenizer) for t in args.analogy]
        mean = L.mean(0)
        coords = (L[ids] - mean) @ comps.T
        result_vec = (coords[0] - coords[1] + coords[2]) @ comps + mean
        norms = np.linalg.norm(L, axis=1)
        sims = (L @ result_vec) / (norms * np.linalg.norm(result_vec))
        best = int(np.argmax(sims))
        print(
            f"{args.analogy[0]} - {args.analogy[1]} + {args.analogy[2]} \u2248 {tokenizer.decode(best)}"
        )


if __name__ == "__main__":
    main()
