"""Pivot embedding demo plus a HellaSwag sanity check with a real model.

This script stitches together two ideas:
1) Pivot embeddings to keep multi-step reasoning centered on a goal.
2) A quick HellaSwag benchmark using a small causal LM so we can confirm a
   non-zero accuracy with an actual model (default: gpt2).

Usage examples:
    python examples/pivot_embeddings_hellaswag.py
    python examples/pivot_embeddings_hellaswag.py --model gpt2 --eval-samples 64

Dependencies:
    pip install datasets transformers torch
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


# -----------------------------
# Core helpers
# -----------------------------

def l2_normalize(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(vector) + eps
    return vector / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-9) -> float:
    a_n = l2_normalize(a, eps)
    b_n = l2_normalize(b, eps)
    return float(np.dot(a_n, b_n))


# -----------------------------
# Embedding backend
# -----------------------------

@dataclass
class EmbeddingBackend:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: Iterable[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        encoding = self.tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**encoding)
        token_embeddings = outputs.last_hidden_state
        attention_mask = encoding.attention_mask.unsqueeze(-1)
        summed = (token_embeddings * attention_mask).sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        mean_pooled = summed / counts
        return l2_normalize(mean_pooled.detach().cpu().numpy())


TASK_TYPES = ["math", "planning", "coding", "analysis", "generic"]
TASK_VECTORS = {task: None for task in TASK_TYPES}
W1, W2, W3 = 0.6, 0.2, 0.2


def _task_vector(task: str, encoder: EmbeddingBackend) -> np.ndarray:
    task = task if task in TASK_VECTORS else "generic"
    cached = TASK_VECTORS[task]
    if cached is None:
        TASK_VECTORS[task] = encoder.encode([f"task type: {task}"])[0]
    return TASK_VECTORS[task]


def compute_pivot_embedding(
    prompt: str,
    history_summary: str = "",
    task_type: str = "generic",
    encoder: EmbeddingBackend | None = None,
) -> np.ndarray:
    encoder = encoder or EmbeddingBackend()
    e_prompt = encoder.encode([prompt])[0]
    e_history = encoder.encode([history_summary])[0] if history_summary else np.zeros_like(e_prompt)
    e_task = _task_vector(task_type, encoder)
    pivot_raw = W1 * e_prompt + W2 * e_history + W3 * e_task
    return l2_normalize(pivot_raw)


def encode_step(step_text: str, encoder: EmbeddingBackend) -> np.ndarray:
    return encoder.encode([step_text])[0]


def realign_step_to_pivot(step_text: str) -> str:
    return f"[REALIGNED TO PIVOT] {step_text}"


def process_step_with_pivot(
    step_text: str,
    pivot: np.ndarray,
    encoder: EmbeddingBackend,
    alignment_threshold: float = 0.65,
    alpha: float = 0.9,
) -> Tuple[str, np.ndarray, float]:
    step_emb = encode_step(step_text, encoder)
    sim = cosine_similarity(step_emb, pivot)

    if sim < alignment_threshold:
        step_text = realign_step_to_pivot(step_text)
        step_emb = encode_step(step_text, encoder)
        sim = cosine_similarity(step_emb, pivot)

    pivot_raw = alpha * pivot + (1.0 - alpha) * step_emb
    return step_text, l2_normalize(pivot_raw), sim


# -----------------------------
# HellaSwag evaluator
# -----------------------------

def score_ending(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    context: str,
    ending: str,
    device: torch.device,
) -> float:
    context_ids = tokenizer(context, return_tensors="pt").to(device)
    full_ids = tokenizer(context + " " + ending, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**full_ids)
        logprobs = torch.log_softmax(outputs.logits, dim=-1)

    prefix_len = context_ids.input_ids.shape[1]
    target_ids = full_ids.input_ids[0, prefix_len:]
    token_logprobs = logprobs[0, prefix_len - 1 : -1, :]
    selected = token_logprobs.gather(1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(selected.sum().cpu())


def evaluate_hellaswag(
    model_name: str = "gpt2",
    max_samples: int = 32,
    device: str = "cpu",
) -> float:
    print(f"🚀 [HellaSwag] Loading model {model_name} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model.eval()

    print(f"📦 [HellaSwag] Fetching first {max_samples} validation samples...")
    dataset = load_dataset("hellaswag", split=f"validation[:{max_samples}]")

    correct = 0
    for example in dataset:
        context = (example["ctx_a"] + " " + example["ctx_b"]).strip()
        endings: Sequence[str] = example["endings"]
        scores = [score_ending(model, tokenizer, context, ending, model.device) for ending in endings]
        prediction = int(np.argmax(scores))
        if prediction == int(example["label"]):
            correct += 1

    accuracy = correct / len(dataset)
    print(f"✅ [HellaSwag] Accuracy: {accuracy:.3f} over {len(dataset)} samples")
    return accuracy


# -----------------------------
# Demo runner
# -----------------------------

def demo_pivot_chain(encoder: EmbeddingBackend) -> None:
    prompt = "Plan a 6-month roadmap to ship a production-grade AutoML platform."
    history = "Discussed STELLA agents, HF datasets, and AutoML leaderboards."
    pivot = compute_pivot_embedding(prompt=prompt, history_summary=history, task_type="planning", encoder=encoder)

    steps: List[str] = [
        "Clarify user outcomes and success metrics for the platform.",
        "Enumerate data connectors, feature store, and orchestration needs.",
        "Start talking about unrelated movie plots.",
        "Draft a phased deployment plan with safety gates.",
    ]

    print("\n=== REASONING CHAIN WITH PIVOT EMBEDDING ===\n")
    for idx, step in enumerate(steps, start=1):
        aligned_step, pivot, similarity = process_step_with_pivot(
            step_text=step,
            pivot=pivot,
            encoder=encoder,
            alignment_threshold=0.65,
            alpha=0.9,
        )
        print(f"After Step {idx}:")
        print(f"  Text      : {aligned_step}")
        print(f"  Alignment : {similarity:.3f}")
        print("-" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pivot embedding demo + HellaSwag check")
    parser.add_argument("--model", default="gpt2", help="Causal LM to score HellaSwag endings")
    parser.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding model for pivot alignment")
    parser.add_argument("--eval-samples", type=int, default=32, help="Number of HellaSwag validation samples")
    parser.add_argument("--device", default="cpu", help="Torch device")
    args = parser.parse_args()

    encoder = EmbeddingBackend(model_name=args.embedder)
    demo_pivot_chain(encoder)
    evaluate_hellaswag(model_name=args.model, max_samples=args.eval_samples, device=args.device)


if __name__ == "__main__":
    main()
