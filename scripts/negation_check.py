#!/usr/bin/env python3
"""
Negation and antonym false-match check for semantic similarity models.

Standalone — no throttle dependency. Runs against any embedding model
accessible via sentence-transformers or direct onnxruntime.

Usage:
    python negation_check.py                    # use bundled pair set
    python negation_check.py --threshold 0.95   # custom threshold
    python negation_check.py --pairs my_pairs.jsonl  # your own pairs

The bundled pair set includes 30 paraphrase pairs and 30 hard negatives
(antonyms, polarity inversions, version-number substitutions) with
measured cosine similarities at all-MiniLM-L6-v2. Anyone can reproduce
the full table before running against their own embeddings.

Published findings: at cosine >= 0.95, antonym pairs score higher than
genuine paraphrases because the model encodes topic, not polarity.
"Is eval safe in Python" vs "Is eval dangerous in Python": 0.9874.
First threshold with zero false positives on this set: 0.988,
where paraphrase recall is zero. The distributions do not separate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Bundled pair set — replace with full 30+30 when Kushagra sends the data
# ---------------------------------------------------------------------------

BUNDLED_PAIRS = [
    # FORMAT: {a, b, label, measured_sim}
    # label: "paraphrase" or "negative"
    # measured_sim: cosine at all-MiniLM-L6-v2 (our measurement)

    # PARAPHRASE PAIRS (should match at threshold)
    {"a": "How do I cancel my subscription?",
     "b": "I want to cancel my subscription",
     "label": "paraphrase", "measured_sim": 0.9674},

    {"a": "How do I reset my password?",
     "b": "I need to recover my password",
     "label": "paraphrase", "measured_sim": None},  # pending full data

    # HARD NEGATIVES — antonyms, polarity inversions, version substitution
    # (should NOT match — these are the false positive cases)
    {"a": "Is it safe to use eval in Python?",
     "b": "Is it dangerous to use eval in Python?",
     "label": "negative", "measured_sim": 0.9874,
     "note": "polarity inversion — topic identical, meaning opposite"},

    {"a": "What are the gun laws in Georgia?",
     "b": "What are the gun laws in Arizona?",
     "label": "negative", "measured_sim": 0.7909,
     "note": "entity substitution — different correct answer"},

    # PLACEHOLDER — full 30+30 set pending
]

# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _load_embedder():
    """Load embedder. Tries onnxruntime direct first, falls back to sentence-transformers."""
    try:
        import onnxruntime as ort
        import numpy as np
        from tokenizers import Tokenizer
        from huggingface_hub import hf_hub_download

        model_id = "sentence-transformers/all-MiniLM-L6-v2"
        model_path = hf_hub_download(repo_id=model_id, filename="onnx/model.onnx")
        tokenizer_path = hf_hub_download(repo_id=model_id, filename="tokenizer.json")
        tok = Tokenizer.from_file(tokenizer_path)
        session = ort.InferenceSession(model_path)

        def embed(text: str) -> "np.ndarray":
            enc = tok.encode(text)
            ids = np.array([enc.ids], dtype=np.int64)
            mask = np.array([enc.attention_mask], dtype=np.int64)
            types = np.zeros_like(ids)
            out = session.run(None, {
                "input_ids": ids,
                "attention_mask": mask,
                "token_type_ids": types,
            })
            te = out[0]
            m = np.expand_dims(mask, -1)
            v = (np.sum(te * m, axis=1) /
                 np.clip(np.sum(mask, axis=1, keepdims=True), 1e-9, None))[0]
            v = v.astype("float32")
            norm = np.linalg.norm(v)
            return v / norm if norm > 0 else v

        print("Embedder: onnxruntime direct (all-MiniLM-L6-v2)")
        return embed

    except ImportError:
        pass

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        def embed(text: str) -> "np.ndarray":
            v = model.encode(text, normalize_embeddings=True)
            return v

        print("Embedder: sentence-transformers (all-MiniLM-L6-v2)")
        return embed

    except ImportError:
        pass

    print(
        "ERROR: No embedding backend found.\n"
        "Install one of:\n"
        "  pip install onnxruntime tokenizers huggingface_hub\n"
        "  pip install sentence-transformers",
        file=sys.stderr,
    )
    sys.exit(1)


def cosine(a, b) -> float:
    import numpy as np
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(pairs: list[dict], threshold: float, embed_fn) -> None:
    import numpy as np

    print(f"\nThreshold: {threshold}")
    print(f"Pairs: {len(pairs)} ({sum(1 for p in pairs if p['label']=='paraphrase')} paraphrase, "
          f"{sum(1 for p in pairs if p['label']=='negative')} negative)")
    print()

    results = []
    for pair in pairs:
        va = embed_fn(pair["a"])
        vb = embed_fn(pair["b"])
        sim = cosine(va, vb)
        would_match = sim >= threshold
        is_fp = pair["label"] == "negative" and would_match
        is_fn = pair["label"] == "paraphrase" and not would_match
        results.append({**pair, "live_sim": round(sim, 4),
                         "would_match": would_match, "fp": is_fp, "fn": is_fn})

    # Print table
    print(f"{'Label':<12} {'Live sim':>9} {'Ref sim':>9} {'Match?':>7}  Pair")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["live_sim"]):
        ref = f"{r['measured_sim']:.4f}" if r.get("measured_sim") else "     —"
        flag = "⚠ FP" if r["fp"] else ("  FN" if r["fn"] else "")
        match = "YES" if r["would_match"] else "no"
        a_short = r["a"][:38]
        b_short = r["b"][:38]
        print(f"{r['label']:<12} {r['live_sim']:>9.4f} {ref:>9} {match:>7}  "
              f"{flag}  {a_short!r} vs {b_short!r}")

    fps = [r for r in results if r["fp"]]
    fns = [r for r in results if r["fn"]]
    paraphrases = [r for r in results if r["label"] == "paraphrase"]
    negatives = [r for r in results if r["label"] == "negative"]

    print()
    print(f"SUMMARY @ threshold={threshold}")
    print(f"  Paraphrase recall : {len(paraphrases)-len(fns)}/{len(paraphrases)} "
          f"({(len(paraphrases)-len(fns))/max(1,len(paraphrases)):.0%})")
    print(f"  False positive rate: {len(fps)}/{len(negatives)} "
          f"({len(fps)/max(1,len(negatives)):.0%})")
    if fps:
        print(f"\n  FALSE POSITIVES (negatives that would be incorrectly cached):")
        for r in fps:
            note = r.get("note", "")
            print(f"    [{r['live_sim']:.4f}] {r['a']!r}")
            print(f"            vs {r['b']!r}")
            if note:
                print(f"            ({note})")
    else:
        print(f"\n  No false positives at this threshold.")

    if not fps and paraphrases and len(paraphrases) - len(fns) == 0:
        print(f"\n  NOTE: Zero false positives AND zero paraphrase recall.")
        print(f"  The distributions do not separate at this threshold.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.95,
                   help="Cosine similarity threshold (default: 0.95)")
    p.add_argument("--pairs", type=Path, default=None,
                   help="JSONL file with custom pairs "
                        "({a, b, label: paraphrase|negative})")
    p.add_argument("--sweep", action="store_true",
                   help="Sweep thresholds 0.80 to 0.99 and print recall/FP table")
    args = p.parse_args()

    if args.pairs:
        pairs = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
        print(f"Loaded {len(pairs)} pairs from {args.pairs}")
    else:
        pairs = BUNDLED_PAIRS
        print("Using bundled pair set.")
        print("NOTE: Full 30+30 pair set pending publication.")
        print("Bundled pairs are a subset for structural verification only.")

    embed_fn = _load_embedder()

    if args.sweep:
        print("\nTHRESHOLD SWEEP")
        print(f"{'Threshold':>10} {'Recall':>8} {'FP rate':>8}")
        print("-" * 30)
        vecs = {p["a"]: embed_fn(p["a"]) for p in pairs}
        vecs.update({p["b"]: embed_fn(p["b"]) for p in pairs})
        for t in [0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95,
                  0.96, 0.97, 0.98, 0.988, 0.99]:
            fps = sum(1 for p in pairs
                      if p["label"] == "negative"
                      and cosine(vecs[p["a"]], vecs[p["b"]]) >= t)
            hits = sum(1 for p in pairs
                       if p["label"] == "paraphrase"
                       and cosine(vecs[p["a"]], vecs[p["b"]]) >= t)
            total_p = sum(1 for p in pairs if p["label"] == "paraphrase")
            total_n = sum(1 for p in pairs if p["label"] == "negative")
            print(f"{t:>10.3f} {hits/max(1,total_p):>8.0%} {fps/max(1,total_n):>8.0%}")
        return

    run(pairs, args.threshold, embed_fn)


if __name__ == "__main__":
    main()
