#!/usr/bin/env python3
"""Check semantic cache false positive risk with negation pairs."""
import json
import math
import os
import sys
import argparse
import subprocess

def get_repo_url():
    """Get repository URL from git remote -v."""
    try:
        result = subprocess.run(
            ["git", "remote", "-v"],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        for line in result.stdout.splitlines():
            if "fetch" in line:
                url = line.split()[1]
                if url.startswith("git@"):
                    url = url.replace(":", "/").replace("git@", "https://")
                if url.startswith("https://"):
                    url = url.replace("https://", "")
                if url.endswith(".git"):
                    url = url[:-4]
                return url
    except:
        pass
    return "github.com/KushagraKanaujia/throttle"

def cosine_sim(a, b):
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

def strip_prefix(text):
    """Strip 'user: ' prefix from prompt."""
    if text.startswith("user: "):
        return text[6:]
    return text

def supports_color():
    """Check if terminal supports color."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

def supports_unicode():
    """Check if terminal supports unicode."""
    encoding = sys.stdout.encoding or ''
    return encoding.lower() in ('utf-8', 'utf8')

def main():
    parser = argparse.ArgumentParser(
        description="Check semantic cache false positive risk"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold (default: 0.95)"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show complete table with all pairs and recall figures"
    )
    args = parser.parse_args()

    # Load data from repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    data_path = os.path.join(repo_root, "data", "negation_pairs.json")
    
    with open(data_path) as f:
        data = json.load(f)
    
    pairs = data["pairs"]
    
    # Load sentence-transformers
    try:
        from sentence_transformers import SentenceTransformer
        import sentence_transformers
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        embedder_name = f"sentence-transformers {sentence_transformers.__version__} (all-MiniLM-L6-v2)"
    except ImportError:
        print("Error: sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        return 1
    
    # Compute similarities
    results = []
    for pair in pairs:
        p1 = pair["prompt1"]
        p2 = pair["prompt2"]
        cat = pair["category"]
        
        e1 = model.encode([p1])[0]
        e2 = model.encode([p2])[0]
        sim = cosine_sim(e1, e2)
        
        results.append({
            "p1": p1,
            "p2": p2,
            "cat": cat,
            "sim": sim
        })
    
    # Sort by similarity descending
    results.sort(key=lambda x: x["sim"], reverse=True)
    
    # Find false matches
    false_matches = [
        r for r in results
        if r["cat"] != "paraphrase" and r["sim"] >= args.threshold
    ]
    
    # Count paraphrases
    paraphrases = [r for r in results if r["cat"] == "paraphrase"]
    lost = [r for r in paraphrases if r["sim"] < args.threshold]
    
    # Default output
    if not args.full:
        use_color = supports_color()
        use_unicode = supports_unicode()

        # ANSI color codes
        RED = '\033[91m' if use_color else ''
        DIM = '\033[2m' if use_color else ''
        RESET = '\033[0m' if use_color else ''

        # Box drawing characters
        if use_unicode:
            top_left = '╭'
            top_right = '╮'
            bottom_left = '╰'
            bottom_right = '╯'
            horiz = '─'
            sep_line = '─' * 64
        else:
            top_left = '+'
            top_right = '+'
            bottom_left = '+'
            bottom_right = '+'
            horiz = '-'
            sep_line = '-' * 64

        # Header box
        title = f"  {len(false_matches)} QUESTION PAIRS WOULD GET THE WRONG CACHED ANSWER"
        box_width = 64
        title_padding = box_width - len(title.strip()) - 2

        print(f"  {top_left}{horiz * (box_width - 2)}{top_right}")
        print(f"  │{title}{' ' * title_padding}│" if use_unicode else f"  |{title}{' ' * title_padding}|")
        print(f"  {bottom_left}{horiz * (box_width - 2)}{bottom_right}")
        print()

        # False matches
        for fm in false_matches:
            p1 = strip_prefix(fm["p1"])
            p2 = strip_prefix(fm["p2"])
            score = f"{fm['sim']:.4f}"

            print(f"    {RED}{score}{RESET}   {p1}")
            print(f"             {p2}")
            print(f"             {DIM}→ your cache treats these as the same question{RESET}")
            print()

        # Footer
        print(f"  {sep_line}")
        print(f"  Embedder    {embedder_name}")
        print(f"  Threshold   {args.threshold}")
        print(f"  Pairs       {len(pairs)} measured · run --full for the complete table")
        print()
        print(f"  {get_repo_url()}")
    else:
        # Full output
        print(f"COMPLETE TABLE")
        print()
        print(f"Embedder: {embedder_name}")
        print(f"Threshold: {args.threshold}")
        print()

        print(f"FALSE MATCHES ({len(false_matches)} pairs):")
        print()
        for fm in false_matches:
            p1 = strip_prefix(fm["p1"])
            p2 = strip_prefix(fm["p2"])
            print(f'  "{p1}"')
            print(f'  "{p2}"')
            print(f'  Similarity: {fm["sim"]:.4f} [{fm["cat"]}]')
            print()

        print(f"PARAPHRASE RECALL: {len(paraphrases) - len(lost)}/{len(paraphrases)} ({100*(len(paraphrases) - len(lost))/len(paraphrases):.0f}%)")
        print(f"Lost paraphrases: {len(lost)}")
        print()

        print("ALL PAIRS (sorted by similarity):")
        print()
        for r in results:
            marker = "✓" if r["sim"] >= args.threshold else " "
            cat_label = r["cat"][:10].ljust(10)
            p1 = strip_prefix(r["p1"])
            p2 = strip_prefix(r["p2"])
            p1_short = p1[:30] + "..." if len(p1) > 30 else p1
            p2_short = p2[:30] + "..." if len(p2) > 30 else p2
            print(f"{marker} {r['sim']:.4f} [{cat_label}] {p1_short} / {p2_short}")

        print()
        print(get_repo_url())
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
