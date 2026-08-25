#!/usr/bin/env python3
"""Test semantic cache thresholds with tricky negation pairs."""
import json, sys, argparse, math

PAIRS = [("What is the capital of France?", "Tell me the capital of France", "paraphrase"), ("How do I install Python?", "What's the process for installing Python?", "paraphrase"), ("Explain async/await in JavaScript", "Can you describe async/await in JavaScript?", "paraphrase"), ("How to reverse a list in Python?", "What's the way to reverse a list in Python?", "paraphrase"), ("Best practices for REST APIs", "What are the best practices for REST APIs?", "paraphrase"), ("Debug memory leak in Node.js", "How can I debug a memory leak in Node.js?", "paraphrase"), ("Optimize SQL query performance", "How do I optimize SQL query performance?", "paraphrase"), ("Set up CI/CD with GitHub Actions", "How to set up CI/CD using GitHub Actions?", "paraphrase"), ("Docker vs Kubernetes differences", "What are the differences between Docker and Kubernetes?", "paraphrase"), ("React hooks best practices", "What are best practices for React hooks?", "paraphrase"), ("Handle errors in async Python", "How should I handle errors in async Python?", "paraphrase"), ("GraphQL vs REST comparison", "Can you compare GraphQL and REST?", "paraphrase"), ("Test React components", "How do I test React components?", "paraphrase"), ("Set up PostgreSQL on Ubuntu", "How to set up PostgreSQL on Ubuntu?", "paraphrase"), ("Improve API response time", "How can I improve my API response time?", "paraphrase"), ("Configure Nginx for production", "How do I configure Nginx for production?", "paraphrase"), ("Implement JWT authentication", "How to implement JWT authentication?", "paraphrase"), ("Profile Python code performance", "How can I profile Python code performance?", "paraphrase"), ("Set up Redis caching", "How do I set up Redis caching?", "paraphrase"), ("Deploy app to AWS", "How to deploy an app to AWS?", "paraphrase"), ("Migrate from MySQL to PostgreSQL", "How do I migrate from MySQL to PostgreSQL?", "paraphrase"), ("Handle CORS in Express", "How should I handle CORS in Express?", "paraphrase"), ("Optimize React rendering", "How can I optimize React rendering?", "paraphrase"), ("Set up monitoring with Prometheus", "How to set up monitoring using Prometheus?", "paraphrase"), ("Implement rate limiting", "How do I implement rate limiting?", "paraphrase"), ("Configure SSL certificates", "How to configure SSL certificates?", "paraphrase"), ("Debug TypeScript compilation errors", "How can I debug TypeScript compilation errors?", "paraphrase"), ("Set up load balancing", "How do I set up load balancing?", "paraphrase"), ("Implement WebSocket connections", "How to implement WebSocket connections?", "paraphrase"), ("Optimize database indexes", "How can I optimize database indexes?", "paraphrase"), ("How do I enable caching in Redis?", "How do I disable caching in Redis?", "antonym"), ("Is it safe to use eval in Python?", "Is it dangerous to use eval in Python?", "antonym"), ("How to increase request timeout?", "How to decrease request timeout?", "antonym"), ("Should I encrypt API keys?", "Should I decrypt API keys?", "antonym"), ("How to compress images?", "How to decompress images?", "antonym"), ("Why is my API slow?", "Why is my API fast?", "antonym"), ("How to enable debug mode?", "How to disable debug mode?", "antonym"), ("Should I cache this query?", "Should I skip caching this query?", "antonym"), ("How to lock database rows?", "How to unlock database rows?", "antonym"), ("When to scale up?", "When to scale down?", "antonym"), ("How to allow CORS?", "How to block CORS?", "antonym"), ("Should I use synchronous calls?", "Should I use asynchronous calls?", "antonym"), ("How to start the server?", "How to stop the server?", "antonym"), ("When to use caching?", "When to avoid caching?", "antonym"), ("How to open database connections?", "How to close database connections?", "antonym"), ("How do I install pandas 1.5?", "How do I install pandas 2.0?", "version"), ("How to configure React Router v5?", "How to configure React Router v6?", "version"), ("Migrate from Python 3.8 to 3.9", "Migrate from Python 3.8 to 3.11", "version"), ("Use OpenSSL 1.1 instead of 1.0", "Use OpenSSL 3.0 instead of 1.0", "version"), ("Set max connections to 100", "Set max connections to 500", "version"), ("Install Node.js 16", "Install Node.js 18", "version"), ("Use TensorFlow 2.x", "Use TensorFlow 1.x", "version"), ("Update to Ubuntu 20.04", "Update to Ubuntu 22.04", "version"), ("How does Docker networking work?", "How much does Docker cost?", "scope"), ("What is Kubernetes?", "Who created Kubernetes?", "scope"), ("How to set up PostgreSQL?", "Why use PostgreSQL?", "scope"), ("Explain REST API design", "When was REST invented?", "scope"), ("How does Redis work?", "Where is Redis used?", "scope"), ("How to configure Nginx?", "Who maintains Nginx?", "scope"), ("What is GraphQL?", "Where is GraphQL used?", "scope")]

def get_embedder():
    try:
        from sentence_transformers import SentenceTransformer
        return ("sentence-transformers", SentenceTransformer('all-MiniLM-L6-v2').encode)
    except: pass
    try:
        import openai, os
        if os.getenv("OPENAI_API_KEY"):
            client = openai.OpenAI()
            return ("openai", lambda texts: [e.embedding for e in client.embeddings.create(input=texts, model="text-embedding-3-small").data])
    except: pass
    try:
        from transformers import AutoTokenizer
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        model = ORTModelForFeatureExtraction.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        return ("onnx", lambda texts: model(**tokenizer(texts, padding=True, truncation=True, return_tensors="pt")).last_hidden_state[:, 0, :].detach().numpy())
    except: pass
    sys.exit("No embedding library found. Install: pip install sentence-transformers")

def cosine_sim(a, b):
    dot, mag_a, mag_b = sum(x*y for x,y in zip(a,b)), math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0

def main():
    parser = argparse.ArgumentParser(description="Test semantic cache threshold with negation pairs")
    parser.add_argument("--pairs-file", help="JSON with {pairs:[{prompt1,prompt2,category}]}")
    parser.add_argument("--threshold", type=float, default=0.95)
    args = parser.parse_args()

    pairs = PAIRS
    if args.pairs_file:
        with open(args.pairs_file) as f:
            pairs = [(p["prompt1"], p["prompt2"], p["category"]) for p in json.load(f)["pairs"]]

    embedder_name, embed = get_embedder()
    print(f"Using {embedder_name}\nThreshold: {args.threshold}\n")

    results = []
    for p1, p2, cat in pairs:
        e1, e2 = embed([p1, p2]) if embedder_name == "openai" else (embed([p1])[0], embed([p2])[0])
        results.append((cosine_sim(e1, e2), p1, p2, cat))
    results.sort(reverse=True)

    false_matches, lost_paraphrases = [], []
    for sim, p1, p2, cat in results:
        marker = "✓ MATCH" if sim >= args.threshold else "  "
        print(f"{marker} {sim:.4f} [{cat:10}] {p1[:40]}... / {p2[:40]}...")
        if cat != "paraphrase" and sim >= args.threshold: false_matches.append((sim, p1, p2, cat))
        if cat == "paraphrase" and sim < args.threshold: lost_paraphrases.append((sim, p1, p2))

    paraphrase_count = sum(1 for _,_,_,cat in results if cat == "paraphrase")
    print(f"\nFalse matches: {len(false_matches)}")
    for sim, p1, p2, cat in false_matches: print(f"  {sim:.4f} [{cat}] {p1} / {p2}")
    print(f"\nLost paraphrases: {len(lost_paraphrases)}/{paraphrase_count} ({len(lost_paraphrases)/paraphrase_count*100:.1f}%)")
    print("\nhttps://github.com/anthropics/throttle - measures semantic cache false positive risk")

if __name__ == "__main__": main()
