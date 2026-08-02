"""Semantic Routing Embedding Fine-Tuning — Training-Hub Example

Reproduces the vSR (vLLM Semantic Router) POC workflow using training-hub's
embedding_sft algorithm. Fine-tunes all-MiniLM-L6-v2 to classify prompts
into 4 complexity tiers (SIMPLE, MEDIUM, COMPLEX, REASONING) using
contrastive learning.

POC results: 80.39% baseline → 98.53% after fine-tuning with
BatchAllTripletLoss + GROUP_BY_LABEL sampling on <1K synthetic examples.

Usage:
    pip install training-hub[embedding]
    python examples/embedding_sft_semantic_routing.py

Output:
    ./semantic_routing_model/     — fine-tuned sentence-transformers model
    Evaluation metrics printed to stdout
"""
import json
import os
import random

# ── Configuration ────────────────────────────────────────────
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR = "./semantic_routing_model"
EVAL_OUTPUT_DIR = "./semantic_routing_eval"

TIERS = {0: "SIMPLE", 1: "MEDIUM", 2: "COMPLEX", 3: "REASONING"}

# Training config matching the POC's winning combination
TRAIN_CONFIG = {
    "loss_type": "batch_all_triplet",
    "batch_sampler": "group_by_label",
    "num_epochs": 20,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "seed": 42,
}

# ── Seed Anchors (12 per tier, 48 total) ─────────────────────
# These represent the core examples that define each complexity tier.
# In the full POC, these are expanded to ~1K examples via SDG.
SEED_ANCHORS = {
    0: [  # SIMPLE — factual recall, single-step answers
        "What is the capital of France?",
        "How many days are in a year?",
        "What color is the sky?",
        "Who wrote Romeo and Juliet?",
        "What is 15 times 3?",
        "Name the largest planet in our solar system.",
        "What language is spoken in Brazil?",
        "How many continents are there?",
        "What is the chemical symbol for water?",
        "Who painted the Mona Lisa?",
        "What is the boiling point of water in Celsius?",
        "Name the first president of the United States.",
    ],
    1: [  # MEDIUM — multi-step reasoning, comparisons, explanations
        "Explain the difference between TCP and UDP.",
        "How does a binary search algorithm work?",
        "What are the pros and cons of microservices vs monolithic architecture?",
        "Describe how DNS resolution works step by step.",
        "Compare Python and Java for backend development.",
        "Explain the CAP theorem in distributed systems.",
        "How does garbage collection work in Java?",
        "What is the difference between SQL and NoSQL databases?",
        "Explain how HTTPS encrypts data in transit.",
        "Describe the observer design pattern with an example.",
        "How does a load balancer distribute traffic?",
        "What are the SOLID principles in object-oriented programming?",
    ],
    2: [  # COMPLEX — system design, architecture, multi-component
        "Design a real-time chat application that scales to 1 million concurrent users.",
        "Architect a distributed cache with eventual consistency and partition tolerance.",
        "Design a recommendation engine for an e-commerce platform with 10M products.",
        "Build a CI/CD pipeline for a microservices application with 50 services.",
        "Design a multi-region active-active database deployment strategy.",
        "Architect an event-driven system for processing 100K financial transactions per second.",
        "Design a content delivery network from scratch.",
        "Build a fault-tolerant payment processing pipeline with exactly-once semantics.",
        "Design a real-time fraud detection system for credit card transactions.",
        "Architect a data lake that handles both batch and streaming workloads.",
        "Design a search engine that indexes 1 billion web pages.",
        "Build a distributed task scheduler with priority queues and dead letter handling.",
    ],
    3: [  # REASONING — mathematical proofs, theoretical analysis, novel synthesis
        "Prove that the square root of 2 is irrational.",
        "Derive the time complexity of Dijkstra's algorithm from first principles.",
        "Analyze why P vs NP is considered the most important open problem in CS.",
        "Prove that every continuous function on a closed interval is uniformly continuous.",
        "Derive the Euler-Lagrange equation from the principle of least action.",
        "Analyze the computational complexity of quantum error correction codes.",
        "Prove that the halting problem is undecidable using diagonalization.",
        "Compare and contrast category theory and set theory as foundations for mathematics.",
        "Derive Bayes' theorem from the axioms of probability and prove its optimality.",
        "Analyze the relationship between information entropy and thermodynamic entropy.",
        "Prove that no comparison-based sorting algorithm can do better than O(n log n).",
        "Derive the Black-Scholes equation and analyze its assumptions.",
    ],
}

# ── Synthetic Data Generation ────────────────────────────────
# Simplified version of the POC's SDG pipeline. Generates variations
# via paraphrase templates. The full POC uses 6 strategies (paraphrase,
# domain transfer, boundary cases, hard negatives, real-world patterns,
# perturbation) with LLM-generated expansions.

PARAPHRASE_TEMPLATES = [
    "Can you tell me {}",
    "I need to know {}",
    "Please explain {}",
    "Help me understand {}",
    "What can you tell me about {}",
    "I'm curious about {}",
    "Could you clarify {}",
]

DOMAIN_PREFIXES = [
    "In the context of cloud computing, ",
    "From a healthcare perspective, ",
    "For a startup building a SaaS product, ",
    "In the financial services industry, ",
    "For an educational platform, ",
]


def generate_training_data(seed_anchors, target_per_tier=50):
    """Generate synthetic training examples from seed anchors."""
    random.seed(42)
    samples = []

    for label, anchors in seed_anchors.items():
        tier_samples = []

        # Include originals
        for anchor in anchors:
            tier_samples.append({"text": anchor, "label": label})

        # Generate paraphrases
        for anchor in anchors:
            for template in random.sample(PARAPHRASE_TEMPLATES, min(3, len(PARAPHRASE_TEMPLATES))):
                topic = anchor.lower().rstrip("?.!").replace("what is ", "").replace("how does ", "")
                tier_samples.append({"text": template.format(topic), "label": label})

        # Generate domain variations (for MEDIUM and COMPLEX tiers)
        if label in (1, 2):
            for anchor in anchors[:6]:
                for prefix in random.sample(DOMAIN_PREFIXES, 2):
                    tier_samples.append({"text": prefix + anchor.lower(), "label": label})

        # Deduplicate and trim to target
        seen = set()
        unique = []
        for s in tier_samples:
            if s["text"] not in seen:
                seen.add(s["text"])
                unique.append(s)
        samples.extend(unique[:target_per_tier])

    random.shuffle(samples)
    return samples


def split_train_eval(samples, eval_ratio=0.2):
    """Deterministic 80/20 split."""
    random.seed(42)
    indices = list(range(len(samples)))
    random.shuffle(indices)
    split = int(len(indices) * (1 - eval_ratio))
    train = [samples[i] for i in indices[:split]]
    eval_ = [samples[i] for i in indices[split:]]
    return train, eval_


# ── Evaluation ───────────────────────────────────────────────
def evaluate_model(model_path, eval_data, anchors, top_k=3):
    """Evaluate routing accuracy using cosine similarity against anchors.

    Mirrors the vSR router's classification: encode the query and all
    anchors, compute cosine similarity, pick the tier whose top-k
    anchors are most similar.
    """
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_path)

    # Build anchor embeddings per tier
    anchor_texts = []
    anchor_labels = []
    for label, texts in anchors.items():
        for t in texts:
            anchor_texts.append(t)
            anchor_labels.append(label)

    anchor_embeddings = model.encode(anchor_texts, normalize_embeddings=True)
    anchor_labels = np.array(anchor_labels)

    correct = 0
    total = 0
    per_tier = {i: {"correct": 0, "total": 0} for i in TIERS}
    confusion = {i: {j: 0 for j in TIERS} for i in TIERS}

    for sample in eval_data:
        query_embedding = model.encode([sample["text"]], normalize_embeddings=True)
        similarities = np.dot(query_embedding, anchor_embeddings.T)[0]

        # Top-k voting per tier
        tier_scores = {}
        for tier_id in TIERS:
            tier_mask = anchor_labels == tier_id
            tier_sims = similarities[tier_mask]
            tier_scores[tier_id] = np.mean(np.sort(tier_sims)[-top_k:])

        predicted = max(tier_scores, key=tier_scores.get)
        actual = sample["label"]

        confusion[actual][predicted] += 1
        per_tier[actual]["total"] += 1
        total += 1
        if predicted == actual:
            correct += 1
            per_tier[actual]["correct"] += 1

    accuracy = correct / total if total > 0 else 0

    # Compute per-tier F1
    f1_scores = {}
    for tier_id in TIERS:
        tp = confusion[tier_id][tier_id]
        fp = sum(confusion[other][tier_id] for other in TIERS if other != tier_id)
        fn = sum(confusion[tier_id][other] for other in TIERS if other != tier_id)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[tier_id] = f1

    macro_f1 = sum(f1_scores.values()) / len(f1_scores)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_tier_f1": {TIERS[k]: v for k, v in f1_scores.items()},
        "total": total,
        "correct": correct,
        "misrouting_rate": 1 - accuracy,
    }


# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    from training_hub import embedding_sft

    # Step 1: Generate synthetic data
    print("Generating synthetic training data from 48 seed anchors...")
    all_samples = generate_training_data(SEED_ANCHORS, target_per_tier=50)
    train_data, eval_data = split_train_eval(all_samples)
    print(f"  Training:   {len(train_data)} samples")
    print(f"  Evaluation: {len(eval_data)} samples")

    # Write to JSONL
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    eval_path = os.path.join(OUTPUT_DIR, "eval.jsonl")
    for path, data in [(train_path, train_data), (eval_path, eval_data)]:
        with open(path, "w") as f:
            for s in data:
                f.write(json.dumps(s) + "\n")

    # Step 2: Evaluate baseline (pretrained model)
    print(f"\nEvaluating baseline model: {MODEL}")
    baseline = evaluate_model(MODEL, eval_data, SEED_ANCHORS)
    print(f"  Accuracy:        {baseline['accuracy']:.2%}")
    print(f"  Macro F1:        {baseline['macro_f1']:.4f}")
    print(f"  Misrouting rate: {baseline['misrouting_rate']:.2%}")
    for tier, f1 in baseline["per_tier_f1"].items():
        print(f"    {tier:10s} F1: {f1:.4f}")

    # Step 3: Fine-tune with training-hub
    print(f"\nFine-tuning with embedding_sft ({TRAIN_CONFIG['loss_type']})...")
    result = embedding_sft(
        model_path=MODEL,
        data_path=train_path,
        ckpt_output_dir=OUTPUT_DIR,
        eval_data_path=eval_path,
        **TRAIN_CONFIG,
    )
    print(f"  Status: {result['status']}")

    # Step 4: Evaluate fine-tuned model
    print(f"\nEvaluating fine-tuned model: {OUTPUT_DIR}")
    finetuned = evaluate_model(OUTPUT_DIR, eval_data, SEED_ANCHORS)
    print(f"  Accuracy:        {finetuned['accuracy']:.2%}")
    print(f"  Macro F1:        {finetuned['macro_f1']:.4f}")
    print(f"  Misrouting rate: {finetuned['misrouting_rate']:.2%}")
    for tier, f1 in finetuned["per_tier_f1"].items():
        print(f"    {tier:10s} F1: {f1:.4f}")

    # Step 5: Summary
    print(f"\n{'='*60}")
    print(f"  Semantic Routing Fine-Tuning Results")
    print(f"{'='*60}")
    print(f"  {'Metric':<20} {'Pretrained':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print(f"  {'-'*54}")
    print(f"  {'Accuracy':<20} {baseline['accuracy']:>11.2%} {finetuned['accuracy']:>11.2%} {finetuned['accuracy']-baseline['accuracy']:>+9.2%}")
    print(f"  {'Macro F1':<20} {baseline['macro_f1']:>12.4f} {finetuned['macro_f1']:>12.4f} {finetuned['macro_f1']-baseline['macro_f1']:>+10.4f}")
    print(f"  {'Misrouting rate':<20} {baseline['misrouting_rate']:>11.2%} {finetuned['misrouting_rate']:>11.2%} {finetuned['misrouting_rate']-baseline['misrouting_rate']:>+9.2%}")
    print(f"{'='*60}")
    print(f"  Model saved to: {OUTPUT_DIR}")
    print(f"  Load with: SentenceTransformer('{OUTPUT_DIR}')")
