# Embedding SFT — Contrastive Embedding Fine-Tuning

Embedding SFT is a **contrastive** fine-tuning algorithm for sentence embedding models (e.g. `all-MiniLM-L6-v2`). Instead of training a classifier head, it reshapes the embedding space so that inputs sharing the same label end up close together and different labels end up far apart. At inference time you classify by **nearest-anchor cosine similarity** — no task-specific head is shipped with the model.

> The canonical use case is **semantic routing**: classify a natural-language query into one of N lanes (e.g. route a request to the right specialist model, or pick the right complexity tier). The same mechanism improves any task that benefits from tighter same-label clusters — retrieval, deduplication, clustering.

## When to Use

- You want to **route** queries to one of N specialist models/agents by intent
- You need a lightweight classifier (23M-param `all-MiniLM-L6-v2` beats running a full LLM for routing)
- You have labeled text data and want the model to separate the classes in embedding space
- You want to improve retrieval/clustering quality for your domain vocabulary

For weight-training on generative models, see [SFT](/algorithms/sft), [OSFT](/algorithms/osft), [LoRA + SFT](/algorithms/lora), or [GRPO](/algorithms/grpo). For prompt optimization without touching weights, see [GEPA](/algorithms/gepa).

## Quick Start

```python
from training_hub import embedding_sft

result = embedding_sft(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    data_path="routing_train.jsonl",    # one {"text": "...", "label": 0} per line
    ckpt_output_dir="./routing_model",
    loss_type="batch_all_triplet",
    num_epochs=20,
    batch_size=32,
    learning_rate=2e-5,
)
# ./routing_model is a standard sentence-transformers model:
#   SentenceTransformer("./routing_model")
```

## How It Works

1. **Load** a sentence-transformers model (`model_path` — a HuggingFace ID or local path).
2. **Sample batches** with `GROUP_BY_LABEL` so every batch contains examples from every class — this is required for triplet mining (an anchor needs positives and negatives in the same batch).
3. **Compute a contrastive loss** that pulls same-label embeddings together and pushes different-label embeddings apart.
4. **Save** the fine-tuned model in standard sentence-transformers format, ready for `SentenceTransformer("<output_dir>")`.

At inference, classification is done by encoding the query and a set of labeled **anchor** texts, then picking the class whose top-k anchors have the highest mean cosine similarity to the query. See the [example notebook](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/routing_demo.ipynb) for a reusable router runtime.

## Losses

| `loss_type` | Loss | Description | Best for |
|-------------|------|-------------|----------|
| `batch_all_triplet` (default) | `BatchAllTripletLoss` | Mines all valid triplets per batch | Default — exhaustive boundary learning |
| `batch_hard_triplet` | `BatchHardTripletLoss` | Mines the hardest triplet per anchor | Focused on worst-case boundaries |
| `mnrl` | `MultipleNegativesRankingLoss` | Ranking loss over in-batch negatives | General semantic similarity |

For triplet losses, the dataset is consumed as `{"text", "label"}` rows. For `mnrl`, label datasets are **auto-converted** to `(anchor, positive)` pairs (all within-label combinations, capped by `max_pairs_per_label`), and the batch sampler defaults to `no_duplicates`. Pass a custom `loss_fn` to override any of these.

## Batch Sampling

| `batch_sampler` | Description |
|-----------------|-------------|
| `group_by_label` (default for triplets) | Every batch contains all classes — required for triplet mining |
| `no_duplicates` (default for MNRL) | No duplicate texts in a batch |
| `default` | Standard batching |

Using a triplet loss without `group_by_label` is allowed but logs a warning — batches may lack some classes, which degrades triplet mining.

## Data Format

Training data is a JSONL file (or a HuggingFace dataset ID) where each row has a text field and an integer label field:

```json
{"text": "What is the cabin pressure trend?", "label": 0}
{"text": "How much propellant remains?", "label": 1}
```

Column names default to `text` and `label`; override with `text_column` / `label_column`. CSV inputs are also accepted.

## Custom Loss

Pass any sentence-transformers-compatible loss as `loss_fn` to override `loss_type`:

```python
from sentence_transformers import losses

result = embedding_sft(
    model_path="sentence-transformers/all-MiniLM-L6-v2",
    data_path="routing_train.jsonl",
    ckpt_output_dir="./routing_model",
    loss_fn=losses.ContrastiveLoss(model),  # overrides loss_type
)
```

## Output

- The fine-tuned model is saved to `ckpt_output_dir` in standard sentence-transformers format.
- `training_metrics.jsonl` is appended with the final step/loss/epoch.

The call returns a dict: `{"status", "model_path", "num_samples", "num_epochs", "loss_type"}`.

## Inference (Routing)

The algorithm produces an **embedding model**, not a classifier. Classification is a thin layer on top — encode the query and a set of labeled anchor texts, then take the class with the highest top-k mean cosine similarity. A confidence threshold (τ) drops low-confidence queries to a fallback class rather than force-routing them. The [routing demo notebook](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/routing_demo.ipynb) includes a ready-to-use `Router` runtime.

## Tips

- **Small models are enough.** `all-MiniLM-L6-v2` (23M params) trains in seconds on a single GPU and runs on CPU. Use it as the default.
- **`group_by_label` is mandatory for triplets.** Without it, batches may miss classes and triplet mining produces no signal.
- **Hard negatives matter more than data volume.** Cross-class pairs that share vocabulary (e.g. *cabin pressure* vs *chamber pressure*) are what teach the model to use context. Mine these deliberately.
- **Pretrained embeddings may already separate easy classes.** If your classes are vocabulary-distinct, the baseline will be high and gains will be small. The algorithm earns its keep when classes share vocabulary and the pretrained model confuses them.
- **Use a held-out eval split that is NOT paraphrased from training.** Paraphrase-templated eval sets overestimate accuracy because the model recognizes the template. Hold out entire seeds for a honest generalization estimate.

## API Reference

See [`embedding_sft()`](/api/functions/embedding_sft) for the full parameter reference and [`EmbeddingSFTAlgorithm`](/api/classes/EmbeddingSFTAlgorithm) for the class-based API.

## Example Notebook

- [Semantic Routing Demo](https://github.com/Red-Hat-AI-Innovation-Team/training_hub/blob/main/examples/notebooks/routing_demo.ipynb) — runnable notebook training a 4-class router end-to-end (data generation → fine-tuning → router runtime → evaluation), including a confidence-threshold fallback.
