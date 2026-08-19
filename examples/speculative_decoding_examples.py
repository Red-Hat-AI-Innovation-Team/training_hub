"""Speculative decoding training examples — all configurations.

Each example trains an Eagle3 draft model for Qwen3-8B using the sharegpt dataset.
The difference is how/when hidden states are generated, who manages vLLM,
and how many GPUs are used for training.

GPU layout used in examples:
  - GPUs 0,1: vLLM hidden state extraction (data-parallel)
  - GPUs 2,3: Draft model training (FSDP)
"""

from training_hub import train_speculator


# ============================================================================
# 1. Offline + Managed vLLM (fully automated)
#    Backend handles everything: data prep, launch vLLM, bulk-generate hidden
#    states, kill vLLM, train from disk. No manual steps.
# ============================================================================

result = train_speculator(
    verifier_name_or_path="Qwen/Qwen3-8B",
    ckpt_output_dir="./eagle3_output/checkpoints",
    data_path="sharegpt",
    mode="offline",
    speculator_type="eagle3",
    vllm_gpu_ids=[0, 1],       # managed vLLM, data-parallel on 2 GPUs
    training_gpu_ids=[2, 3],   # FSDP training on 2 GPUs
    epochs=3,
    draft_vocab_size=32000,
    max_samples=5000,
)


# ============================================================================
# 2. Online + Managed vLLM
#    Backend launches vLLM on specified GPUs, generates hidden states per-batch
#    during training, then kills vLLM when done.
# ============================================================================

result = train_speculator(
    verifier_name_or_path="Qwen/Qwen3-8B",
    ckpt_output_dir="./eagle3_output/checkpoints",
    data_path="sharegpt",
    mode="online",
    speculator_type="eagle3",
    vllm_gpu_ids=[0, 1],
    training_gpu_ids=[2, 3],
    epochs=3,
    draft_vocab_size=32000,
    max_samples=5000,
)


# ============================================================================
# 3. Offline + User Endpoint
#    You launch vLLM separately. Backend bulk-generates all hidden states to
#    disk, then trains from disk.
#
#    Launch vLLM first:
#      CUDA_VISIBLE_DEVICES=0,1 python speculators/scripts/launch_vllm.py \
#          Qwen/Qwen3-8B -- --data-parallel-size 2 --port 8234
# ============================================================================

result = train_speculator(
    verifier_name_or_path="Qwen/Qwen3-8B",
    ckpt_output_dir="./eagle3_output/checkpoints",
    data_path="sharegpt",
    mode="offline",
    speculator_type="eagle3",
    vllm_endpoint="http://localhost:8234/v1",
    training_gpu_ids=[2, 3],
    epochs=3,
    draft_vocab_size=32000,
    max_samples=5000,
)


# ============================================================================
# 4. Online + User Endpoint
#    You launch vLLM separately, backend generates hidden states per-batch
#    during training. vLLM stays alive throughout.
#
#    Launch vLLM first:
#      CUDA_VISIBLE_DEVICES=0,1 python speculators/scripts/launch_vllm.py \
#          Qwen/Qwen3-8B -- --data-parallel-size 2 --port 8234
# ============================================================================

result = train_speculator(
    verifier_name_or_path="Qwen/Qwen3-8B",
    ckpt_output_dir="./eagle3_output/checkpoints",
    data_path="sharegpt",
    mode="online",
    speculator_type="eagle3",
    vllm_endpoint="http://localhost:8234/v1",
    training_gpu_ids=[2, 3],
    epochs=3,
    draft_vocab_size=32000,
    max_samples=5000,
)


# ============================================================================
# 5. Single-GPU training (no FSDP, in-process)
#    For simpler setups or smaller models. Uses training_gpu_id instead of
#    training_gpu_ids.
# ============================================================================

result = train_speculator(
    verifier_name_or_path="Qwen/Qwen3-8B",
    ckpt_output_dir="./eagle3_output/checkpoints",
    data_path="sharegpt",
    mode="offline",
    speculator_type="eagle3",
    vllm_gpu_ids=[0],         # single GPU for vLLM
    training_gpu_id=1,        # single GPU for training (in-process)
    epochs=3,
    draft_vocab_size=32000,
    max_samples=5000,
)


# ============================================================================
# 6. Train-only — from pre-existing hidden states
#    Skips data prep and hidden state extraction entirely. Useful for resuming
#    training or experimenting with different hyperparameters on the same data.
# ============================================================================

result = train_speculator(
    verifier_name_or_path="Qwen/Qwen3-8B",
    ckpt_output_dir="./eagle3_output/checkpoints",
    mode="train_only",
    speculator_type="eagle3",
    data_output_dir="./existing_data",
    hidden_states_path="./existing_data/hidden_states",
    training_gpu_ids=[2, 3],
    epochs=3,
    draft_vocab_size=32000,
)
