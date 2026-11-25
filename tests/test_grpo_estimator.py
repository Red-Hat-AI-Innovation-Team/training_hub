"""Tests for the verl lora_grpo / grpo memory and GPU-hour estimates.

Uses tiny configs on the meta device (no network, GPU or weights). The calibration against the
Qwen3-4B runs needs the real config, so it lives outside the test suite; these tests check the
formulas' structure.
"""

import copy

import pytest
from transformers import Qwen2Config

from training_hub.profiling import memory_estimator as me
from training_hub.profiling import timing_estimator as te

HIDDEN, INTERMEDIATE, LAYERS, VOCAB = 64, 128, 2, 256
# LoRA weight dims (in + out) of one decoder layer, as in test_memory_estimator.py
DENSE_LAYER = 128 + 96 + 96 + 128 + 192 + 192 + 192
GPU = 80 * 1024**3


@pytest.fixture
def configs(monkeypatch, tmp_path):
    """Serve a local config instead of downloading it, and keep the CSV cache in tmp_path."""
    config = Qwen2Config(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=LAYERS,
                         num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB, tie_word_embeddings=True)
    monkeypatch.setattr(me, "MODEL_STORAGE_PATH", str(tmp_path / "model_storage.csv"))
    monkeypatch.setattr(me.AutoConfig, "from_pretrained", lambda path, **kwargs: copy.deepcopy(config))


def grpo(**kwargs) -> me.GRPOEstimator:
    kwargs = {"num_gpus": 1, "gpu_memory": GPU, "model_path": "dense", "max_seq_len": 1000,
              "lora_r": 8, "verbose": 0, **kwargs}
    return me.GRPOEstimator(**kwargs)


def test_components_follow_verl_precision(configs):
    est = grpo()
    # fp32 base + fp32 LoRA weights; gradients and 2 AdamW moments for the LoRA params only
    assert est.lora_params == LAYERS * DENSE_LAYER * 8
    assert est._calc_model_params() == (est.num_params + est.lora_params) * 4
    assert est._calc_gradients() == est.lora_params * 4
    assert est._calc_optimizer() == est.lora_params * 8
    assert est._calc_intermediate_activations() == 1000 * 2 * HIDDEN * LAYERS
    assert est._calc_outputs() == 1000 * VOCAB * (4 + 2)


def test_rollout_phase_is_resident_state_plus_vllm(configs):
    est = grpo(gpu_memory_utilization=0.45)
    est._calc_subtotal()
    _, _, grad, act, outputs, additional = est.phases["rollout"]
    assert (grad, act, outputs) == (0, 0, 0)
    assert additional == 0.45 * GPU + me.VERL_RUNTIME_OVERHEAD_BYTES


def test_estimate_is_the_peak_phase_with_the_multipliers(configs):
    est = grpo()
    low, mid, high = est.estimate()
    assert est.peak_phase == "rollout"
    subtotal = sum(est.phases["rollout"])
    assert (low, mid, high) == (int(0.95 * subtotal), int(subtotal), int(1.1 * subtotal))


def test_long_sequences_move_the_peak_to_training(configs):
    short, long = grpo(max_seq_len=1000), grpo(max_seq_len=100_000_000)
    short.estimate()
    long.estimate()
    assert short.peak_phase == "rollout" and long.peak_phase == "training"


def test_training_state_shards_across_gpus(configs):
    one, four = grpo(num_gpus=1), grpo(num_gpus=4)
    assert four._calc_model_params() == pytest.approx(one._calc_model_params() / 4)
    assert four._calc_optimizer() == pytest.approx(one._calc_optimizer() / 4)
    # FSDP's block gather only exists with more than one GPU
    assert four._calc_additional("training") > one._calc_additional("training") == me.VERL_RUNTIME_OVERHEAD_BYTES


def test_full_grpo_trains_every_weight(configs):
    est = grpo(lora_r=0)
    assert est.lora_params == 0 and est.num_trainable_params == est.num_params
    assert est._calc_gradients() + est._calc_optimizer() == est.num_params * 12
    vllm_weights = est.num_params * 2
    assert est._calc_additional("weight sync") == vllm_weights + me.VERL_SYNC_BUCKET_BYTES + me.VERL_RUNTIME_OVERHEAD_BYTES


def test_dummy_load_format_gathers_the_full_model_on_multi_gpu(configs):
    dummy = grpo(num_gpus=2, load_format="dummy")
    preloaded = grpo(num_gpus=2, load_format="safetensors")
    gap = dummy._calc_additional("weight sync") - preloaded._calc_additional("weight sync")
    assert gap == dummy.num_params * 4 - dummy.lora_params * 4


def test_estimate_dispatches_grpo_methods(configs):
    lora = me.estimate(training_method="lora_grpo", num_gpus=1, gpu_memory=GPU, model_path="dense",
                       max_seq_len=1000, lora_r=8, verbose=0)
    full = me.estimate(training_method="grpo", num_gpus=1, gpu_memory=GPU, model_path="dense",
                       max_seq_len=1000, lora_r=8, verbose=0)
    assert lora == grpo().estimate()
    assert full == grpo(lora_r=0).estimate()


def test_invalid_settings_raise(configs):
    with pytest.raises(ValueError):
        grpo(gpu_memory_utilization=1.2)
    with pytest.raises(ValueError):
        grpo(max_seq_len=None)


# --- GPU-hours -------------------------------------------------------------------------

def cost(**kwargs) -> dict:
    kwargs = {"model_path": "dense", "mean_prompt_tokens": 2536, "mean_response_tokens": 295,
              "num_prompts": 992, "num_iterations": 3, "lora_r": 16, "verbose": 0, **kwargs}
    return te.estimate_grpo_cost(**kwargs)


def test_steps_drop_the_last_partial_batch(configs):
    result = cost()
    assert result["num_steps"] == 27  # 3 epochs x (992 // 100)
    assert result["rollouts"] == 27 * 800


def test_gpu_hours_follow_the_fitted_formula(configs):
    result = cost()
    sizes = grpo(lora_r=16)
    per_token = (te.GRPO_SECONDS_PER_TOKEN_PER_PARAM * sizes.num_params
                 + te.GRPO_SECONDS_PER_TOKEN_PER_LORA_PARAM * sizes.lora_params)
    expected = 800 * (2536 + 295) * per_token * 27 / 3600
    low, mid, high = result["gpu_hours"]
    assert mid == pytest.approx(expected)
    assert low == pytest.approx(expected * te.GRPO_TIME_LOW_MULTIPLIER)
    assert high == pytest.approx(expected * te.GRPO_TIME_HIGH_MULTIPLIER)


def test_higher_rank_and_more_gpus(configs):
    base = cost()
    assert cost(lora_r=128)["gpu_hours"][1] > base["gpu_hours"][1]
    four = cost(n_gpus=4)
    assert four["gpu_hours"][1] == pytest.approx(base["gpu_hours"][1])
    assert four["wall_hours"][1] == pytest.approx(base["wall_hours"][1] / 4)


def test_price_and_default_response_length(configs):
    result = cost(gpu_hour_price=2.0)
    assert result["cost"][1] == pytest.approx(2.0 * result["gpu_hours"][1])
    default = cost(mean_response_tokens=None, max_tokens=512)
    assert default["tokens_per_step"] == 800 * (2536 + te.DEFAULT_RESPONSE_SHARE * 512)


def test_cost_needs_steps(configs):
    with pytest.raises(ValueError):
        cost(num_prompts=None)
    with pytest.raises(ValueError):
        cost(num_prompts=50)  # fewer prompts than one batch
