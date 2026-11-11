"""Tests for the LoRA/QLoRA/FSDP parts of the memory estimator.

The models are tiny configs built on the meta device, so the tests need no network,
no GPU and no weights. LoRA counts are checked against hand-computed values.
"""

import copy

import pytest
import torch
from transformers import (
    CLIPVisionConfig,
    LlamaConfig,
    LlavaConfig,
    Qwen2Config,
    Qwen3MoeConfig,
)

from training_hub.profiling import memory_estimator as me

HIDDEN, INTERMEDIATE, KV_DIM, VOCAB = 64, 128, 32, 256
# LoRA weight dims (in + out) of one dense decoder layer:
# q 64+64, k 64+32, v 64+32, o 64+64, gate 64+128, up 64+128, down 128+64
DENSE_LAYER = 128 + 96 + 96 + 128 + 192 + 192 + 192
ATTENTION = 128 + 96 + 96 + 128
NUM_EXPERTS, EXPERT_WIDTH = 8, 32


def dense_config(tie: bool = True, layers: int = 2) -> Qwen2Config:
    return Qwen2Config(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=layers,
                       num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB, tie_word_embeddings=tie)


def vlm_config() -> LlavaConfig:
    text = LlamaConfig(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB)
    # More vision layers than text layers, so the vision encoder is the longest layer list
    vision = CLIPVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                              num_attention_heads=2, image_size=32, patch_size=8)
    return LlavaConfig(text_config=text, vision_config=vision)


def moe_config() -> Qwen3MoeConfig:
    # Layer 0 is dense, layer 1 is MoE (and the larger block)
    return Qwen3MoeConfig(hidden_size=HIDDEN, intermediate_size=INTERMEDIATE, moe_intermediate_size=EXPERT_WIDTH,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                          num_experts=NUM_EXPERTS, num_experts_per_tok=2, vocab_size=VOCAB, mlp_only_layers=[0])


@pytest.fixture
def configs(monkeypatch, tmp_path):
    """Serve local configs instead of downloading them, and keep the CSV cache in tmp_path."""
    registry = {
        "dense": dense_config(),
        "dense-untied": dense_config(tie=False),
        "dense-8": dense_config(layers=8),
        "dense-8-untied": dense_config(tie=False, layers=8),
        "vlm": vlm_config(),
        "moe": moe_config(),
    }
    monkeypatch.setattr(me, "MODEL_STORAGE_PATH", str(tmp_path / "model_storage.csv"))
    monkeypatch.setattr(me.AutoConfig, "from_pretrained", lambda path, **kwargs: copy.deepcopy(registry[path]))
    return registry


def placement_or_skip(est):
    """The device_map placement, or skip the test if this transformers version can't compute it."""
    placement = est._calc_split_placement()
    if placement is None:
        pytest.skip("transformers device_map helpers unavailable")
    return placement


def lora(path: str, cls=me.LoRAEstimator, **kwargs):
    kwargs = {"num_gpus": 1, "batch_size": 2, "max_seq_len": 16, "lora_r": 4, "verbose": 0, **kwargs}
    return cls(model_path=path, **kwargs)


def split_weights(est) -> dict:
    """Weights per GPU from the split estimate: params = weights + 4 B x LoRA, gradients = 4 B x LoRA."""
    return {gpu: par - grad for gpu, (par, _, grad, _, _) in est.split_components.items()}


# --- LoRA parameter counts -------------------------------------------------------------

def test_lora_count_dense(configs):
    est = lora("dense")
    assert est._calc_lora_weight_size_per_layer() == {0: DENSE_LAYER, 1: DENSE_LAYER}
    assert est.AB_params == 2 * DENSE_LAYER * 4


def test_lora_count_vlm_skips_vision_tower(configs):
    est = lora("vlm")
    assert "language_model" in est._find_decoder_blocks()
    assert est._calc_lora_weight_size_per_layer() == {0: DENSE_LAYER, 1: DENSE_LAYER}


def test_lora_count_moe_counts_experts_but_not_router(configs):
    est = lora("moe")
    assert est._find_decoder_blocks() == "model.layers"
    per_layer = est._calc_lora_weight_size_per_layer()
    assert per_layer[0] == DENSE_LAYER
    # Fused [experts, out, in] tensors (gate_up 2*32x64, down 64x32) or, in older transformers,
    # one gate/up/down nn.Linear per expert. The router (64 -> 8) is never counted.
    fused = NUM_EXPERTS * ((2 * EXPERT_WIDTH + HIDDEN) + (HIDDEN + EXPERT_WIDTH))
    per_expert_linear = NUM_EXPERTS * 3 * (HIDDEN + EXPERT_WIDTH)
    assert per_layer[1] in (ATTENTION + fused, ATTENTION + per_expert_linear)


def test_lora_count_skips_linear_router(configs):
    # Older transformers versions (and some remote-code models) use an nn.Linear router
    est = lora("moe")
    before = est._calc_lora_weight_size_per_layer()
    est.model.model.layers[1].mlp.gate = torch.nn.Linear(HIDDEN, NUM_EXPERTS, bias=False, device="meta")
    assert est._calc_lora_weight_size_per_layer() == before


def test_decoder_blocks_not_a_longer_per_layer_expert_list(configs):
    # transformers 4.x stores MoE experts as one nn.ModuleList per layer, longer than the layer list
    est = lora("moe")
    experts = torch.nn.ModuleList(torch.nn.Linear(HIDDEN, EXPERT_WIDTH, device="meta") for _ in range(NUM_EXPERTS))
    est.model.model.layers[1].mlp.expert_list = experts
    assert est._find_decoder_blocks() == "model.layers"
    assert set(est._calc_lora_weight_size_per_layer()) == {0, 1}


def test_lora_count_target_modules(configs):
    est = lora("dense", target_modules=["q_proj", "v_proj"])
    assert est.lora_weight_size == 2 * (128 + 96)


def test_lora_count_cached_in_csv(configs):
    first = lora("dense")
    second = lora("dense")
    assert second.found_model
    assert second.lora_weight_size == first.lora_weight_size


# --- DDP vs model splitting ------------------------------------------------------------

def test_ddp_every_gpu_holds_a_full_replica(configs):
    one = lora("dense-8", num_gpus=1)
    ddp = lora("dense-8", num_gpus=4)
    assert ddp.tokens_per_gpu == 2 * 16  # the whole micro-batch, not divided by num_gpus
    assert ddp._calc_model_params() == one._calc_model_params() == ddp.num_params * 2 + ddp.AB_params * 4
    assert ddp._calc_optimizer() == one._calc_optimizer()
    assert ddp.estimate() == one.estimate()


def test_split_counts_offloaded_weights(configs):
    # GPUs far too small for the model: the placement offloads, and the estimate must not fit
    tiny = 64 * 1024
    est = lora("dense-8", num_gpus=2, enable_model_splitting=True, gpu_memory=tiny)
    with pytest.warns(UserWarning, match="doesn't fit"):
        est.estimate()
    total = me._PlacementSizer(est.model, quantize_4bit=False).total_size(est.model)
    assert sum(split_weights(est).values()) == pytest.approx(total, rel=0.01)
    assert max(sum(c) for c in est.split_components.values()) > tiny


def test_cache_rows_from_older_versions_are_rebuilt(configs):
    lora("dense")
    rows = me.pd.read_csv(me.MODEL_STORAGE_PATH, index_col="name")
    assert rows.loc["dense", "cache_version"] == me.MODEL_STORAGE_VERSION
    rows.loc["dense", "cache_version"] = me.MODEL_STORAGE_VERSION - 1
    rows.loc["dense", "num_params"] = 1  # a stale value that must not be reused
    rows.to_csv(me.MODEL_STORAGE_PATH, index=True, index_label="name")
    est = lora("dense")
    assert not est.found_model
    assert est.num_params > 1


def test_split_spreads_weights_across_gpus(configs):
    est = lora("dense-8", num_gpus=2, enable_model_splitting=True)
    est.estimate()
    weights = split_weights(est)
    total = me._PlacementSizer(est.model, quantize_4bit=False).total_size(est.model)
    assert set(weights) == {0, 1}
    assert all(0 < w < total for w in weights.values())
    assert sum(weights.values()) == pytest.approx(total, rel=0.01)  # plus small buffers
    assert est.tokens_per_gpu == 2 * 16  # every GPU sees the whole micro-batch


# --- Logits -----------------------------------------------------------------------------

def test_fused_loss_has_no_logits(configs):
    assert lora("dense", num_gpus=2)._calc_outputs() == 0
    est = lora("dense-8", num_gpus=2, enable_model_splitting=True)
    est.estimate()
    assert all(components[4] == 0 for components in est.split_components.values())


@pytest.mark.parametrize("path", ["dense-8", "dense-8-untied"])
def test_logits_only_on_the_lm_head_gpu(configs, path):
    est = lora(path, num_gpus=2, enable_model_splitting=True, fused_loss=False)
    est.estimate()
    device_map, _ = placement_or_skip(est)
    head_gpu = est._device_of("lm_head", device_map)
    if est.model.config.tie_word_embeddings:
        assert head_gpu == est._device_of("model.embed_tokens", device_map)
    logits = 2 * 16 * VOCAB * (4 + 2)  # fp32 logits + bf16 copy
    assert {gpu: c[4] for gpu, c in est.split_components.items()} == \
        {gpu: (logits if gpu == head_gpu else 0) for gpu in est.split_components}


def test_measured_model_types_materialize_logits(configs, monkeypatch):
    monkeypatch.setattr(me, "MATERIALIZES_LOGITS", ("qwen2",))
    assert lora("dense", num_gpus=2)._calc_outputs() == 2 * 16 * VOCAB * (4 + 2)
    assert lora("dense", num_gpus=2, fused_loss=True)._calc_outputs() == 0


# --- QLoRA --------------------------------------------------------------------------------

def test_qlora_quantizes_only_linear_weights(configs):
    dense = lora("dense-untied", cls=me.QLoRAEstimator)
    linear = 2 * (HIDDEN * HIDDEN * 2 + HIDDEN * KV_DIM * 2 + 3 * HIDDEN * INTERMEDIATE)  # per layer x 2 layers
    assert dense.loaded_weight_bytes == linear * 0.5 + (dense.num_params - linear) * 2

    moe = lora("moe", cls=me.QLoRAEstimator)
    sizer = me._PlacementSizer(moe.model, quantize_4bit=True)
    fused_experts = [name for name, p in moe.model.named_parameters() if p.dim() == 3]
    assert all(name not in sizer.quantized_params for name in fused_experts)
    assert "lm_head.weight" not in sizer.quantized_params


@pytest.mark.parametrize("path,untied", [("dense-8", False), ("dense-8-untied", True)])
def test_qlora_lm_head_workspace_for_untied_models(configs, path, untied):
    est = lora(path, cls=me.QLoRAEstimator, num_gpus=2, enable_model_splitting=True)
    est.estimate()
    device_map, _ = placement_or_skip(est)
    head_gpu = est._device_of("lm_head", device_map)
    workspace = VOCAB * HIDDEN * 2 if untied else 0
    assert est.split_components[head_gpu][4] == workspace
    assert lora(path, cls=me.QLoRAEstimator)._calc_outputs() == workspace  # 1 GPU


# --- FSDP gather ------------------------------------------------------------------------

def test_fsdp_gather_uses_the_largest_block(configs):
    est = me.BasicEstimator(model_path="moe", num_gpus=2, max_tokens_per_gpu=1000,
                            count_fsdp_gather=True, verbose=0)
    blocks = [sum(p.numel() for p in layer.parameters()) for layer in est.model.model.layers]
    assert blocks[1] > blocks[0]
    root = 2 * VOCAB * HIDDEN  # untied: embeddings + lm_head
    assert est._calc_additional() == 4 * max(root, max(blocks))
    assert est._calc_additional() > 4 * max(root, sum(blocks) / len(blocks))

    # The second run reads the largest block from the CSV cache
    cached = me.BasicEstimator(model_path="moe", num_gpus=2, max_tokens_per_gpu=1000,
                               count_fsdp_gather=True, verbose=0)
    assert cached.found_model
    assert cached._calc_additional() == est._calc_additional()


def test_fsdp_gather_off_by_default(configs):
    est = me.BasicEstimator(model_path="moe", num_gpus=2, max_tokens_per_gpu=1000, verbose=0)
    assert est._calc_additional() == 0
