try: from typing import override
except ImportError: from typing_extensions import override
import logging
import warnings
import torch
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig, PreTrainedModel
from mini_trainer.osft_utils import MODEL_CONFIGS

from torch.distributed.fsdp import ShardingStrategy 

import os
from collections import defaultdict
import pandas as pd

"""
Code assisted by Cursor/Claude4
"""

sh = ShardingStrategy.NO_SHARD

# Constants that specify
FLOAT32_BYTES_N: int = 4
FLOAT16_BYTES_N: int = 2
FLOAT8_BYTES_N: int = 1
FLOAT4_BYTES_N: float = 0.5
ADAMW_PARAMS_N: int = 2

# Module-name terms marking non-text parts of multimodal models, which LoRA leaves frozen by default
NON_TEXT_MODULE_TERMS: tuple[str, ...] = ("vision", "visual", "audio", "multi_modal_projector")

# Peak bytes per token per unit of MLP width for the decoder layer being computed. With Unsloth's
# offloaded gradient checkpointing, each GPU holds about one layer's live working set, no matter
# how many layers it holds. Fitted on Muse Glimmer 30B; holds for Gemma 4 31B and Nemotron 30B-A3B.
WS_BYTES_PER_MLP_UNIT: int = 30

# Model types whose lm_head runs every step under Unsloth, so the logits are materialized
# (fp32 + a bf16 copy). Other models use Unsloth's fused linear + cross-entropy loss, which
# never materializes them. Extend from measurements.
MATERIALIZES_LOGITS: tuple[str, ...] = ("gemma4",)

# Where the estimator caches each model's sizes (tests point this somewhere temporary)
MODEL_STORAGE_PATH: str = os.path.join(os.path.dirname(os.path.realpath(__file__)), "model_storage.csv")

# Helper function to do the rounding when printing 
def ROUNDER(value: int) -> str: return str(round(value / 1073741824, 1))

# Helper function to calculate how much the given unfrozen_rank_ratio 
# will be affecting the OSFT estimation (through a quadratic mapping where
# 0 is 0.5 of SFT's value, 1/3 is equal to SFT's value, and 1 is twice of SFT's value)
def OSFT_RATIO(value: float) -> float: return 0.5 + (1.5 * value)

class _ModelStorage:
    """
    To reduce the amount of hard disk space and RAM needed to run the estimator, we're going
    to store the essential information of various models into a CSV file. This CSV
    will be updated whenever a new model is used for the estimator. 
    """
    def __init__(self, data: dict):
        load_param = lambda key: data[key] if key in data else None # store None if the param doesn't exist

        self.num_params = load_param('num_params')
        self.num_trainable_params = load_param('num_trainable_params')
        self.num_layers = load_param('num_layers') 
        self.hidden_size = load_param('hidden_size')
        self.vocab_size = load_param('vocab_size')
        self.lora_weight_size = load_param('lora_weight_size')
        self.osft_params = load_param('osft_params') 
        self.max_block_params = load_param('max_block_params')


class _PlacementSizer:
    """
    Stands in for a transformers HfQuantizer so its device_map helpers size each parameter
    as Unsloth loads it: bf16, or 4-bit for the nn.Linear weights that bitsandbytes quantizes
    under QLoRA (everything except the lm_head and the model's fp32 modules).
    The real bitsandbytes quantizer needs the bitsandbytes package, even just for sizing.
    Only the methods those helpers call are implemented.

    Args:
        model (PreTrainedModel): The meta model to size
        quantize_4bit (bool): If true, size the quantized weights at 4 bits
    """
    def __init__(self, model: PreTrainedModel, quantize_4bit: bool):
        self.quantize_4bit = quantize_4bit
        self.quantized_params: set[str] = set()
        if quantize_4bit:
            skip = set(getattr(model, '_keep_in_fp32_modules', None) or [])
            output_embeddings = model.get_output_embeddings()
            for name, module in model.named_modules():
                if module is output_embeddings or not isinstance(module, torch.nn.Linear):
                    continue
                if any(name == key or name.endswith("." + key) or name.startswith(key + ".") for key in skip):
                    continue
                self.quantized_params.add(name + ".weight")

    def param_element_size(self, model: torch.nn.Module, param_name: str, param: torch.Tensor) -> float:
        """
        Bytes per element of the given parameter once loaded.

        Returns:
            float: 0.5 for 4-bit quantized weights, 2 for other float tensors (bf16)
        """
        if param_name in self.quantized_params:
            return FLOAT4_BYTES_N
        return FLOAT16_BYTES_N if param.is_floating_point() else param.element_size()

    def adjust_max_memory(self, max_memory: dict[int | str, float]) -> dict[int | str, float]:
        """
        Leave 10% for quantization buffers, as transformers' bitsandbytes quantizer does.

        Returns:
            dict[int | str, float]: The adjusted per-device memory caps
        """
        if not self.quantize_4bit:
            return max_memory
        return {device: memory * 0.9 for device, memory in max_memory.items()}

    def total_size(self, model: PreTrainedModel) -> float:
        """
        Returns:
            float: The model's total loaded size in bytes
        """
        return sum(param.numel() * self.param_element_size(model, name, param)
                   for name, param in model.named_parameters())


class BasicEstimator:
    """
    An estimator for the memory usage of a typical LLM training pass via SFT.
    Estimators for other methods can subclass this class.

    Args:
        num_gpus (int): Number of GPUs to use for training (default: 8)
        gpu_memory (int): The VRAM of each GPU in bytes (default: 85899345920 for 80 GB)
        model_path (str): HuggingFace model path to the model to fine-tune
                        (default: "ibm-granite/granite-3.3-8b-instruct")
        batch_size (int): The number of samples in a minibatch that the model
                            has to see before backpropping.
        max_seq_len (int): Maximum sequence length of dataset samples 
        max_tokens_per_gpu (int): The maximum number of tokens that can be placed
                                on a single GPU during each mini-batch.
        use_liger (bool): If true, estimate assuming Liger Kernels are used.
        verbose (int): The level of verbosity to print out. Set to 0 for no printing,
                        set to 1 to print out only hardware recommendations,
                        set to 2 for a detailed memory breakdown.
        count_fsdp_gather (bool): If true and num_gpus > 1, add the memory each GPU needs
                        to gather the largest FSDP unit (default: False).
    """

    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        batch_size: int | None = None,
        max_seq_len: int | None = None,
        max_tokens_per_gpu: int | None = None,
        use_liger: bool = False,
        verbose: int = 1,
        trust_remote_code: bool = False,
        count_fsdp_gather: bool = False,
    ):
        self.num_gpus = num_gpus
        self.gpu_memory = gpu_memory
        self.model_path = model_path
        self.use_liger = use_liger
        self.verbose = verbose
        self.trust_remote_code = trust_remote_code
        self.count_fsdp_gather = count_fsdp_gather

        self.storage_path = MODEL_STORAGE_PATH

        # Load model information from the CSV, or load it directly if it's not in the CSV.
        # TODO: Adjust this code so that we download and preprocess the information from safetensors instead
        # of downloading the entire model. 
        self.found_model = False
        if os.path.exists(self.storage_path):
                read_csv = pd.read_csv(self.storage_path, index_col='name') 
                self.model_storage = read_csv.to_dict(orient='index')
                if model_path in self.model_storage:
                    try:
                        self.found_model = True
                        self.model = _ModelStorage(self.model_storage[model_path])
                        self.num_params: int = self.model.num_params
                        self.num_trainable_params: int = self.model.num_trainable_params
                        self.num_layers: int = self.model.num_layers
                        self.hidden_size: int = self.model.hidden_size
                    except:
                        self.found_model = False
                        self._load_model("Model missing necessary parameters.", trust_remote_code=trust_remote_code)
                else: self._load_model("Model not found in the CSV.", trust_remote_code=trust_remote_code)                
        else:
            self.model_storage = {}
            self._load_model("No CSV file found.", trust_remote_code=trust_remote_code)

        # This is a scalar that's applied during the output memory calculation
        self.output_constant = 8/3
        self.main_dtype_bytes = FLOAT32_BYTES_N
        self.opt_params = ADAMW_PARAMS_N

        # Set the multipliers to define the estimation bounds
        self.LOW_MULTIPLIER = 1.0
        self.MEDIUM_MULTIPLIER = 1.1
        self.HIGH_MULTIPLIER = 1.3

        # Determine how many tokens will be placed on each GPU
        self._resolve_tokens_per_gpu(batch_size, max_seq_len, max_tokens_per_gpu)

        # Update the CSV if necessary.
        if not self.found_model: self._update_model_storage()


    def _load_model(self, warn: str, trust_remote_code: bool = False):
        """
        Load the model if we can't find the information locally.
        """
        warnings.warn(warn + "\nThe model's config will be downloaded from HuggingFace and the model built on the meta device.")
        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=trust_remote_code)
        self.model = self._build_meta_model(config, trust_remote_code)

        # Determine parameters needed for calculations. Multimodal models (e.g. Gemma 4) keep
        # the language model's settings in a nested text config, not at the top level.
        text_config = config.get_text_config()
        self.num_params: int = self.model.num_parameters(only_trainable=False)
        self.num_trainable_params: int = self.model.num_parameters(only_trainable=True)
        self.num_layers: int = text_config.num_hidden_layers
        self.hidden_size: int = text_config.hidden_size


    @staticmethod
    def _build_meta_model(config: PretrainedConfig, trust_remote_code: bool = False) -> PreTrainedModel:
        """
        Build the model on the meta device from its config, so no weights are downloaded.

        Uses a class with an lm_head (AutoModelForCausalLM, then AutoModelForImageTextToText for
        multimodal models) so untied output heads are counted. Falls back to AutoModel.

        Args:
            config (PretrainedConfig): The model's config
            trust_remote_code (bool): Whether to allow custom model code from the Hub

        Returns:
            PreTrainedModel: The model with all parameters on the meta device
        """
        auto_classes = [AutoModelForCausalLM]
        try:
            from transformers import AutoModelForImageTextToText
            auto_classes.append(AutoModelForImageTextToText)
        except ImportError:
            pass
        auto_classes.append(AutoModel)

        with torch.device('meta'):
            for auto_class in auto_classes:
                try:
                    return auto_class.from_config(config, trust_remote_code=trust_remote_code)
                except ValueError:  # this config isn't registered for this auto class
                    continue
        raise ValueError(f"Could not build a model for config type '{config.model_type}'")


    def _update_model_storage(self):
        """
        Utility function to store the models' information into a CSV. 
        """
        self.model_storage[self.model_path] = \
            {
                'num_params': self.num_params,
                'num_trainable_params': self.num_trainable_params,
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size,
                'vocab_size': self._get_vocab_size(),
                'lora_weight_size': self._calc_lora_weight_size(),
                'osft_params': self._calc_osft_params(),
                'max_block_params': self._calc_max_block_params(),
            }
        data_out = pd.DataFrame.from_dict(self.model_storage, orient='index')
        data_out.to_csv(self.storage_path, index=True, index_label='name')


    def _resolve_tokens_per_gpu(self, batch_size: int | None, max_seq_len: int | None, max_tokens_per_gpu: int | None):
        """
        Determine how many tokens will be placed on each GPU during each mini-batch
        It will be bounded by either the largest number of tokens in a batch (divided by # GPUs)
        *or* the value of max_tokens_per_gpu.
        """
        self.tokens_per_gpu = None            
        if batch_size is None or max_seq_len is None:
            self.tokens_per_gpu: int = max_tokens_per_gpu
        elif max_tokens_per_gpu is None:
            self.tokens_per_gpu: int = batch_size * max_seq_len / self.num_gpus
        else:
            self.tokens_per_gpu: int = min(max_tokens_per_gpu,
                                            batch_size * max_seq_len / self.num_gpus)

        if self.tokens_per_gpu is None:
            raise ValueError("At least one of (batch_size, max_seq_len) or " +
                                "max_tokens_per_gpu must be provided")


    def _find_decoder_blocks(self) -> str:
        """
        Find the module list holding the text decoder blocks: the longest nn.ModuleList
        outside the vision/audio towers (e.g. "model.layers" or "model.language_model.layers").

        Returns:
            str: The module name of the decoder block list
        """
        candidates = [(name, module) for name, module in self.model.named_modules()
                      if isinstance(module, torch.nn.ModuleList) and len(module) > 1
                      and not any(term in name for term in NON_TEXT_MODULE_TERMS)]
        if not candidates:
            raise ValueError(f"Could not find the decoder blocks of {self.model_path}")
        return max(candidates, key=lambda candidate: len(candidate[1]))[0]


    def _calc_lora_weight_size_per_layer(self, target_modules: list[str] | None = None) -> dict[int, int]:
        """
        Sum of (in_features + out_features) over the weights LoRA adapts, per decoder block.
        Multiplying by lora_r gives the number of LoRA A+B parameters.

        Matches Unsloth's default (target_modules=None): every nn.Linear in the text decoder
        blocks except MoE routers, plus fused [num_experts, out, in] expert weights, which get
        one adapter per expert. Embeddings, lm_head and vision/audio towers are left out.

        Args:
            target_modules (list[str] | None): Only count modules (or fused expert parameters)
                                whose name ends with one of these names. None counts all of them.

        Returns:
            dict[int, int]: Block index -> summed weight dimensions in that block
        """
        text_config = self.model.config.get_text_config()
        num_experts = (getattr(text_config, "n_routed_experts", None)
                       or getattr(text_config, "num_experts", None)
                       or getattr(text_config, "num_local_experts", None))
        blocks = self._find_decoder_blocks()

        def is_target(name: str) -> bool:
            """Whether a module or parameter name is one of target_modules (all are, if None)."""
            return target_modules is None or name.split(".")[-1] in target_modules

        per_layer: dict[int, int] = {}
        for name, module in self.model.named_modules():
            if not name.startswith(blocks + ".") or any(term in name for term in NON_TEXT_MODULE_TERMS):
                continue
            layer = int(name[len(blocks) + 1:].split(".")[0])
            size = 0
            # A router has one output per expert; LoRA doesn't adapt it
            if isinstance(module, torch.nn.Linear) and module.out_features != num_experts and is_target(name):
                size += module.in_features + module.out_features
            # Fused expert weights are stored as one [num_experts, out, in] parameter
            for param_name, param in module.named_parameters(recurse=False):
                if (num_experts and param.dim() == 3 and param.shape[0] == num_experts
                        and (is_target(name) or is_target(param_name))):
                    size += param.shape[0] * (param.shape[1] + param.shape[2])
            per_layer[layer] = per_layer.get(layer, 0) + size
        return per_layer


    def _calc_lora_weight_size(self, target_modules: list[str] | None = None) -> int:
        """
        Sum of (in_features + out_features) over all the weights LoRA adapts.
        See _calc_lora_weight_size_per_layer.

        Args:
            target_modules (list[str] | None): Only count modules with these names. None counts all.

        Returns:
            int: The summed weight dimensions. Multiply by lora_r for the LoRA A+B parameters.
        """
        return sum(self._calc_lora_weight_size_per_layer(target_modules).values())


    def _calc_max_block_params(self) -> int:
        """
        The number of parameters in the largest decoder block. For MoE and hybrid models the
        blocks differ a lot (e.g. Nemotron's MoE blocks are ~2.2x the average block).

        Returns:
            int: The largest decoder block's parameter count
        """
        return max(sum(param.numel() for param in block.parameters())
                   for block in self.model.get_submodule(self._find_decoder_blocks()))


    def _calc_osft_params(self) -> int:
        """
        Iterate through the layers in this model and determine how
        many bytes would be needed to store this model when trained with OSFT
        """
        total_bytes: int = 0
        # named_parameters() lists tied weights (e.g. embeddings shared with lm_head) once
        for layer_name, _ in self.model.named_parameters():
            total_bytes += self._check_layer_params(layer_name)
        return total_bytes


    def _check_layer_params(
                        self,
                        layer_name: str
                    ) -> int:
        """
        For the given layer, determine how many bytes would be needed
        to store this layer when trained with OSFT
        """

        # If the layer is trainable, i.e. it contains any of these terms,
        # then we will need to store a SVD decomposition of the layer in memory.
        layer_data = self.model.state_dict()[layer_name]
        if layer_data.dim() < 2:
            return layer_data.numel() * FLOAT32_BYTES_N
        for term in self._find_valid_layers():
            lowest_dim = layer_data.shape[0] if layer_data.shape[0] < \
                            layer_data.shape[1] else layer_data.shape[1]
            if layer_name.find(term) > -1 and layer_name.find('weight') > -1:
                U_bytes_n: int = layer_data.shape[0] * lowest_dim
                S_bytes_n: int = lowest_dim
                V_bytes_n: int = lowest_dim * layer_data.shape[1]
                return (U_bytes_n + S_bytes_n + V_bytes_n) * FLOAT32_BYTES_N

        # If not, we'll only be storing the layer itself in memory. 
        byte_val = layer_data.numel() * FLOAT32_BYTES_N
        return byte_val


    def _calc_model_params(self):
        """
        Calcuate the VRAM for storing the model parameters
        """
        return self.num_params * self.main_dtype_bytes / self.num_gpus


    def _calc_gradients(self):
        """
        Calcuate the VRAM for storing the gradients
        """
        return (self.main_dtype_bytes * self.num_trainable_params / self.num_gpus)


    def _calc_optimizer(self):
        """
        Calculate the VRAM for storing the optimizer states
        """
        return (self.main_dtype_bytes * self.num_trainable_params * self.opt_params / self.num_gpus)


    def _calc_intermediate_activations(self):
        """
        Calculate the VRAM for storing the model's intermediate activations
        """
        return (self.tokens_per_gpu * self.main_dtype_bytes  * self.num_layers * self.hidden_size) 


    def _get_vocab_size(self):
        """
        Get the vocabulary size of the model
        """
        if hasattr(self.model, 'embed_tokens') and hasattr(self.model.embed_tokens, 'num_embeddings'):
            vocab_size = self.model.embed_tokens.num_embeddings
        elif hasattr(self.model, 'config') and hasattr(self.model.config.get_text_config(), 'vocab_size'):
            vocab_size = self.model.config.get_text_config().vocab_size
        elif hasattr(self.model, 'vocab_size'):
            vocab_size = self.model.vocab_size
        else:
            try:
                vocab_size = self.model.get_input_embeddings().num_embeddings
            except AttributeError as e:
                raise ValueError("Could not find the given model's vocabulary size") from e
        return vocab_size


    def _calc_outputs(self):
        """
        Calculate the VRAM for storing the model's activated outputs.
        Note that this value is 0 if Liger Kernels are used.
        """
        if not self.use_liger:
            self.vocab_size = self._get_vocab_size()
            return (self.tokens_per_gpu * self.main_dtype_bytes * self.vocab_size) * self.output_constant
        else:
            return 0


    def _calc_additional(self, **kwargs):
        """
        Calculate any additional VRAM that this training method might need.
        With count_fsdp_gather on more than one GPU, each GPU also gathers one whole FSDP
        unit at a time: 4 bytes (16-bit weight + gradient) x the largest unit's parameters,
        as in DeepSpeed's estimate_zero3_model_states_mem_needs. The units are each decoder
        block and the root (the embeddings, plus the lm_head when it isn't tied).
        """
        if not self.count_fsdp_gather or self.num_gpus <= 1:
            return 0
        # The largest block needs the model structure, unless the CSV cache already has it
        cached = getattr(self.model, 'max_block_params', None) if self.found_model else None
        if self.found_model and (cached is None or pd.isna(cached)):
            self._load_model("Could not find the largest decoder block's size in the CSV cache.",
                            trust_remote_code=self.trust_remote_code)
            self._update_model_storage()
            cached = None
        block = int(cached) if cached is not None else self._calc_max_block_params()
        config = getattr(self.model, 'config', None) or \
            AutoConfig.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        embed = self._get_vocab_size() * self.hidden_size
        root = embed if config.get_text_config().tie_word_embeddings else 2 * embed
        return 4 * max(root, block)


    def _apply_overhead(self, subtotal):
        """
        Apply 0-30% overhead to the subtotal of VRAM needed

        Args:
            subtotal (int): The subtotal of VRAM needed

        Returns:
            gpu_tuple: A triplet of the lower bound, the specific estimate,
                        and upper bound of VRAM required to train this model
            overhead_tuple: A triplet of the lowest, expected, and highest amount of
                            overhead memory required for training
        """
        gpu_vram_total_low: int = int(self.LOW_MULTIPLIER * subtotal)
        gpu_vram_total_mid: int = int(self.MEDIUM_MULTIPLIER *  subtotal)
        gpu_vram_total_high: int = int(self.HIGH_MULTIPLIER * subtotal)
        extra_low: int = max(gpu_vram_total_low - subtotal, 0)
        extra_mid: int = gpu_vram_total_mid - subtotal
        extra_high: int = gpu_vram_total_high - subtotal
        return(gpu_vram_total_low, gpu_vram_total_mid, gpu_vram_total_high), \
                (extra_low, extra_mid, extra_high)


    def _print_results(self, results, overhead, gpu_vram_par, gpu_vram_opt,
                        gpu_vram_grad, gpu_vram_act, gpu_vram_outputs, gpu_vram_additional):
        """
        Print out a breakdown of the estimated memory requirements
        """

        print("Estimations for " + self.model_path + ":\n\n")

        print("Summary:")
        print("The expected amount of memory needed to run this model is about " + 
                ROUNDER(results[1] * self.num_gpus) + " GB") 
        print("The lower and upper bounds are " + 
                ROUNDER(results[0] * self.num_gpus)  + " - " + 
                ROUNDER(results[2] * self.num_gpus) + " GB") 
        print("If you have " + str(self.num_gpus) + " GPUs, you will need about " + \
                ROUNDER(results[1]) + " GB, with bounds of " +
                ROUNDER(results[0]) + " - " + ROUNDER(results[2]) + " GB per GPU")
        print("\n")

        print("Component Breakdown:")
        comp_str = lambda val, name: "Each GPU will need " + ROUNDER(val) + " GB to store the " + name
        print(comp_str(gpu_vram_par, "model parameters"))
        print(comp_str(gpu_vram_opt, "optimizer states"))
        print(comp_str(gpu_vram_grad, "gradients"))
        print(comp_str(gpu_vram_act, "intermediate activations"))

        if self.use_liger:
            print("Since Liger Kernels are being used, no additional memory is needed to store the outputs")
        else:
            print(comp_str(gpu_vram_outputs, "outputs"))
        if gpu_vram_additional > 0:
            print("This method also requires each GPU to use an additional " + ROUNDER(gpu_vram_additional) + " GB")
        print("Up to " + ROUNDER(overhead[2]) + " GB can be expected as overhead")


    def _print_tips(self, results):
        """
        Print out information on how feasible it is to train the model on
        the user's given hardware and, if necessary, recommend the necessary
        amount of additional memory. 
        """

        min_message = "Minimum extra memory required (to reach the low bound of memory requirements): " + \
            ROUNDER(results[0] - self.gpu_memory) + " GB"
        mid_message = "Recommended amount of extra memory (to reach the likely estimated memory requirements): " + \
            ROUNDER(results[1] - self.gpu_memory) + " GB"
        max_message = "Ideal amount of extra memory required (to reach the upper bound of memory requirements): " + \
            ROUNDER(results[2] - self.gpu_memory) + " GB"

        print("\nDecision:")
        
        if results[2] <= self.gpu_memory:
            print("The proposed training setup should work for your hardware.\n")
        elif results[1] <= self.gpu_memory: 
            print("The proposed training setup will likely work for your hardware.\n")
            print(max_message)
        elif results[0] <= self.gpu_memory:
            print("The proposed training setup may work but isn't recommended for your hardware.\n")
            print(mid_message)
            print(max_message)
        else: 
            print("The proposed training setup is impossible for your hardware.\n\n")
            print(min_message)
            print(mid_message)
            print(max_message)
        

    def _calc_subtotal(self):
        """
        Calculate the amount of memory expected before applying overhead.
        """
        # Calculate each piece of memory to be factored into the estimation
        gpu_vram_par = self._calc_model_params()
        gpu_vram_opt: int = self._calc_optimizer()
        gpu_vram_grad: int = self._calc_gradients()
        gpu_vram_act: int = self._calc_intermediate_activations()
        gpu_vram_outputs: int = self._calc_outputs()
        gpu_vram_additional: int = self._calc_additional()

        # Sum up each proposed amount of memory
        subtotal: int = gpu_vram_par + gpu_vram_opt + gpu_vram_grad + \
                        gpu_vram_act + gpu_vram_outputs + gpu_vram_additional

        return subtotal, gpu_vram_par, gpu_vram_opt, gpu_vram_grad, \
            gpu_vram_act, gpu_vram_outputs, gpu_vram_additional


    def estimate(self) -> tuple[int, int, int]:
        """
        Calculate the memory needed to fine tune the given model for the 
        given hyperparameters. After that, determine how possible it is for
        the given hardware to run the model, and note how much more memory
        is needed to make the training more feasible. 

        Note that this estimate assumes training_hub will be used, 
        in which all data types are float32 and the optimizer is always AdamW.

        Return:
            A tuple containing three values:
                lower_bound (int): The lower bound of the memory usage (in bytes)
                expected (int): The expected amount of memory usage (in bytes)
                upper_bound (int): The upper bound of the memory usage (in bytes)
        """  

        subtotal, gpu_vram_par, gpu_vram_opt, gpu_vram_grad, \
            gpu_vram_act, gpu_vram_outputs, gpu_vram_additional = self._calc_subtotal()
    
        # Apply some proportion of overhead to get the final memory calculations
        results, overhead = self._apply_overhead(subtotal)

        # Print out details of the memory breakdown and recommendations based
        # on the user's given hardware. 
        if self.verbose > 1:
            self._print_results(results, overhead, gpu_vram_par,
                                gpu_vram_opt, gpu_vram_grad, gpu_vram_act,
                                gpu_vram_outputs, gpu_vram_additional)
        if self.verbose > 0:
            self._print_tips(results)

        # Return the lower bound, estimated value, and upper bound 
        return results 

    def _find_valid_layers(self):
        """
        Check to see which terms need to be included in the search for valid layers
        """
        target_terms = MODEL_CONFIGS['default']['patterns']
        lowered_model_path = self.model_path.lower()
        if lowered_model_path.find('phi-3') > -1:
            target_terms = MODEL_CONFIGS['phi3']['patterns']
            return target_terms
        for key in MODEL_CONFIGS.keys():
            if lowered_model_path.find(key.lower()) > -1:
                target_terms = MODEL_CONFIGS[key]['patterns']
                return target_terms
        return target_terms


class LoRAEstimator(BasicEstimator):
    """
    An estimator for the memory usage of an LLM trained via LoRA.
    Subclasses the BasicEstimator class.

    batch_size is the micro-batch per process (lora_sft's micro_batch_size), not the
    effective batch size: gradient accumulation doesn't change the peak memory.

    Args (in addition to the BasicEstimator args):
        lora_r (int): The rank size of LoRA's approximation matrices
        enable_model_splitting (bool): If true, estimate for model splitting
                        (device_map="balanced"): work out which GPU holds each layer
                        and return the busiest GPU's memory (default: False). If false
                        and num_gpus > 1, estimate for DDP: one full replica per GPU.
        target_modules (list[str] | None): Module names LoRA is applied to. None (default)
                        matches Unsloth's default: every linear layer in the text decoder.
        fused_loss (bool | None): Whether the loss fuses the lm_head into the cross-entropy, so the
                        logits are never materialized. None (default) decides from measurements:
                        fused, except for the model types in MATERIALIZES_LOGITS. Set it for
                        models whose behavior hasn't been measured yet.
    """
    # QLoRA loads the quantized weights at 4 bits; used when working out the GPU placement
    quantize_4bit: bool = False

    @override
    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        batch_size: int | None = None,
        max_seq_len: int | None = None,
        use_liger: bool = False,
        verbose: int = 1,
        trust_remote_code: bool = False,
        lora_r: int = 32,
        enable_model_splitting: bool = False,
        target_modules: list[str] | None = None,
        fused_loss: bool | None = None,
    ):
        self.enable_model_splitting = enable_model_splitting  # used by _resolve_tokens_per_gpu
        super().__init__(num_gpus=num_gpus,
                        gpu_memory=gpu_memory,
                        model_path=model_path,
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                        max_tokens_per_gpu=None,
                        use_liger=use_liger,
                        verbose=verbose,
                        trust_remote_code=trust_remote_code)

        # LoRA estimates need a looser lower bound but can afford a tighter upper bound
        self.LOW_MULTIPLIER = 0.95
        self.MEDIUM_MULTIPLIER = 1.1
        self.HIGH_MULTIPLIER = 1.2

        # LoRA stores everything in Float16
        self.main_dtype_bytes = FLOAT16_BYTES_N
        self.model_bytes = FLOAT16_BYTES_N

        self.lora_r = lora_r
        self.target_modules = target_modules
        self.fused_loss = fused_loss

        # Determine the number of parameters needed by LoRA. The CSV cache only holds the
        # default (all linear layers), so custom target_modules need the meta model, and so
        # does model splitting, to work out which GPU holds each layer, and QLoRA, to work out
        # which weights bitsandbytes quantizes.
        cached = getattr(self.model, 'lora_weight_size', None) if self.found_model else None
        has_cached = cached is not None and not pd.isna(cached)
        needs_structure = target_modules is not None or enable_model_splitting or self.quantize_4bit
        if self.found_model and (needs_structure or not has_cached):
            self._load_model("The estimate needs the model structure.", trust_remote_code=trust_remote_code)
            if not has_cached:
                self._update_model_storage()
        if target_modules is None and has_cached:
            self.lora_weight_size = int(cached)
        else:
            self.lora_weight_size = self._calc_lora_weight_size(target_modules)
        self.AB_params = self.lora_weight_size * lora_r


    @override
    def _resolve_tokens_per_gpu(self, batch_size: int | None,
                                max_seq_len: int | None, max_tokens_per_gpu: int | None):
        """
        Find the number of tokens that will be processed by each GPU.
        For LoRA, batch_size is the micro-batch (lora_sft's micro_batch_size, i.e. the
        per_device_train_batch_size), so each GPU processes batch_size * max_seq_len tokens in
        both modes: with model splitting the one process's micro-batch passes through every GPU,
        and without it (DDP) every process runs its own micro-batch.
        """
        if batch_size is None or max_seq_len is None:
            raise ValueError("batch_size and max_seq_len must be provided")
        self.tokens_per_gpu = batch_size * max_seq_len


    def _replica_divisor(self) -> int:
        """
        How many GPUs share one copy of the model. With model splitting the GPUs share one
        copy; without it (DDP) every GPU holds a full replica with its own LoRA and optimizer state.

        Returns:
            int: num_gpus with model splitting, otherwise 1
        """
        return self.num_gpus if self.enable_model_splitting else 1


    @override
    def _calc_model_params(self):
        """
        Calcuate the VRAM for storing the model parameters.
        For LORA, we include the memory used by the low rank matrices
        """
        return ((self.num_params * self.model_bytes) + (self.AB_params * FLOAT32_BYTES_N)) / self._replica_divisor()


    @override
    def _calc_gradients(self):
        """
        Calcuate the VRAM for storing the gradients.
        In LoRA, this value is based on number of parameters in the low rank matrices
        """
        return (self.main_dtype_bytes * self.AB_params / self._replica_divisor())


    @override
    def _calc_optimizer(self):
        """
        Calculate the VRAM for storing the optimizer states
        In LoRA, this value is based on number of parameters in the low rank matrices
        """
        return (self.main_dtype_bytes * self.AB_params * self.opt_params / self._replica_divisor())


    def _calc_split_placement(self) -> tuple[dict[str, int | str], dict[int | str, float]] | None:
        """
        Work out which device holds each module under device_map="balanced", using the same
        transformers helpers that from_pretrained runs, on the meta model.

        transformers' get_balanced_memory is used rather than accelerate's: accelerate sizes the
        per-GPU buffer from the first no-split block, transformers from the largest one, which
        matters for models mixing block types (e.g. Nemotron's MoE and Mamba layers).

        Returns:
            tuple | None: (device_map, loaded weight bytes per device), or None if this
                          transformers version doesn't provide the helpers or accelerate
                          (which they need) isn't installed
        """
        from transformers.utils import is_accelerate_available
        if not is_accelerate_available():
            return None
        try:
            from transformers.integrations.accelerate import (
                compute_module_sizes,
                get_balanced_memory,
                infer_auto_device_map,
            )
        except ImportError:
            return None

        sizer = _PlacementSizer(self.model, self.quantize_4bit)
        no_split = self.model._no_split_modules
        max_memory = {gpu: self.gpu_memory for gpu in range(self.num_gpus)} | {"cpu": 10 * 1024**4}
        # accelerate warns about every GPU in max_memory that this machine doesn't have,
        # but the estimate is usually run away from the training hardware
        accelerate_logger = logging.getLogger("accelerate.utils.modeling")
        old_level = accelerate_logger.level
        accelerate_logger.setLevel(logging.ERROR)
        try:
            balanced = get_balanced_memory(self.model, max_memory=max_memory,
                                           no_split_module_classes=no_split, hf_quantizer=sizer)
            balanced = sizer.adjust_max_memory(balanced)
            device_map = infer_auto_device_map(self.model, max_memory=balanced,
                                               no_split_module_classes=no_split, hf_quantizer=sizer)
        finally:
            accelerate_logger.setLevel(old_level)
        module_sizes, _ = compute_module_sizes(self.model, sizer)
        weights: dict[int | str, float] = defaultdict(float)
        for module_name, device in device_map.items():
            weights[device] += module_sizes[module_name]
        return device_map, dict(weights)


    @staticmethod
    def _device_of(module_name: str, device_map: dict[str, int | str]) -> int | str | None:
        """
        Find the device a module was placed on.

        Args:
            module_name (str): The module's name
            device_map (dict[str, int | str]): Module name -> device, as from infer_auto_device_map

        Returns:
            int | str | None: The device of the most specific entry containing the module, or of the
                              module's first child if it was split across devices
        """
        best_len, device = -1, None
        for key, dev in device_map.items():
            if (key == "" or module_name == key or module_name.startswith(key + ".")) and len(key) > best_len:
                best_len, device = len(key), dev
        if device is None:
            device = next((dev for key, dev in device_map.items() if key.startswith(module_name + ".")), None)
        return device


    def _calc_mlp_width(self) -> int:
        """
        The MLP width a token goes through in one decoder layer: intermediate_size for dense
        models, or experts per token x expert width + shared expert width for MoE models.

        Returns:
            int: The MLP width
        """
        text_config = self.model.config.get_text_config()

        def first(*names: str) -> int | None:
            """The first of these text config attributes that is set (the largest, if per-layer)."""
            for name in names:
                value = getattr(text_config, name, None)
                if value:
                    return max(value) if isinstance(value, (list, tuple)) else value
            return None

        experts_per_token = first("num_experts_per_tok")
        if experts_per_token and first("n_routed_experts", "num_experts", "num_local_experts"):
            expert_width = first("moe_intermediate_size", "expert_intermediate_size", "intermediate_size")
            shared_width = (first("moe_shared_expert_intermediate_size", "shared_expert_intermediate_size")
                            or (first("n_shared_experts") or 0) * expert_width)
            return experts_per_token * expert_width + shared_width
        return first("intermediate_size") or 4 * self.hidden_size


    def _materializes_logits(self) -> bool:
        """
        Whether training materializes the output logits. Unsloth fuses the lm_head into the
        cross-entropy loss by default, so the logits never exist, except for the model types
        in MATERIALIZES_LOGITS, whose lm_head was observed to run every step.

        Returns:
            bool: True if the logits are materialized
        """
        if self.fused_loss is not None:
            return not self.fused_loss
        if self.use_liger:  # Liger's loss is a fused linear + cross-entropy too
            return False
        # The CSV cache doesn't store the model type, so fall back to the (small) config file
        config = getattr(self.model, 'config', None) or \
            AutoConfig.from_pretrained(self.model_path, trust_remote_code=self.trust_remote_code)
        model_types = (config.model_type, config.get_text_config().model_type)
        return any(model_type.startswith(MATERIALIZES_LOGITS) for model_type in model_types)


    def _calc_logits(self) -> int:
        """
        Memory for the output logits of one GPU's tokens, or 0 if they're never materialized
        (see _materializes_logits).

        Returns:
            int: The logits' bytes: fp32 logits plus a bf16 copy
        """
        if not self._materializes_logits():
            return 0
        return int(self.tokens_per_gpu * self._get_vocab_size() * (FLOAT32_BYTES_N + FLOAT16_BYTES_N))


    def _calc_lm_head_workspace(self) -> int:
        """
        Extra memory the loss needs next to the lm_head. None for LoRA; see QLoRAEstimator.

        Returns:
            int: The workspace's bytes
        """
        return 0


    @override
    def _calc_outputs(self):
        """
        Calculate the VRAM for storing the output logits and the loss's lm_head workspace.
        Without model splitting every GPU runs its own lm_head on its own tokens.
        """
        return self._calc_logits() + self._calc_lm_head_workspace()


    def _calc_split_subtotal(self):
        """
        Estimate each GPU's memory under model splitting and return the busiest GPU's.

        device_map="balanced" doesn't divide the model evenly, and the lm_head lives on a single
        GPU, so each GPU gets its own estimate:
            weights on the GPU (bf16, or 4-bit for QLoRA)
            + LoRA params on the GPU x 16 bytes (fp32 weight, gradient and 2 AdamW moments)
            + one decoder layer's working set: 30 bytes x MLP width x tokens. Not divided by the
              number of GPUs, and not multiplied by the number of layers.
            + logits, on the lm_head GPU only, and only if they're materialized (_calc_logits),
              plus QLoRA's lm_head workspace there (_calc_lm_head_workspace).
        Falls back to an even split if transformers can't work out the placement.

        Returns:
            The busiest GPU's subtotal and its components, in the same order as _calc_subtotal
        """
        blocks = self._find_decoder_blocks()
        lora_per_layer = self._calc_lora_weight_size_per_layer(self.target_modules)
        head = self.model.get_output_embeddings()
        head_name = next((name for name, module in self.model.named_modules() if module is head), None)

        placement = self._calc_split_placement()
        lora_on: dict[int | str, int] = defaultdict(int)
        if placement is not None:
            device_map, weights = placement
            gpus = sorted(device for device in weights if isinstance(device, int))
            for layer, size in lora_per_layer.items():
                lora_on[self._device_of(f"{blocks}.{layer}", device_map)] += size
            head_gpu = self._device_of(head_name, device_map) if head_name else gpus[-1]
            if any(not isinstance(device, int) for device in weights):
                warnings.warn("The model doesn't fit on " + str(self.num_gpus) + " GPUs: "
                              "device_map=\"balanced\" would offload part of it to the CPU or disk.")
        else:
            warnings.warn("Could not work out the device_map=\"balanced\" placement with this "
                          "transformers version. Assuming the model is split evenly across GPUs.")
            gpus = list(range(self.num_gpus))
            total = _PlacementSizer(self.model, self.quantize_4bit).total_size(self.model)
            weights = {gpu: total / self.num_gpus for gpu in gpus}
            lora_on = {gpu: sum(lora_per_layer.values()) / self.num_gpus for gpu in gpus}
            head_gpu = gpus[-1]

        working_set = WS_BYTES_PER_MLP_UNIT * self._calc_mlp_width() * self.tokens_per_gpu
        logits = self._calc_logits() + self._calc_lm_head_workspace()

        # Components per GPU: (params, optimizer, gradients, activations, outputs).
        # Each LoRA parameter takes 16 bytes: fp32 weight (4) + 2 fp32 AdamW moments (8) + fp32 gradient (4)
        self.split_components: dict[int, tuple[float, ...]] = {}
        for gpu in gpus:
            lora_params = lora_on.get(gpu, 0) * self.lora_r
            self.split_components[gpu] = (
                weights[gpu] + lora_params * FLOAT32_BYTES_N,
                lora_params * FLOAT32_BYTES_N * ADAMW_PARAMS_N,
                lora_params * FLOAT32_BYTES_N,
                working_set,
                logits if gpu == head_gpu else 0,
            )
        busiest = max(gpus, key=lambda gpu: sum(self.split_components[gpu]))
        par, opt, grad, act, outputs = self.split_components[busiest]
        subtotal = par + opt + grad + act + outputs
        return subtotal, par, opt, grad, act, outputs, 0


    @override
    def _calc_subtotal(self):
        """
        Calculate the amount of memory expected before applying overhead.
        With model splitting, this is the busiest GPU's memory (see _calc_split_subtotal).
        """
        if self.enable_model_splitting:
            return self._calc_split_subtotal()

        # Calculate each piece of memory to be factored into the estimation
        gpu_vram_par = self._calc_model_params()
        gpu_vram_opt: int = self._calc_optimizer()
        gpu_vram_grad: int = self._calc_gradients()
        gpu_vram_act: int = self._calc_intermediate_activations()
        gpu_vram_outputs: int = self._calc_outputs()
        gpu_vram_additional: int = self._calc_additional()

        # The memory is bounded by either the size of the outputs or the size of
        # the gradients, whichever is higher (these tensors are NOT allocated simultaneously)
        if gpu_vram_grad > gpu_vram_outputs:
            if self.verbose > 0 and gpu_vram_outputs > 0:  # 0 when the loss fuses away the logits
                print("NOTE: The peak memory allocation of the output tensors is " + ROUNDER(gpu_vram_outputs) + " GB.")
                print("However, in practice, this allocation does not impact the peak memory usage since the gradient allocation is higher.")
                print("To reduce memory usage, focus on reducing the gradient allocation first.\n")
            gpu_vram_outputs = 0
        else:
            if self.verbose > 0:
                print("NOTE: The peak memory allocation of the gradient tensors is " + ROUNDER(gpu_vram_grad) + " GB.")
                print("However, in practice, this allocation does not impact the peak memory usage since the output allocation is higher.")
                print("To reduce memory usage, focus on reducing the output allocation first.\n")
            gpu_vram_grad = 0

        # Sum up each proposed amount of memory
        subtotal: int = gpu_vram_par + gpu_vram_opt + gpu_vram_grad + \
                        gpu_vram_act + gpu_vram_outputs + gpu_vram_additional

        return subtotal, gpu_vram_par, gpu_vram_opt, gpu_vram_grad, \
            gpu_vram_act, gpu_vram_outputs, gpu_vram_additional

    def _print_results(self, results, overhead, gpu_vram_par, gpu_vram_opt,
                        gpu_vram_grad, gpu_vram_act, gpu_vram_outputs, gpu_vram_additional):
        if self.verbose > 0:
            print("NOTE: Due to its memory efficiency, " + \
                    "LoRA's lower bound estimate is lower than the basic sum of the components.\n")
        if self.enable_model_splitting:
            print("With model splitting, the GPUs don't use the same amount of memory. Before overhead:")
            for gpu, components in self.split_components.items():
                print("  GPU " + str(gpu) + ": " + ROUNDER(sum(components)) + " GB")
            print("The estimate below is for the busiest GPU.\n")
        super()._print_results(results, overhead, gpu_vram_par, gpu_vram_opt,
                        gpu_vram_grad, gpu_vram_act, gpu_vram_outputs, gpu_vram_additional)


class QLoRAEstimator(LoRAEstimator):
    """
    An estimator for the memory usage of an LLM trained via QLoRA.
    It's the same as LoRA, except that bitsandbytes loads the nn.Linear weights in 4 bits.
    Everything else (embeddings, lm_head, norms, vision towers, fused 3D expert tensors)
    stays in bf16, so MoE models with fused experts save much less than num_params x 0.5.
    """
    quantize_4bit: bool = True
    @override
    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        batch_size: int | None = None,
        max_seq_len: int | None = None,
        use_liger: bool = False,
        verbose: int = 1,
        trust_remote_code: bool = False,
        lora_r: int = 32,
        enable_model_splitting: bool = False,
        target_modules: list[str] | None = None,
        fused_loss: bool | None = None,
    ):
        super().__init__(num_gpus=num_gpus,
                        gpu_memory=gpu_memory,
                        model_path=model_path,
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                        use_liger=use_liger,
                        verbose=verbose,
                        trust_remote_code=trust_remote_code,
                        lora_r=lora_r,
                        enable_model_splitting=enable_model_splitting,
                        target_modules=target_modules,
                        fused_loss=fused_loss)
        self.HIGH_MULTIPLIER = 1.3 # Use a looser upper bound
        self.model_bytes = FLOAT4_BYTES_N # The quantized weights are stored in 4 bits
        # The size bitsandbytes actually loads: the same sizes transformers uses for the placement
        self.loaded_weight_bytes = _PlacementSizer(self.model, quantize_4bit=True).total_size(self.model)


    @override
    def _calc_model_params(self):
        """
        Calculate the VRAM for storing the model parameters: the weights as bitsandbytes loads
        them (4-bit nn.Linear weights, everything else bf16) plus the fp32 LoRA matrices.
        """
        return (self.loaded_weight_bytes + self.AB_params * FLOAT32_BYTES_N) / self._replica_divisor()


    @override
    def _calc_lm_head_workspace(self) -> int:
        """
        With the fused loss, an untied lm_head needs a workspace the size of a bf16 copy of
        itself on its GPU. Measured on Muse Glimmer 30B (PR #163 review).

        Returns:
            int: lm_head parameters x 2 bytes for untied models with a fused loss, otherwise 0
        """
        head = self.model.get_output_embeddings()
        if head is None or self._materializes_logits() or self.model.config.get_text_config().tie_word_embeddings:
            return 0
        return head.weight.numel() * FLOAT16_BYTES_N


class OSFTEstimator(BasicEstimator):
    """
    An estimator for the memory usage of an LLM trained via OSFT. 
    Subclasses the BasicEstimator class. Note that OSFT uses Liger Kernels, so use_liger
    is set to True by default. 

    Args (in addition to the BasicEstimator args):
        unfreeze_rank_ratio (float): The portion of the weight matrix that is unfrozen
                                    during OSFT.
    """
    @override
    def __init__(
        self,
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        batch_size: int | None = None,
        max_seq_len: int | None = None,
        max_tokens_per_gpu: int | None = None,
        use_liger: bool = True,
        verbose: int = 1,
        trust_remote_code: bool = False,
        unfreeze_rank_ratio: float = 0.25,
        count_fsdp_gather: bool = False,
    ):
        super().__init__(num_gpus=num_gpus,
                        gpu_memory=gpu_memory,
                        model_path=model_path,
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                        max_tokens_per_gpu=max_tokens_per_gpu,
                        use_liger=use_liger,
                        verbose=verbose,
                        trust_remote_code=trust_remote_code,
                        count_fsdp_gather=count_fsdp_gather)

        self.output_constant = 7/3
        self.unfreeze_rank_ratio = unfreeze_rank_ratio
        if not (0.0 <= self.unfreeze_rank_ratio <= 1.0):
            raise ValueError("Ratio must be in the range [0, 1]")

        # Check to see which terms need to be included in the search for valid layers
        self.target_terms = self._find_valid_layers()

        # Determine the number of additional OSFT params
        if self.found_model:
            if hasattr(self.model, 'osft_params'):
                self.osft_params = self.model.osft_params
            else:
                self.found_model = False
                self._load_model("Could not find the number of OSFT parameters in the CSV cache.", 
                                trust_remote_code=trust_remote_code)
                self.osft_params = self._calc_osft_params()
                self._update_model_storage()
        else:
            self.osft_params = self._calc_osft_params()


    @override
    def _calc_model_params(self):
        """
        Override the model parameter calculation by calculating based on OSFT's parameters
        """
        return self.osft_params / self.num_gpus 


    @override
    def _calc_gradients(self):
        """
        Calculate the VRAM for storing the gradients.
        In OSFT, only the unfrozen gradients are stored in memory,
        so we factor in the unfreeze rank ratio
        """
        return self.num_trainable_params * self.main_dtype_bytes * self.unfreeze_rank_ratio / self.num_gpus 

    
    @override
    def _calc_intermediate_activations(self):
        """
        Calculate the VRAM for storing the intermediate activations.
        In OSFT, the intermediate activations are only stored for the unfrozen parameters,
        so we factor in the unfreeze rank ratio
        """
        return (self.tokens_per_gpu * self.main_dtype_bytes  * self.num_layers * self.hidden_size)  * self.unfreeze_rank_ratio


class OSFTEstimatorExperimental(OSFTEstimator):
    def __init__(self,
                num_gpus: int = 8,
                gpu_memory: int = 85899345920,
                model_path: str = "ibm-granite/granite-3.3-8b-instruct",
                batch_size: int | None = None,
                max_seq_len: int | None = None,
                max_tokens_per_gpu: int | None = None,
                use_liger: bool = True,
                verbose: int = 1,
                trust_remote_code: bool = False,
                unfreeze_rank_ratio: float = 0.25,
                count_fsdp_gather: bool = False,
                ):
        reason = "All experimental functionality has been moved into the main OSFTEstimator class, " + \
            "please use that instead.\nIf you are seeing this by using the 'osft-e' option in the " + \
            "estimate function, please use 'osft' instead."
        warnings.warn(reason, category=DeprecationWarning, stacklevel=2)
        
        super().__init__(num_gpus=num_gpus,
                        gpu_memory=gpu_memory,
                        model_path=model_path,
                        batch_size=batch_size,
                        max_seq_len=max_seq_len,
                        max_tokens_per_gpu=max_tokens_per_gpu,
                        use_liger=use_liger,
                        verbose=verbose,
                        trust_remote_code=trust_remote_code,
                        unfreeze_rank_ratio=unfreeze_rank_ratio,
                        count_fsdp_gather=count_fsdp_gather)

def estimate(
        training_method: str = "sft",
        num_gpus: int = 8,
        gpu_memory: int = 85899345920,
        model_path: str = "ibm-granite/granite-3.3-8b-instruct",
        batch_size: int | None = None,
        max_seq_len: int | None = None,
        max_tokens_per_gpu: int | None = None,
        use_liger: bool = None,
        verbose: int = 1,
        trust_remote_code: bool = False,
        unfreeze_rank_ratio: float = 0.25,
        lora_r: int = 32,
        count_fsdp_gather: bool = False,
        enable_model_splitting: bool = False,
        target_modules: list[str] | None = None,
        fused_loss: bool | None = None,
    ):
    """
    Convenience function for performing estimation

    Args:
        training_method (str): The training method to estimate the memory for. 
                                Set to "osft" to estimate for OSFT,
                                By default, SFT is assumed.
        num_gpus (int): Number of GPUs to use for training (default: 8)
        gpu_memory (int): The VRAM of each GPU in bytes (default: 85899345920 for 80 GB)
        model_path (str): HuggingFace model path to the model to fine-tune
                        (default: "ibm-granite/granite-3.3-8b-instruct")
        batch_size (int): The number of samples in a minibatch that the model
                            has to see before backpropping.
        max_seq_len (int): Maximum sequence length of dataset samples 
        max_tokens_per_gpu (int): The maximum number of tokens that can be placed
                                on a single GPU during each mini-batch.
        use_liger (bool): If true, estimate assuming Liger Kernels are used.
        verbose (int): The level of verbosity to print out. Set to 0 for no printing,
                        set to 1 to print out only hardware recommendations,
                        set to 2 for a detailed memory breakdown.
        unfreeze_rank_ratio (float): The portion of the weight matrix that is unfrozen
                                    during OSFT.
        count_fsdp_gather (bool): sft/osft only. Add the per-GPU FSDP unit gather when num_gpus > 1.
        enable_model_splitting (bool): lora/qlora only. Estimate for model splitting
                                    (device_map="balanced"). If false and num_gpus > 1, estimate
                                    for DDP (one full replica per GPU). For lora/qlora, batch_size
                                    is the micro-batch per process (lora_sft's micro_batch_size)
                                    in both modes.
        target_modules (list[str] | None): lora/qlora only. Module names LoRA is applied to.
                                    None (default) means every linear layer in the text decoder.
        fused_loss (bool | None): lora/qlora only. Whether the loss fuses away the logits.
                                    None (default) decides from the model type (see MATERIALIZES_LOGITS).

    Return:
        A tuple containing three values:
            lower_bound (int): The lower bound of the memory usage (in bytes)
            expected (int): The expected amount of memory usage (in bytes)
            upper_bound (int): The upper bound of the memory usage (in bytes)
    """
    
    if training_method.lower() == "osft":
        if use_liger is None:
            use_liger = True
        estimator = OSFTEstimator(num_gpus=num_gpus,
                                    gpu_memory=gpu_memory,
                                    model_path=model_path,
                                    batch_size=batch_size,
                                    max_seq_len=max_seq_len,
                                    max_tokens_per_gpu=max_tokens_per_gpu,
                                    use_liger=use_liger,
                                    verbose=verbose,
                                    trust_remote_code=trust_remote_code,
                                    unfreeze_rank_ratio=unfreeze_rank_ratio,
                                    count_fsdp_gather=count_fsdp_gather)
    elif training_method.lower() == "osft-e":
        if use_liger is None:
            use_liger = True
        estimator = OSFTEstimatorExperimental(num_gpus=num_gpus,
                                                gpu_memory=gpu_memory,
                                                model_path=model_path,
                                                batch_size=batch_size,
                                                max_seq_len=max_seq_len,
                                                max_tokens_per_gpu=max_tokens_per_gpu,
                                                use_liger=use_liger,
                                                verbose=verbose,
                                                trust_remote_code=trust_remote_code,
                                                unfreeze_rank_ratio=unfreeze_rank_ratio,
                                                count_fsdp_gather=count_fsdp_gather)
    elif training_method.lower() == "lora":
        if use_liger is None:
            use_liger = False
        estimator = LoRAEstimator(num_gpus=num_gpus,
                                    gpu_memory=gpu_memory,
                                    model_path=model_path,
                                    batch_size=batch_size,
                                    max_seq_len=max_seq_len,
                                    use_liger=use_liger,
                                    verbose=verbose,
                                    trust_remote_code=trust_remote_code,
                                    lora_r=lora_r,
                                    enable_model_splitting=enable_model_splitting,
                                    target_modules=target_modules,
                                    fused_loss=fused_loss)
    elif training_method.lower() == "qlora":
        if use_liger is None:
            use_liger = False
        estimator = QLoRAEstimator(num_gpus=num_gpus,
                                    gpu_memory=gpu_memory,
                                    model_path=model_path,
                                    batch_size=batch_size,
                                    max_seq_len=max_seq_len,
                                    use_liger=use_liger,
                                    verbose=verbose,
                                    trust_remote_code=trust_remote_code,
                                    lora_r=lora_r,
                                    enable_model_splitting=enable_model_splitting,
                                    target_modules=target_modules,
                                    fused_loss=fused_loss)
    else:
        if use_liger is None:
            use_liger = False
        estimator = BasicEstimator(num_gpus=num_gpus,
                                    gpu_memory=gpu_memory,
                                    model_path=model_path,
                                    batch_size=batch_size,
                                    max_seq_len=max_seq_len,
                                    max_tokens_per_gpu=max_tokens_per_gpu,
                                    use_liger=use_liger,
                                    verbose=verbose,
                                    trust_remote_code=trust_remote_code,
                                    count_fsdp_gather=count_fsdp_gather)

    
    return estimator.estimate()
