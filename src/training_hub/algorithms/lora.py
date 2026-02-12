import json
import os
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path

from . import Algorithm, Backend, AlgorithmRegistry
from .sft import SFTAlgorithm
from .peft_extender import LoRAPEFTExtender, get_lora_parameters, apply_lora_defaults
from training_hub import utils

# Import TrainerCallback for JSONL logging - handle gracefully if not available
try:
    from transformers import TrainerCallback
except ImportError:
    # Fallback if transformers not available - will be checked later in execute_training
    TrainerCallback = object


class JSONLLoggingCallback(TrainerCallback):
    """
    Custom callback to write training metrics to JSONL format.

    This provides consistency with SFT/OSFT backends which output training_log.jsonl.
    Writes live metrics during training, not just at the end.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the callback.

        Args:
            output_dir: Directory where training_log.jsonl will be written
        """
        self.output_file = os.path.join(output_dir, 'training_log.jsonl')
        # Ensure directory exists
        os.makedirs(output_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        """
        Called when trainer logs metrics.

        Writes entries in the same format as SFT/OSFT:
        {"step": 1, "loss": 4.2727, "epoch": 0.015625, "learning_rate": 2e-6}
        """
        if logs is None:
            return

        # Create entry with consistent format
        entry = {
            "step": state.global_step,
            "epoch": state.epoch,
        }

        # Add metrics from logs (loss, learning_rate, etc.)
        for key, value in logs.items():
            # Convert numpy/torch values to Python types for JSON serialization
            if hasattr(value, 'item'):
                entry[key] = value.item()
            else:
                entry[key] = value

        # Write to JSONL file
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


class UnslothLoRABackend(Backend):
    """Unsloth backend for LoRA algorithm with performance optimizations."""

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute LoRA training using Unsloth optimizations."""
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig
            from transformers import TrainingArguments
        except ImportError as e:
            # Handle common import issues with specific guidance
            error_msg = str(e).lower()
            if "unsloth" in error_msg:
                raise ImportError(
                    "Unsloth is not available. Install with:\n"
                    "pip install 'training-hub[lora]'"
                ) from e
            elif "trl" in error_msg:
                raise ImportError(
                    "TRL is required for Unsloth LoRA training. Install with:\n"
                    "pip install 'training-hub[lora]'"
                ) from e
            else:
                raise ImportError(
                    f"Failed to import dependencies for Unsloth backend: {e}\n"
                    "Install LoRA dependencies with: pip install 'training-hub[lora]'"
                ) from e

        # Use all parameters as training parameters
        # Note: Torchrun parameters (nproc_per_node, etc.) are handled by the torchrun launcher,
        # not by the Python training code. The training code auto-detects distributed environment
        # via environment variables (WORLD_SIZE, LOCAL_RANK, etc.) set by torchrun.
        training_params = algorithm_params

        # Unsloth multi-GPU setup: Let Accelerate/torchrun handle distributed training
        # No custom distributed initialization needed - Unsloth works with standard PyTorch DDP

        # Load model with Unsloth optimizations
        model, tokenizer = self._load_unsloth_model(training_params)

        # Apply LoRA with Unsloth optimizations
        model = self._apply_lora_config(model, training_params)

        # Prepare dataset
        train_dataset = self._prepare_dataset(training_params, tokenizer)

        # Configure training arguments
        training_args = self._build_training_args(training_params)

        # Use the same trainer configuration for all dataset types since we pre-process in _prepare_dataset
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            max_seq_length=training_params.get('max_seq_len', 2048),
            packing=training_params.get('sample_packing', True),
        )

        # Add custom callback for JSONL logging (consistent with SFT/OSFT backends)
        jsonl_callback = JSONLLoggingCallback(training_params['ckpt_output_dir'])
        trainer.add_callback(jsonl_callback)

        # Execute training with error handling for known Unsloth issues
        try:
            trainer.train()
        except AssertionError as e:
            if "wrong number of dimensions" in str(e) and "int8_mixed_scaled_mm" in str(e):
                # Known Unsloth 8-bit quantization issue: https://github.com/unslothai/unsloth/issues/3501
                raise RuntimeError(
                    f"❌ Unsloth 8-bit quantization compatibility issue detected.\n"
                    f"This is a known issue with Unsloth + 8-bit quantization + some model architectures.\n"
                    f"See: https://github.com/unslothai/unsloth/issues/3501\n\n"
                    f"💡 Recommended solutions:\n"
                    f"• Try 4-bit quantization instead: load_in_4bit=True, load_in_8bit=False\n"
                    f"• Use standard training without quantization: load_in_4bit=False, load_in_8bit=False\n"
                    f"• Update Unsloth to the latest version in case this issue is fixed\n\n"
                    f"Original error: {e}"
                ) from e
            else:
                # Re-raise other AssertionErrors
                raise

        # Save model
        if training_params.get('save_model', True):
            trainer.save_model(training_params['ckpt_output_dir'])
            tokenizer.save_pretrained(training_params['ckpt_output_dir'])

        return {
            'model': model,
            'tokenizer': tokenizer,
            'trainer': trainer
        }

    def _load_unsloth_model(self, params: Dict[str, Any]) -> tuple:
        """Load model with Unsloth optimizations."""
        from unsloth import FastLanguageModel

        # Determine quantization settings - only use if explicitly requested by user
        load_in_4bit = params.get('load_in_4bit', False)
        load_in_8bit = params.get('load_in_8bit', False)

        # Handle device placement for multi-GPU training
        device_map_config = {}
        if params.get('enable_model_splitting', False):
            # Use balanced device mapping for large models
            device_map_config['device_map'] = "balanced"
        elif 'LOCAL_RANK' in os.environ:
            # For DDP training, explicitly set device based on LOCAL_RANK
            import torch
            local_rank = int(os.environ['LOCAL_RANK'])
            torch.cuda.set_device(local_rank)
            device_map_config['device_map'] = {"": local_rank}

        # Configure quantization options - only add BitsAndBytes parameters if explicitly provided
        quantization_kwargs = {}
        if load_in_4bit:
            # Only add BitsAndBytes-specific parameters if they were explicitly set by the user
            if 'bnb_4bit_quant_type' in params:
                quantization_kwargs['bnb_4bit_quant_type'] = params['bnb_4bit_quant_type']
            if 'bnb_4bit_compute_dtype' in params:
                quantization_kwargs['bnb_4bit_compute_dtype'] = params['bnb_4bit_compute_dtype']
            if 'bnb_4bit_use_double_quant' in params:
                quantization_kwargs['bnb_4bit_use_double_quant'] = params['bnb_4bit_use_double_quant']

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=params['model_path'],
            max_seq_length=params.get('max_seq_len', 2048),
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
            **quantization_kwargs,
            **device_map_config,
            # Additional Unsloth optimizations
            # trust_remote_code=params.get('trust_remote_code', False),
        )

        # Use default tokenizer chat template

        return model, tokenizer

    def _apply_lora_config(self, model, params: Dict[str, Any]):
        """Apply LoRA configuration using Unsloth optimizations."""
        from unsloth import FastLanguageModel

        # Handle target_modules - use default if None or not specified
        target_modules = params.get('target_modules')
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        # Build LoRA config parameters
        lora_config = {
            'r': params.get('lora_r', 16),
            'target_modules': target_modules,
            'lora_alpha': params.get('lora_alpha', 32),
            'lora_dropout': params.get('lora_dropout', 0.0),  # 0.0 is optimized for Unsloth
            'bias': "none",
            'use_gradient_checkpointing': "unsloth",  # Unsloth's optimized gradient checkpointing
            'random_state': params.get('seed', 3407),
            'use_rslora': params.get('use_rslora', False),
        }

        # Add optional advanced parameters if specified
        if params.get('loftq_config') is not None:
            lora_config['loftq_config'] = params.get('loftq_config')
        if params.get('use_dora') is not None:
            lora_config['use_dora'] = params.get('use_dora')
        if params.get('init_lora_weights') is not None:
            lora_config['init_lora_weights'] = params.get('init_lora_weights')
        if params.get('rank_pattern') is not None:
            lora_config['rank_pattern'] = params.get('rank_pattern')
        if params.get('alpha_pattern') is not None:
            lora_config['alpha_pattern'] = params.get('alpha_pattern')

        model = FastLanguageModel.get_peft_model(model, **lora_config)

        return model

    def _prepare_dataset(self, params: Dict[str, Any], tokenizer) -> Any:
        """Prepare dataset for training."""
        from datasets import load_dataset

        # Load dataset
        if params['data_path'].endswith('.jsonl') or params['data_path'].endswith('.json'):
            dataset = load_dataset('json', data_files=params['data_path'], split='train')
        else:
            dataset = load_dataset(params['data_path'], split='train')

        # Handle different dataset formats
        dataset_type = params.get('dataset_type', 'chat_template')

        if dataset_type == 'chat_template':
            # Convert messages format using chat template
            messages_field = params.get('field_messages', 'messages')

            def format_chat_template(examples):
                # examples[messages_field] is a list of conversations (batched)
                texts = []
                for conversation in examples[messages_field]:
                    text = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                    texts.append(text)
                return {"text": texts}

            dataset = dataset.map(format_chat_template, batched=True)

        elif dataset_type == 'alpaca':
            # Convert alpaca format to text
            instruction_field = params.get('field_instruction', 'instruction')
            input_field = params.get('field_input', 'input')
            output_field = params.get('field_output', 'output')

            def format_alpaca(examples):
                texts = []
                for i in range(len(examples[instruction_field])):
                    instruction = examples[instruction_field][i]
                    input_text = examples.get(input_field, [''] * len(examples[instruction_field]))[i]
                    output = examples[output_field][i]

                    if input_text:
                        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                    else:
                        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    texts.append(text)
                return {"text": texts}

            dataset = dataset.map(format_alpaca, batched=True)

        elif dataset_type == 'passthrough':
            # Pass dataset through without any preprocessing - user handles all formatting
            # Useful for pre-tokenized data, custom formats, or when users want full control
            pass

        return dataset


    def _build_training_args(self, params: Dict[str, Any]):
        """Build training arguments for SFTTrainer using SFTConfig."""
        from trl import SFTConfig

        # Populate logging params from environment variables if not explicitly set
        if not params.get('mlflow_tracking_uri'):
            params['mlflow_tracking_uri'] = os.environ.get('MLFLOW_TRACKING_URI')
        if not params.get('mlflow_experiment_name'):
            params['mlflow_experiment_name'] = os.environ.get('MLFLOW_EXPERIMENT_NAME')
        if not params.get('mlflow_run_name'):
            params['mlflow_run_name'] = os.environ.get('MLFLOW_RUN_NAME')
        if not params.get('wandb_project'):
            params['wandb_project'] = os.environ.get('WANDB_PROJECT')
        if not params.get('wandb_entity'):
            params['wandb_entity'] = os.environ.get('WANDB_ENTITY')
        if not params.get('wandb_run_name'):
            params['wandb_run_name'] = os.environ.get('WANDB_RUN_NAME')

        # Calculate steps and batch sizes
        num_epochs = params.get('num_epochs', 3)

        # Determine actual number of GPUs being used
        import torch

        # If we're in a distributed environment, use world size
        if 'WORLD_SIZE' in os.environ:
            num_gpus = int(os.environ['WORLD_SIZE'])
        elif torch.cuda.is_available():
            # Otherwise use the specified nproc_per_node or 1
            num_gpus = params.get('nproc_per_node', 1)
            if isinstance(num_gpus, str):
                num_gpus = torch.cuda.device_count() if num_gpus == 'auto' else 1
            # But don't use more GPUs than we're actually planning to use
            if num_gpus > 1 and ('RANK' not in os.environ and 'LOCAL_RANK' not in os.environ):
                # Single process mode - only using 1 GPU
                num_gpus = 1
        else:
            num_gpus = 1

        # Handle batch size calculation: effective_batch_size = micro_batch_size * gradient_accumulation_steps * num_gpus
        effective_batch_size = params.get('effective_batch_size')
        gradient_accumulation_steps = params.get('gradient_accumulation_steps', 1)

        if effective_batch_size is not None:
            # Calculate micro_batch_size from effective_batch_size
            micro_batch_size = effective_batch_size // (gradient_accumulation_steps * num_gpus)
            micro_batch_size = max(1, micro_batch_size)
        else:
            # Use provided micro_batch_size or default
            micro_batch_size = params.get('micro_batch_size', 2)

        training_args = SFTConfig(
            output_dir=params['ckpt_output_dir'],
            num_train_epochs=num_epochs,
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=params.get('learning_rate', 2e-4),
            weight_decay=params.get('weight_decay', 0.01),
            fp16=params.get('fp16', False),
            bf16=params.get('bf16', True),
            max_grad_norm=params.get('max_grad_norm', 0.3),
            warmup_steps=params.get('warmup_steps', 10),
            lr_scheduler_type=params.get('lr_scheduler', 'linear'),

            # Logging
            logging_steps=params.get('logging_steps', 1),
            logging_dir=params.get('tensorboard_log_dir'),
            save_steps=params.get('save_steps', 500),
            eval_steps=params.get('eval_steps', 500),
            save_total_limit=params.get('save_total_limit', 3),

            # Performance optimizations
            dataloader_pin_memory=False,  # Unsloth recommendation
            remove_unused_columns=False,  # Required for custom datasets

            # Multi-GPU / DDP configuration (required by Unsloth for multi-GPU)
            ddp_find_unused_parameters=False,  # Required for Unsloth DDP

            # Chat template and conversation handling
            dataset_text_field="text",  # Use our preprocessed text field
            assistant_only_loss=True,  # Only train on assistant responses

            # Logging configuration - use loggers list if provided, otherwise fallback to wandb_project
            report_to=self._get_report_to(params),
            # Use wandb_run_name or mlflow_run_name for the HuggingFace Trainer run_name
            # (used by both wandb and MLflow integrations)
            run_name=params.get('wandb_run_name') or params.get('mlflow_run_name'),
        )

        # Set Weights & Biases project and entity via environment variables if provided
        if params.get('wandb_project'):
            os.environ['WANDB_PROJECT'] = params.get('wandb_project')
            if params.get('wandb_entity'):
                os.environ['WANDB_ENTITY'] = params.get('wandb_entity')

        # Set MLflow tracking URI and related config if provided
        if params.get('mlflow_tracking_uri'):
            os.environ['MLFLOW_TRACKING_URI'] = params.get('mlflow_tracking_uri')
        if params.get('mlflow_experiment_name'):
            os.environ['MLFLOW_EXPERIMENT_NAME'] = params.get('mlflow_experiment_name')
        if params.get('mlflow_run_name'):
            os.environ['MLFLOW_RUN_NAME'] = params.get('mlflow_run_name')

        return training_args

    def _get_report_to(self, params: Dict[str, Any]) -> Optional[Union[str, List[str]]]:
        """Determine which loggers to report to based on config params."""
        # Auto-detect loggers from their config parameters
        detected = []
        if (
            params.get('wandb_project')
            or params.get('wandb_run_name')
            or params.get('wandb_entity')
            or os.environ.get('WANDB_PROJECT')
        ):
            detected.append("wandb")
        if (
            params.get('mlflow_tracking_uri')
            or params.get('mlflow_experiment_name')
            or params.get('mlflow_run_name')
            or os.environ.get('MLFLOW_TRACKING_URI')
            or os.environ.get('MLFLOW_EXPERIMENT_NAME')
        ):
            detected.append("mlflow")
        if params.get('tensorboard_log_dir'):
            detected.append("tensorboard")

        if detected:
            return detected if len(detected) > 1 else detected[0]
        return "none"




class LoRASFTAlgorithm(Algorithm):
    """LoRA + SFT algorithm combining Supervised Fine-Tuning with LoRA parameter-efficient training."""

    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.config = kwargs
        self.peft_extender = LoRAPEFTExtender()

    def train(self,
              model_path: str,
              data_path: str,
              ckpt_output_dir: str,
              # SFT parameters (inherited from SFT algorithm)
              num_epochs: Optional[int] = None,
              effective_batch_size: Optional[int] = None,
              learning_rate: Optional[float] = None,
              max_seq_len: Optional[int] = None,
              max_tokens_per_gpu: Optional[int] = None,
              data_output_dir: Optional[str] = None,
              save_samples: Optional[int] = None,
              warmup_steps: Optional[int] = None,
              accelerate_full_state_at_epoch: Optional[bool] = None,
              checkpoint_at_epoch: Optional[bool] = None,
              # LoRA-specific parameters (from PEFT extender)
              lora_r: Optional[int] = None,
              lora_alpha: Optional[int] = None,
              lora_dropout: Optional[float] = None,
              target_modules: Optional[List[str]] = None,
              use_rslora: Optional[bool] = None,
              use_dora: Optional[bool] = None,
              init_lora_weights: Optional[Union[bool, str]] = None,
              rank_pattern: Optional[Dict[str, int]] = None,
              alpha_pattern: Optional[Dict[str, int]] = None,
              loftq_config: Optional[Dict[str, Any]] = None,
              # Quantization parameters (QLoRA)
              load_in_4bit: Optional[bool] = None,
              load_in_8bit: Optional[bool] = None,
              bnb_4bit_quant_type: Optional[str] = None,
              bnb_4bit_compute_dtype: Optional[str] = None,
              bnb_4bit_use_double_quant: Optional[bool] = None,
              # Additional training parameters (extending SFT)
              micro_batch_size: Optional[int] = None,
              gradient_accumulation_steps: Optional[int] = None,
              lr_scheduler: Optional[str] = None,
              weight_decay: Optional[float] = None,
              max_grad_norm: Optional[float] = None,
              # Optimization parameters
              flash_attention: Optional[bool] = None,
              sample_packing: Optional[bool] = None,
              bf16: Optional[bool] = None,
              fp16: Optional[bool] = None,
              tf32: Optional[bool] = None,
              # Saving and logging
              save_steps: Optional[int] = None,
              eval_steps: Optional[int] = None,
              logging_steps: Optional[int] = None,
              save_total_limit: Optional[int] = None,
              # Logging configuration
              wandb_project: Optional[str] = None,
              wandb_entity: Optional[str] = None,
              wandb_run_name: Optional[str] = None,
              tensorboard_log_dir: Optional[str] = None,
              mlflow_tracking_uri: Optional[str] = None,
              mlflow_experiment_name: Optional[str] = None,
              mlflow_run_name: Optional[str] = None,
              # Dataset format parameters
              dataset_type: Optional[str] = None,
              field_messages: Optional[str] = None,
              field_instruction: Optional[str] = None,
              field_input: Optional[str] = None,
              field_output: Optional[str] = None,
              # Distributed training parameters (inherited from SFT)
              nproc_per_node: Optional[Union[str, int]] = None,
              nnodes: Optional[int] = None,
              node_rank: Optional[int] = None,
              rdzv_id: Optional[Union[str, int]] = None,
              rdzv_endpoint: Optional[str] = None,
              master_addr: Optional[str] = None,
              master_port: Optional[int] = None,
              # Multi-GPU model splitting
              enable_model_splitting: Optional[bool] = None,
              **kwargs) -> Any:
        """Execute LoRA + SFT training combining supervised fine-tuning with LoRA parameter-efficient training.

        This method combines all SFT parameters with LoRA-specific parameters to enable
        parameter-efficient fine-tuning with the performance and flexibility of SFT.

        Args:
            model_path: Path to the model to fine-tune (local path or HuggingFace model ID)
            data_path: Path to the training data (JSON/JSONL format)
            ckpt_output_dir: Directory to save checkpoints and outputs

            SFT Parameters (inherited from SFT algorithm):
            num_epochs: Number of training epochs (default: 3)
            effective_batch_size: Effective batch size across all GPUs
            learning_rate: Learning rate (default: 2e-4)
            max_seq_len: Maximum sequence length (default: 2048)
            max_tokens_per_gpu: Maximum tokens per GPU (reserved for future implementation)
            data_output_dir: Directory to save processed data
            save_samples: Number of samples to save after training
            warmup_steps: Number of warmup steps
            accelerate_full_state_at_epoch: Whether to save full state at epoch
            checkpoint_at_epoch: Whether to checkpoint at each epoch

            LoRA Parameters (from PEFT extender):
            lora_r: LoRA rank (default: 16)
            lora_alpha: LoRA alpha parameter (default: 32)
            lora_dropout: LoRA dropout rate (default: 0.0, optimized for Unsloth)
            target_modules: List of module names to apply LoRA to (default: auto-detect)
            use_rslora: Use Rank-Stabilized LoRA (default: False)
            use_dora: Use DoRA (Weight-Decomposed Low-Rank Adaptation) (default: False)
            init_lora_weights: How to initialize LoRA weights (default: True)
            rank_pattern: Custom rank pattern for different modules
            alpha_pattern: Custom alpha pattern for different modules
            loftq_config: LoFTQ configuration for quantization-aware LoRA

            Extended Training Parameters:
            micro_batch_size: Batch size per GPU (default: 2)
            gradient_accumulation_steps: Steps to accumulate gradients (default: 1)
            lr_scheduler: Learning rate scheduler (default: 'linear')
            weight_decay: Weight decay for regularization (default: 0.01)
            max_grad_norm: Maximum gradient norm for clipping (default: 0.3)

            Quantization Parameters:
            load_in_4bit: Use 4-bit quantization (QLoRA)
            load_in_8bit: Use 8-bit quantization
            bnb_4bit_quant_type: 4-bit quantization type (default: 'nf4')
            bnb_4bit_compute_dtype: Compute dtype for 4-bit (default: 'bfloat16')
            bnb_4bit_use_double_quant: Use double quantization (default: True)

            Optimization Parameters:
            flash_attention: Use flash attention as default over installed xformers (reserved for future implementation)
            sample_packing: Pack multiple samples per sequence (default: True)
            bf16: Use bfloat16 precision (default: True)
            fp16: Use float16 precision (default: False)
            tf32: Use TensorFloat-32 (reserved for future implementation)

            Logging and Saving:
            save_steps: Steps between checkpoints (default: 500)
            eval_steps: Steps between evaluations (default: 500)
            logging_steps: Steps between log outputs (default: 10)
            save_total_limit: Maximum number of checkpoints to keep (default: 3)
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases entity name
            wandb_run_name: Weights & Biases run name
            tensorboard_log_dir: TensorBoard log directory
            mlflow_tracking_uri: MLflow tracking URI
            mlflow_experiment_name: MLflow experiment name
            mlflow_run_name: MLflow run name

            Dataset Format Parameters:
            dataset_type: Dataset format type ('chat_template', 'alpaca', 'passthrough')
            field_messages: Field name for messages (default: 'messages' for chat_template)
            field_instruction: Field name for instruction (for alpaca format)
            field_input: Field name for input (for alpaca format)
            field_output: Field name for output (for alpaca format)

            Distributed Training:
            nproc_per_node: Number of processes (GPUs) per node
            nnodes: Total number of nodes
            node_rank: Rank of this node (0 to nnodes-1)
            rdzv_id: Unique job ID for rendezvous
            rdzv_endpoint: Master node endpoint for multi-node training
            master_addr: Master node address for distributed training
            master_port: Master node port for distributed training

            Multi-GPU Configuration:
            enable_model_splitting: Enable device_map="balanced" for large models (default: False)
                                   Use for models that don't fit on a single GPU (e.g., Llama 70B)
                                   For smaller models, use standard DDP instead

            Advanced:
            **kwargs: Additional parameters passed to the backend

        Returns:
            Dictionary containing trained model, tokenizer, and trainer
        """
        # Build base parameters dict (required parameters)
        params = {
            'model_path': model_path,
            'data_path': data_path,
            'ckpt_output_dir': ckpt_output_dir
        }

        # Add all optional parameters (SFT + LoRA + additional)
        optional_params = {
            # SFT parameters (inherited from SFT algorithm)
            'num_epochs': num_epochs,
            'effective_batch_size': effective_batch_size,
            'learning_rate': learning_rate,
            'max_seq_len': max_seq_len,
            'max_tokens_per_gpu': max_tokens_per_gpu,
            'data_output_dir': data_output_dir,
            'save_samples': save_samples,
            'warmup_steps': warmup_steps,
            'accelerate_full_state_at_epoch': accelerate_full_state_at_epoch,
            'checkpoint_at_epoch': checkpoint_at_epoch,
            # LoRA parameters (from PEFT extender)
            'lora_r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'target_modules': target_modules,
            'use_rslora': use_rslora,
            'use_dora': use_dora,
            'init_lora_weights': init_lora_weights,
            'rank_pattern': rank_pattern,
            'alpha_pattern': alpha_pattern,
            'loftq_config': loftq_config,
            # Quantization parameters (QLoRA)
            'load_in_4bit': load_in_4bit,
            'load_in_8bit': load_in_8bit,
            'bnb_4bit_quant_type': bnb_4bit_quant_type,
            'bnb_4bit_compute_dtype': bnb_4bit_compute_dtype,
            'bnb_4bit_use_double_quant': bnb_4bit_use_double_quant,
            # Extended training parameters
            'micro_batch_size': micro_batch_size,
            'gradient_accumulation_steps': gradient_accumulation_steps,
            'lr_scheduler': lr_scheduler,
            'weight_decay': weight_decay,
            'max_grad_norm': max_grad_norm,
            # Optimization parameters
            'flash_attention': flash_attention,
            'sample_packing': sample_packing,
            'bf16': bf16,
            'fp16': fp16,
            'tf32': tf32,
            # Saving and logging
            'save_steps': save_steps,
            'eval_steps': eval_steps,
            'logging_steps': logging_steps,
            'save_total_limit': save_total_limit,
            # Logging configuration
            'wandb_project': wandb_project,
            'wandb_entity': wandb_entity,
            'wandb_run_name': wandb_run_name,
            'tensorboard_log_dir': tensorboard_log_dir,
            'mlflow_tracking_uri': mlflow_tracking_uri,
            'mlflow_experiment_name': mlflow_experiment_name,
            'mlflow_run_name': mlflow_run_name,
            # Dataset format parameters
            'dataset_type': dataset_type,
            'field_messages': field_messages,
            'field_instruction': field_instruction,
            'field_input': field_input,
            'field_output': field_output,
            # Distributed training parameters (inherited from SFT)
            'nproc_per_node': nproc_per_node,
            'nnodes': nnodes,
            'node_rank': node_rank,
            'rdzv_id': rdzv_id,
            'rdzv_endpoint': rdzv_endpoint,
            'master_addr': master_addr,
            'master_port': master_port,
            # Multi-GPU model splitting
            'enable_model_splitting': enable_model_splitting,
        }

        # Only add non-None parameters
        for key, value in optional_params.items():
            if value is not None:
                params[key] = value

        # Add any additional kwargs
        params.update(kwargs)

        # Apply PEFT configuration using the extender
        params = self.peft_extender.apply_peft_config(params)

        return self.backend.execute_training(params)

    def get_required_params(self) -> Dict[str, Type]:
        """Return required parameters for LoRA SFT (same as base SFT)."""
        return {
            'model_path': str,
            'data_path': str,
            'ckpt_output_dir': str,
        }

    def get_optional_params(self) -> Dict[str, Type]:
        """Return optional parameters for LoRA SFT (combines SFT + LoRA parameters)."""
        # Get SFT parameters from the base algorithm
        sft_params = {
            # SFT parameters (inherited from SFT algorithm)
            'num_epochs': int,
            'effective_batch_size': int,
            'learning_rate': float,
            'max_seq_len': int,
            'max_tokens_per_gpu': int,
            'data_output_dir': str,
            'save_samples': int,
            'warmup_steps': int,
            'accelerate_full_state_at_epoch': bool,
            'checkpoint_at_epoch': bool,
            # Distributed training parameters (inherited from SFT)
            'nproc_per_node': Union[str, int],
            'nnodes': int,
            'node_rank': int,
            'rdzv_id': Union[str, int],
            'rdzv_endpoint': str,
            'master_addr': str,
            'master_port': int,
        }

        # Get LoRA parameters from the PEFT extender
        lora_params = self.peft_extender.get_peft_params()

        # Extended training parameters
        extended_params = {
            'micro_batch_size': int,
            'gradient_accumulation_steps': int,
            'lr_scheduler': str,
            'weight_decay': float,
            'max_grad_norm': float,
            # Optimization parameters
            'flash_attention': bool,
            'sample_packing': bool,
            'bf16': bool,
            'fp16': bool,
            'tf32': bool,
            # Saving and logging
            'save_steps': int,
            'eval_steps': int,
            'logging_steps': int,
            'save_total_limit': int,
            # Logging configuration
            'wandb_project': str,
            'wandb_entity': str,
            'wandb_run_name': str,
            'tensorboard_log_dir': str,
            'mlflow_tracking_uri': str,
            'mlflow_experiment_name': str,
            'mlflow_run_name': str,
            # Dataset format parameters
            'dataset_type': str,
            'field_messages': str,
            'field_instruction': str,
            'field_input': str,
            'field_output': str,
            # Multi-GPU model splitting
            'enable_model_splitting': bool,
            # Model saving
            'save_model': bool,
        }

        # Combine all parameter types
        all_params = {}
        all_params.update(sft_params)
        all_params.update(lora_params)
        all_params.update(extended_params)

        return all_params


# Register the algorithm and backend
AlgorithmRegistry.register_algorithm('lora_sft', LoRASFTAlgorithm)
AlgorithmRegistry.register_backend('lora_sft', 'unsloth', UnslothLoRABackend)


def lora_sft(model_path: str,
         data_path: str,
         ckpt_output_dir: str,
         backend: str = "unsloth",
         # LoRA-specific parameters
         lora_r: Optional[int] = None,
         lora_alpha: Optional[int] = None,
         lora_dropout: Optional[float] = None,
         target_modules: Optional[List[str]] = None,
         # Training parameters
         num_epochs: Optional[int] = None,
         effective_batch_size: Optional[int] = None,
         micro_batch_size: Optional[int] = None,
         gradient_accumulation_steps: Optional[int] = None,
         learning_rate: Optional[float] = None,
         max_seq_len: Optional[int] = None,
         lr_scheduler: Optional[str] = None,
         warmup_steps: Optional[int] = None,
         # Quantization parameters
         load_in_4bit: Optional[bool] = None,
         load_in_8bit: Optional[bool] = None,
         bnb_4bit_quant_type: Optional[str] = None,
         bnb_4bit_compute_dtype: Optional[str] = None,
         bnb_4bit_use_double_quant: Optional[bool] = None,
         # Optimization parameters
         flash_attention: Optional[bool] = None,
         sample_packing: Optional[bool] = None,
         bf16: Optional[bool] = None,
         fp16: Optional[bool] = None,
         tf32: Optional[bool] = None,
         # Saving and logging
         save_steps: Optional[int] = None,
         eval_steps: Optional[int] = None,
         logging_steps: Optional[int] = None,
         save_total_limit: Optional[int] = None,
         # Logging configuration
         wandb_project: Optional[str] = None,
         wandb_entity: Optional[str] = None,
         wandb_run_name: Optional[str] = None,
         tensorboard_log_dir: Optional[str] = None,
         mlflow_tracking_uri: Optional[str] = None,
         mlflow_experiment_name: Optional[str] = None,
         mlflow_run_name: Optional[str] = None,
         # Dataset format parameters
         dataset_type: Optional[str] = None,
         field_messages: Optional[str] = None,
         field_instruction: Optional[str] = None,
         field_input: Optional[str] = None,
         field_output: Optional[str] = None,
         # Distributed training parameters
         nproc_per_node: Optional[Union[str, int]] = None,
         nnodes: Optional[int] = None,
         node_rank: Optional[int] = None,
         rdzv_id: Optional[Union[str, int]] = None,
         rdzv_endpoint: Optional[str] = None,
         master_addr: Optional[str] = None,
         master_port: Optional[int] = None,
         # Multi-GPU model splitting
         enable_model_splitting: Optional[bool] = None,
         **kwargs) -> Any:
    """Convenience function to run LoRA + SFT training.

    Args:
        model_path: Path to the model to fine-tune (local path or HuggingFace model ID)
        data_path: Path to the training data (JSON/JSONL format)
        ckpt_output_dir: Directory to save checkpoints and outputs
        backend: Backend implementation to use (default: "unsloth")

        LoRA Parameters:
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha parameter (default: 32)
        lora_dropout: LoRA dropout rate (default: 0.0, optimized for Unsloth)
        target_modules: List of module names to apply LoRA to (default: auto-detect)

        Training Parameters:
        num_epochs: Number of training epochs (default: 3)
        effective_batch_size: Effective batch size across all GPUs
        micro_batch_size: Batch size per GPU (default: 2)
        gradient_accumulation_steps: Steps to accumulate gradients (default: 1)
        learning_rate: Learning rate (default: 2e-4)
        max_seq_len: Maximum sequence length (default: 2048)
        lr_scheduler: Learning rate scheduler (default: 'linear')
        warmup_steps: Number of warmup steps (default: 10)

        Quantization Parameters (QLoRA):
        load_in_4bit: Use 4-bit quantization for QLoRA
        load_in_8bit: Use 8-bit quantization
        bnb_4bit_quant_type: 4-bit quantization type (default: 'nf4')
        bnb_4bit_compute_dtype: Compute dtype for 4-bit (default: 'bfloat16')
        bnb_4bit_use_double_quant: Use double quantization (default: True)

        Optimization Parameters:
        flash_attention: Use Flash Attention for memory efficiency (default: True)
        sample_packing: Pack multiple samples per sequence (default: True)
        bf16: Use bfloat16 precision (default: True)
        fp16: Use float16 precision (default: False)
        tf32: Use TensorFloat-32 (default: True)

        Logging and Saving:
        save_steps: Steps between checkpoints (default: 500)
        eval_steps: Steps between evaluations (default: 500)
        logging_steps: Steps between log outputs (default: 10)
        save_total_limit: Maximum number of checkpoints to keep (default: 3)
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases entity name
        wandb_run_name: Weights & Biases run name
        tensorboard_log_dir: TensorBoard log directory
        mlflow_tracking_uri: MLflow tracking URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name

        Distributed Training:
        nproc_per_node: Number of processes (GPUs) per node
        nnodes: Total number of nodes
        node_rank: Rank of this node (0 to nnodes-1)
        rdzv_id: Unique job ID for rendezvous
        rdzv_endpoint: Master node endpoint for multi-node training
        master_addr: Master node address for distributed training
        master_port: Master node port for distributed training

        Multi-GPU Configuration:
        enable_model_splitting: Enable device_map="balanced" for large models (default: False)
                               Use for models that don't fit on a single GPU (e.g., Llama 70B)
                               For smaller models, use standard DDP with torchrun instead

        Advanced:
        **kwargs: Additional parameters passed to the backend

    Returns:
        Dictionary containing trained model, tokenizer, and trainer

    Example:
        # Basic LoRA training
        result = lora_sft(
            model_path="Qwen/Qwen2.5-1.5B-Instruct",
            data_path="./training_data.jsonl",
            ckpt_output_dir="./outputs",
            lora_r=16,
            lora_alpha=32,
            num_epochs=3,
            learning_rate=2e-4
        )

        # QLoRA with 4-bit quantization
        result = lora_sft(
            model_path="Qwen/Qwen2.5-1.5B-Instruct",
            data_path="./training_data.jsonl",
            ckpt_output_dir="./outputs",
            load_in_4bit=True,
            lora_r=64,
            lora_alpha=128,
            max_seq_len=4096
        )
    """
    from . import create_algorithm

    algorithm = create_algorithm('lora_sft', backend)
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        num_epochs=num_epochs,
        effective_batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=bnb_4bit_compute_dtype,
        bnb_4bit_use_double_quant=bnb_4bit_use_double_quant,
        flash_attention=flash_attention,
        sample_packing=sample_packing,
        bf16=bf16,
        fp16=fp16,
        tf32=tf32,
        save_steps=save_steps,
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_total_limit=save_total_limit,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        tensorboard_log_dir=tensorboard_log_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        dataset_type=dataset_type,
        field_messages=field_messages,
        field_instruction=field_instruction,
        field_input=field_input,
        field_output=field_output,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        rdzv_id=rdzv_id,
        rdzv_endpoint=rdzv_endpoint,
        master_addr=master_addr,
        master_port=master_port,
        enable_model_splitting=enable_model_splitting,
        **kwargs
    )
