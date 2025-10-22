import os
from typing import Any, Dict, List, Optional, Type, Union
from pathlib import Path

from . import Algorithm, Backend, AlgorithmRegistry
from .sft import SFTAlgorithm
from .peft_extender import LoRAPEFTExtender, get_lora_parameters, apply_lora_defaults
from training_hub import utils


class UnslothLoRABackend(Backend):
    """Unsloth backend for LoRA algorithm with performance optimizations."""

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute LoRA training using Unsloth optimizations."""
        try:
            from unsloth import FastLanguageModel
            from trl import SFTTrainer, SFTConfig
            from transformers import TrainingArguments
        except ImportError as e:
            raise ImportError(
                "Unsloth and TRL are required for Unsloth LoRA training. Install with: "
                "pip install unsloth trl"
            ) from e

        # Separate torchrun parameters from training parameters
        torchrun_keys = {'nproc_per_node', 'nnodes', 'node_rank', 'rdzv_id', 'rdzv_endpoint', 'master_addr', 'master_port'}

        # Extract torchrun parameters
        torchrun_params = {k: v for k, v in algorithm_params.items() if k in torchrun_keys}

        # Extract training parameters (everything except torchrun params)
        training_params = {k: v for k, v in algorithm_params.items() if k not in torchrun_keys}

        # Set up distributed training if needed
        if torchrun_params:
            self._setup_distributed_training(torchrun_params)

        # Load model with Unsloth optimizations
        model, tokenizer = self._load_unsloth_model(training_params)

        # Apply LoRA with Unsloth optimizations
        model = self._apply_lora_config(model, training_params)

        # Prepare dataset
        train_dataset = self._prepare_dataset(training_params, tokenizer)

        # Configure training arguments
        training_args = self._build_training_args(training_params)

        # Create SFT trainer
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            args=training_args,
            max_seq_length=training_params.get('max_seq_len', 2048),
            dataset_text_field="text" if training_params.get('dataset_type') != 'chat_template' else None,
            formatting_func=self._format_chat_template if training_params.get('dataset_type') == 'chat_template' else None,
            packing=training_params.get('sample_packing', True),
        )

        # Execute training
        trainer.train()

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

        # Determine quantization settings
        load_in_4bit = params.get('load_in_4bit', False)
        load_in_8bit = params.get('load_in_8bit', False)

        # Use 4bit by default for memory efficiency unless specified otherwise
        if not load_in_4bit and not load_in_8bit:
            load_in_4bit = True

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=params['model_path'],
            max_seq_length=params.get('max_seq_len', 2048),
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,
            # Additional Unsloth optimizations
            # trust_remote_code=params.get('trust_remote_code', False),
        )

        return model, tokenizer

    def _apply_lora_config(self, model, params: Dict[str, Any]):
        """Apply LoRA configuration using Unsloth optimizations."""
        from unsloth import FastLanguageModel

        model = FastLanguageModel.get_peft_model(
            model,
            r=params.get('lora_r', 16),
            target_modules=params.get('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_alpha=params.get('lora_alpha', 32),
            lora_dropout=params.get('lora_dropout', 0.1),
            bias="none",
            use_gradient_checkpointing="unsloth",  # Unsloth's optimized gradient checkpointing
            random_state=params.get('seed', 3407),
            use_rslora=False,
            loftq_config=None,
        )

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

        if dataset_type == 'alpaca':
            # Convert alpaca format to text
            def format_alpaca(examples):
                texts = []
                for i in range(len(examples['instruction'])):
                    instruction = examples['instruction'][i]
                    input_text = examples.get('input', [''] * len(examples['instruction']))[i]
                    output = examples['output'][i]

                    if input_text:
                        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                    else:
                        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    texts.append(text)
                return {"text": texts}

            dataset = dataset.map(format_alpaca, batched=True)

        return dataset

    def _format_chat_template(self, example):
        """Format chat template for messages format."""
        messages = example.get('messages', [])
        if not messages:
            return ""

        # Simple chat formatting - this could be enhanced based on model-specific templates
        formatted_text = ""
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            if role == 'user':
                formatted_text += f"### Human:\n{content}\n\n"
            elif role == 'assistant':
                formatted_text += f"### Assistant:\n{content}\n\n"
            elif role == 'system':
                formatted_text += f"### System:\n{content}\n\n"

        return formatted_text.strip()

    def _build_training_args(self, params: Dict[str, Any]) -> 'TrainingArguments':
        """Build training arguments for SFTTrainer."""
        from transformers import TrainingArguments

        # Calculate steps and batch sizes
        num_epochs = params.get('num_epochs', 3)
        micro_batch_size = params.get('micro_batch_size', 2)
        gradient_accumulation_steps = params.get('gradient_accumulation_steps', 1)

        # Handle effective batch size calculation
        if 'effective_batch_size' in params:
            num_gpus = params.get('nproc_per_node', 1)
            if isinstance(num_gpus, str):
                num_gpus = 1

            gradient_accumulation_steps = params['effective_batch_size'] // (micro_batch_size * num_gpus)
            gradient_accumulation_steps = max(1, gradient_accumulation_steps)

        training_args = TrainingArguments(
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
            save_steps=params.get('save_steps', 500),
            eval_steps=params.get('eval_steps', 500),
            save_total_limit=params.get('save_total_limit', 3),

            # Performance optimizations
            dataloader_pin_memory=False,  # Unsloth recommendation
            remove_unused_columns=False,  # Required for custom datasets

            # Optional: Weights & Biases
            report_to="wandb" if params.get('wandb_project') else None,
            run_name=params.get('wandb_run_name'),
        )

        return training_args

    def _setup_distributed_training(self, torchrun_params: Dict[str, Any]) -> None:
        """Set up distributed training environment variables."""
        import os

        # Set environment variables for distributed training
        if 'master_addr' in torchrun_params:
            os.environ['MASTER_ADDR'] = str(torchrun_params['master_addr'])
        if 'master_port' in torchrun_params:
            os.environ['MASTER_PORT'] = str(torchrun_params['master_port'])
        if 'node_rank' in torchrun_params:
            os.environ['NODE_RANK'] = str(torchrun_params['node_rank'])
        if 'nnodes' in torchrun_params:
            os.environ['WORLD_SIZE'] = str(torchrun_params['nnodes'])


class AxolotlLoRABackend(Backend):
    """Axolotl backend for LoRA algorithm."""

    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute LoRA training using Axolotl."""
        try:
            from axolotl.train import train
            from axolotl.utils.dict import DictDefault
            from axolotl.utils.data import prepare_dataset
        except ImportError as e:
            raise ImportError(
                "Axolotl is required for LoRA training. Install it with: "
                "pip install axolotl[flash-attn,deepspeed]"
            ) from e

        # Separate torchrun parameters from training parameters
        torchrun_keys = {'nproc_per_node', 'nnodes', 'node_rank', 'rdzv_id', 'rdzv_endpoint', 'master_addr', 'master_port'}

        # Extract torchrun parameters
        torchrun_params = {k: v for k, v in algorithm_params.items() if k in torchrun_keys}

        # Extract training parameters (everything except torchrun params)
        training_params = {k: v for k, v in algorithm_params.items() if k not in torchrun_keys}

        # Build Axolotl configuration
        cfg = self._build_axolotl_config(training_params)

        # Prepare dataset metadata
        dataset_meta = self._prepare_dataset_meta(training_params)

        # Set up distributed training if needed
        if torchrun_params:
            self._setup_distributed_training(torchrun_params)

        # Execute training
        model, tokenizer, trainer = train(cfg=cfg, dataset_meta=dataset_meta)

        return {
            'model': model,
            'tokenizer': tokenizer,
            'trainer': trainer
        }

    def _build_axolotl_config(self, params: Dict[str, Any]) -> 'DictDefault':
        """Build Axolotl configuration from training hub parameters."""
        from axolotl.utils.dict import DictDefault

        # Base configuration
        cfg = DictDefault({
            # Model configuration
            'base_model': params['model_path'],
            'model_type': 'AutoModelForCausalLM',
            'tokenizer_type': 'AutoTokenizer',

            # Dataset configuration
            'datasets': [{
                'path': params['data_path'],
                'type': params.get('dataset_type', 'chat_template'),  # Default to messages format for compatibility
            }],

            # LoRA configuration
            'adapter': 'lora',
            'lora_r': params.get('lora_r', 16),
            'lora_alpha': params.get('lora_alpha', 32),
            'lora_dropout': params.get('lora_dropout', 0.1),
            'lora_target_modules': params.get('target_modules'),

            # Training configuration
            'sequence_len': params.get('max_seq_len', 2048),
            'sample_packing': params.get('sample_packing', True),
            'pad_to_sequence_len': True,

            # Training hyperparameters
            'num_epochs': params.get('num_epochs', 3),
            'micro_batch_size': params.get('micro_batch_size', 1),
            'gradient_accumulation_steps': params.get('gradient_accumulation_steps', 1),
            'learning_rate': params.get('learning_rate', 2e-4),
            'lr_scheduler': params.get('lr_scheduler', 'cosine'),
            'warmup_steps': params.get('warmup_steps', 10),

            # Output configuration
            'output_dir': params['ckpt_output_dir'],
            'save_steps': params.get('save_steps', 500),
            'eval_steps': params.get('eval_steps', 500),
            'save_total_limit': params.get('save_total_limit', 3),

            # Optimization features
            'flash_attention': params.get('flash_attention', True),
            'bf16': params.get('bf16', True),
            'fp16': params.get('fp16', False),
            'tf32': params.get('tf32', True),

            # Quantization (if specified)
            'load_in_8bit': params.get('load_in_8bit', False),
            'load_in_4bit': params.get('load_in_4bit', False),

            # Logging
            'logging_steps': params.get('logging_steps', 10),
            'wandb_project': params.get('wandb_project'),
            'wandb_entity': params.get('wandb_entity'),
            'wandb_watch': params.get('wandb_watch'),

            # Early stopping
            'early_stopping_patience': params.get('early_stopping_patience'),
        })

        # Handle quantization settings
        if params.get('load_in_4bit'):
            cfg.update({
                'load_in_4bit': True,
                'bnb_4bit_quant_type': params.get('bnb_4bit_quant_type', 'nf4'),
                'bnb_4bit_compute_dtype': params.get('bnb_4bit_compute_dtype', 'bfloat16'),
                'bnb_4bit_use_double_quant': params.get('bnb_4bit_use_double_quant', True),
            })

        # Handle dataset format configuration
        dataset_config = cfg['datasets'][0]
        if params.get('field_messages'):
            dataset_config['field_messages'] = params['field_messages']
        if params.get('field_instruction'):
            dataset_config['field_instruction'] = params['field_instruction']
        if params.get('field_input'):
            dataset_config['field_input'] = params['field_input']
        if params.get('field_output'):
            dataset_config['field_output'] = params['field_output']

        # Calculate effective batch size if needed
        if 'effective_batch_size' in params:
            micro_batch_size = cfg.get('micro_batch_size', 1)
            num_gpus = params.get('nproc_per_node', 1)
            if isinstance(num_gpus, str):
                num_gpus = 1  # Default for 'auto' or 'gpu'

            gradient_accumulation_steps = params['effective_batch_size'] // (micro_batch_size * num_gpus)
            cfg['gradient_accumulation_steps'] = max(1, gradient_accumulation_steps)

        # Handle any additional axolotl-specific parameters
        axolotl_params = params.get('axolotl_config', {})
        cfg.update(axolotl_params)

        return cfg

    def _prepare_dataset_meta(self, params: Dict[str, Any]) -> Any:
        """Prepare dataset metadata for Axolotl training."""
        # For now, return None - Axolotl will handle dataset preparation
        # This can be extended to provide custom dataset handling if needed
        return None

    def _setup_distributed_training(self, torchrun_params: Dict[str, Any]) -> None:
        """Set up distributed training environment variables."""
        # Set environment variables for distributed training
        if 'master_addr' in torchrun_params:
            os.environ['MASTER_ADDR'] = str(torchrun_params['master_addr'])
        if 'master_port' in torchrun_params:
            os.environ['MASTER_PORT'] = str(torchrun_params['master_port'])
        if 'node_rank' in torchrun_params:
            os.environ['NODE_RANK'] = str(torchrun_params['node_rank'])
        if 'nnodes' in torchrun_params:
            os.environ['WORLD_SIZE'] = str(torchrun_params['nnodes'])


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
              # Weights & Biases
              wandb_project: Optional[str] = None,
              wandb_entity: Optional[str] = None,
              wandb_watch: Optional[str] = None,
              wandb_run_name: Optional[str] = None,
              # Early stopping
              early_stopping_patience: Optional[int] = None,
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
              # Backend-specific configurations
              axolotl_config: Optional[Dict[str, Any]] = None,
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
            max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch
            data_output_dir: Directory to save processed data
            save_samples: Number of samples to save after training
            warmup_steps: Number of warmup steps
            accelerate_full_state_at_epoch: Whether to save full state at epoch
            checkpoint_at_epoch: Whether to checkpoint at each epoch

            LoRA Parameters (from PEFT extender):
            lora_r: LoRA rank (default: 16)
            lora_alpha: LoRA alpha parameter (default: 32)
            lora_dropout: LoRA dropout rate (default: 0.1)
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
            flash_attention: Use Flash Attention (default: True)
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
            wandb_watch: What to watch in W&B ('gradients', 'all', etc.)
            early_stopping_patience: Early stopping patience (epochs)

            Dataset Format Parameters:
            dataset_type: Dataset format type ('chat_template', 'alpaca', 'input_output', etc.)
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

            Advanced:
            axolotl_config: Additional Axolotl configuration dictionary
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
            'wandb_project': wandb_project,
            'wandb_entity': wandb_entity,
            'wandb_watch': wandb_watch,
            'wandb_run_name': wandb_run_name,
            'early_stopping_patience': early_stopping_patience,
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
            # Backend-specific configurations
            'axolotl_config': axolotl_config,
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
            'wandb_project': str,
            'wandb_entity': str,
            'wandb_watch': str,
            'wandb_run_name': str,
            'early_stopping_patience': int,
            # Dataset format parameters
            'dataset_type': str,
            'field_messages': str,
            'field_instruction': str,
            'field_input': str,
            'field_output': str,
            # Backend-specific configurations
            'axolotl_config': Dict[str, Any],
        }

        # Combine all parameter types
        all_params = {}
        all_params.update(sft_params)
        all_params.update(lora_params)
        all_params.update(extended_params)

        return all_params


# Register the algorithm and backends
AlgorithmRegistry.register_algorithm('lora_sft', LoRASFTAlgorithm)
AlgorithmRegistry.register_backend('lora_sft', 'unsloth', UnslothLoRABackend)
AlgorithmRegistry.register_backend('lora_sft', 'axolotl', AxolotlLoRABackend)


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
         # Weights & Biases
         wandb_project: Optional[str] = None,
         wandb_entity: Optional[str] = None,
         wandb_watch: Optional[str] = None,
         # Early stopping
         early_stopping_patience: Optional[int] = None,
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
         # Additional Axolotl configuration
         axolotl_config: Optional[Dict[str, Any]] = None,
         **kwargs) -> Any:
    """Convenience function to run LoRA + SFT training.

    Args:
        model_path: Path to the model to fine-tune (local path or HuggingFace model ID)
        data_path: Path to the training data (JSON/JSONL format)
        ckpt_output_dir: Directory to save checkpoints and outputs
        backend: Backend implementation to use (default: "axolotl")

        LoRA Parameters:
        lora_r: LoRA rank (default: 16)
        lora_alpha: LoRA alpha parameter (default: 32)
        lora_dropout: LoRA dropout rate (default: 0.1)
        target_modules: List of module names to apply LoRA to (default: auto-detect)

        Training Parameters:
        num_epochs: Number of training epochs (default: 3)
        effective_batch_size: Effective batch size across all GPUs
        micro_batch_size: Batch size per GPU (default: 1)
        gradient_accumulation_steps: Steps to accumulate gradients (default: 1)
        learning_rate: Learning rate (default: 2e-4)
        max_seq_len: Maximum sequence length (default: 2048)
        lr_scheduler: Learning rate scheduler (default: 'cosine')
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
        wandb_watch: What to watch in W&B ('gradients', 'all', etc.)
        early_stopping_patience: Early stopping patience (epochs)

        Distributed Training:
        nproc_per_node: Number of processes (GPUs) per node
        nnodes: Total number of nodes
        node_rank: Rank of this node (0 to nnodes-1)
        rdzv_id: Unique job ID for rendezvous
        rdzv_endpoint: Master node endpoint for multi-node training
        master_addr: Master node address for distributed training
        master_port: Master node port for distributed training

        Advanced:
        axolotl_config: Additional Axolotl configuration dictionary to override defaults
        **kwargs: Additional parameters passed to the backend

    Returns:
        Dictionary containing trained model, tokenizer, and trainer

    Example:
        # Basic LoRA training
        result = lora(
            model_path="microsoft/DialoGPT-medium",
            data_path="./training_data.jsonl",
            ckpt_output_dir="./outputs",
            lora_r=16,
            lora_alpha=32,
            num_epochs=3,
            learning_rate=2e-4
        )

        # QLoRA with 4-bit quantization
        result = lora(
            model_path="meta-llama/Llama-2-7b-hf",
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
        wandb_watch=wandb_watch,
        early_stopping_patience=early_stopping_patience,
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
        axolotl_config=axolotl_config,
        **kwargs
    )