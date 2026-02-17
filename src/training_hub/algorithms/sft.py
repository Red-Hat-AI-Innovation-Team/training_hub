import os
from typing import Any, Dict, Type, Optional
from instructlab.training import (
    run_training,
    TorchrunArgs,
    TrainingArgs,
    PretrainingConfig,
)

from . import Algorithm, Backend, AlgorithmRegistry
from training_hub import utils


class InstructLabTrainingSFTBackend(Backend):
    """InstructLab Training backend for SFT algorithm."""
    
    def execute_training(self, algorithm_params: Dict[str, Any]) -> Any:
        """Execute SFT training using instructlab-training."""
        # Separate torchrun parameters from training parameters
        torchrun_keys = {'nproc_per_node', 'nnodes', 'node_rank', 'rdzv_id', 'rdzv_endpoint', 'master_addr', 'master_port'}
        
        # Extract torchrun parameters
        torchrun_params = {k: v for k, v in algorithm_params.items() if k in torchrun_keys}

        # Extract training parameters (everything except torchrun params)
        # Note: instructlab-training auto-detects loggers based on mlflow_tracking_uri,
        # wandb_project, and tensorboard_log_dir parameters
        training_params = {k: v for k, v in algorithm_params.items() if k not in torchrun_keys}
        
        # Map training_hub parameter names to instructlab-training parameter names
        if 'max_tokens_per_gpu' in training_params:
            training_params['max_batch_len'] = training_params.pop('max_tokens_per_gpu')

        # AdamW parameter translation
        if 'beta1' in training_params and 'beta2' in training_params:
            training_params['adamw_betas'] = (
                training_params.pop('beta1'),
                training_params.pop('beta2')
            )
        if 'eps' in training_params:
            training_params['adamw_eps'] = training_params.pop('eps')
        if 'weight_decay' in training_params:
            training_params['adamw_weight_decay'] = training_params.pop('weight_decay')

        # Create the pretraining config if it was requested
        block_size = training_params.pop('block_size', None)
        document_column_name = training_params.pop('document_column_name', None)
        is_pretraining = training_params.pop('is_pretraining', None)

        if is_pretraining and block_size is None:
            raise ValueError("block_size is required when is_pretraining=True")

        if is_pretraining:
            pretraining_kwargs: Dict[str, Any] = {}
            if document_column_name is not None:
                pretraining_kwargs['document_column_name'] = document_column_name
            training_params['pretraining_config'] = PretrainingConfig(
                block_size=block_size,
                **pretraining_kwargs,
            )

        # Create TrainingArgs with all provided parameters, letting it handle defaults
        training_args = TrainingArgs(**training_params)
        
        # Set up torchrun arguments with single-node defaults (except nproc_per_node)
        final_torchrun_params = utils.get_torchrun_params(torchrun_params)
        torchrun_args = TorchrunArgs(**final_torchrun_params)

        # Execute training
        return run_training(
            torch_args=torchrun_args,
            train_args=training_args
        )


class SFTAlgorithm(Algorithm):
    """Supervised Fine-Tuning algorithm."""
    
    def __init__(self, backend: Backend, **kwargs):
        self.backend = backend
        self.config = kwargs
    
    def train(self, 
              model_path: str,
              data_path: str, 
              ckpt_output_dir: str,
              # Training parameters (defaults from TrainingArgs)
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
              is_pretraining: Optional[bool] = None,
              block_size: Optional[int] = None,
              document_column_name: Optional[str] = None,
              # AdamW optimizer parameters
              beta1: Optional[float] = None,
              beta2: Optional[float] = None,
              eps: Optional[float] = None,
              weight_decay: Optional[float] = None,
              # Torchrun parameters for multi-node support
              nproc_per_node: Optional[str | int] = None,
              nnodes: Optional[int] = None,
              node_rank: Optional[int] = None,
              rdzv_id: Optional[str | int] = None,
              rdzv_endpoint: Optional[str] = None,
              master_addr: Optional[str] = None,
              master_port: Optional[int] = None,
              # Logging parameters
              wandb_project: Optional[str] = None,
              wandb_entity: Optional[str] = None,
              wandb_run_name: Optional[str] = None,
              tensorboard_log_dir: Optional[str] = None,
              mlflow_tracking_uri: Optional[str] = None,
              mlflow_experiment_name: Optional[str] = None,
              mlflow_run_name: Optional[str] = None,
              **kwargs) -> Any:
        """Execute SFT training.
        
        Args:
            model_path: Path to the model to fine-tune
            data_path: Path to the training data
            ckpt_output_dir: Directory to save checkpoints
            num_epochs: Number of training epochs
            effective_batch_size: Effective batch size for training
            learning_rate: Learning rate for training
            max_seq_len: Maximum sequence length
            max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.
            data_output_dir: Directory to save processed data
            save_samples: Number of samples to save after training (0 disables saving based on sample count)
            warmup_steps: Number of warmup steps
            accelerate_full_state_at_epoch: Whether to save full state at epoch for automatic checkpoint resumption
            checkpoint_at_epoch: Whether to checkpoint at each epoch
            is_pretraining: Enable document-style continual pretraining mode.
            block_size: Required when `is_pretraining=True`. Token length of each document block.
            document_column_name: Column name containing raw documents when `is_pretraining=True` (defaults to "document").
            beta1: AdamW optimizer beta1 coefficient (momentum).
            beta2: AdamW optimizer beta2 coefficient (RMSprop).
            eps: AdamW optimizer epsilon for numerical stability.
            weight_decay: AdamW optimizer weight decay coefficient.
            nproc_per_node: Number of processes (GPUs) per node
            nnodes: Total number of nodes
            node_rank: Rank of this node (0 to nnodes-1)
            rdzv_id: Unique job ID for rendezvous
            rdzv_endpoint: Master node endpoint for multi-node training
            master_addr: Master node address for distributed training
            master_port: Master node port for distributed training
            wandb_project: Weights & Biases project name
            wandb_entity: Weights & Biases team/entity name
            wandb_run_name: Weights & Biases run name
            tensorboard_log_dir: Directory for TensorBoard logs
            mlflow_tracking_uri: MLflow tracking server URI
            mlflow_experiment_name: MLflow experiment name
            mlflow_run_name: MLflow run name
            **kwargs: Additional parameters passed to the backend
            
        Returns:
            Training result from the backend
        """
        # Build parameters dict, only including non-None values
        params = {'model_path': model_path, 'data_path': data_path, 'ckpt_output_dir': ckpt_output_dir}
        
        # Add optional parameters if provided
        optional_params = {
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
            'is_pretraining': is_pretraining,
            'block_size': block_size,
            'document_column_name': document_column_name,
            # AdamW optimizer parameters
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
            'weight_decay': weight_decay,
            # Torchrun parameters
            'nproc_per_node': nproc_per_node,
            'nnodes': nnodes,
            'node_rank': node_rank,
            'rdzv_id': rdzv_id,
            'rdzv_endpoint': rdzv_endpoint,
            'master_addr': master_addr,
            'master_port': master_port,
            # Logging parameters
            'wandb_project': wandb_project,
            'wandb_entity': wandb_entity,
            'wandb_run_name': wandb_run_name,
            'tensorboard_log_dir': tensorboard_log_dir,
            'mlflow_tracking_uri': mlflow_tracking_uri,
            'mlflow_experiment_name': mlflow_experiment_name,
            'mlflow_run_name': mlflow_run_name,
        }
        
        # Only add non-None parameters (let TrainingArgs handle defaults)
        for key, value in optional_params.items():
            if value is not None:
                params[key] = value
                
        params.update(kwargs)
        
        return self.backend.execute_training(params)
    
    def get_required_params(self) -> Dict[str, Type]:
        """Return required parameters for SFT."""
        return {
            'model_path': str,
            'data_path': str,
            'ckpt_output_dir': str,
            'num_epochs': int,
            'effective_batch_size': int,
            'learning_rate': float,
            'max_seq_len': int,
            'max_batch_len': int,
        }

    def get_optional_params(self) -> Dict[str, Type]:
        """Return optional parameters for SFT."""
        return {
            'max_tokens_per_gpu': int,
            'data_output_dir': str,
            'save_samples': int,
            'warmup_steps': int,
            'accelerate_full_state_at_epoch': bool,
            'checkpoint_at_epoch': bool,
            'is_pretraining': bool,
            'block_size': int,
            'document_column_name': str,
            # AdamW optimizer parameters
            'beta1': float,
            'beta2': float,
            'eps': float,
            'weight_decay': float,
            # Torchrun parameters
            'nproc_per_node': str | int,
            'nnodes': int,
            'node_rank': int,
            'rdzv_id': str | int,
            'rdzv_endpoint': str,
            'master_addr': str,
            'master_port': int,
            # Logging parameters
            'wandb_project': str,
            'wandb_entity': str,
            'wandb_run_name': str,
            'tensorboard_log_dir': str,
            'mlflow_tracking_uri': str,
            'mlflow_experiment_name': str,
            'mlflow_run_name': str,
        }


# Register the algorithm and backend
AlgorithmRegistry.register_algorithm('sft', SFTAlgorithm)
AlgorithmRegistry.register_backend('sft', 'instructlab-training', InstructLabTrainingSFTBackend)


def sft(model_path: str, 
        data_path: str, 
        ckpt_output_dir: str,
        backend: str = "instructlab-training",
        # Training parameters (defaults from TrainingArgs)
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
        is_pretraining: Optional[bool] = None,
        block_size: Optional[int] = None,
        document_column_name: Optional[str] = None,
        # AdamW optimizer parameters
        beta1: Optional[float] = None,
        beta2: Optional[float] = None,
        eps: Optional[float] = None,
        weight_decay: Optional[float] = None,
        # Torchrun parameters for multi-node support
        nproc_per_node: Optional[str | int] = None,
        nnodes: Optional[int] = None,
        node_rank: Optional[int] = None,
        rdzv_id: Optional[str | int] = None,
        rdzv_endpoint: Optional[str] = None,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        # Logging parameters
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        tensorboard_log_dir: Optional[str] = None,
        mlflow_tracking_uri: Optional[str] = None,
        mlflow_experiment_name: Optional[str] = None,
        mlflow_run_name: Optional[str] = None,
        **kwargs) -> Any:
    """Convenience function to run SFT training.
    
    Args:
        model_path: Path to the model to fine-tune
        data_path: Path to the training data
        ckpt_output_dir: Directory to save checkpoints
        backend: Backend implementation to use (default: "instructlab-training")
        num_epochs: Number of training epochs
        effective_batch_size: Effective batch size for training
        learning_rate: Learning rate for training
        max_seq_len: Maximum sequence length
        max_tokens_per_gpu: Maximum tokens per GPU in a mini-batch (hard-cap for memory to avoid OOMs). Used to automatically calculate mini-batch size and gradient accumulation to maintain the desired effective_batch_size while staying within memory limits.
        data_output_dir: Directory to save processed data
        save_samples: Number of samples to save after training (0 disables saving based on sample count)
        warmup_steps: Number of warmup steps
        accelerate_full_state_at_epoch: Whether to save full state at epoch for automatic checkpoint resumption
        checkpoint_at_epoch: Whether to checkpoint at each epoch
        is_pretraining: Enable document-style continual pretraining mode.
        block_size: Required when `is_pretraining=True`. Token length of each document block.
        document_column_name: Column name containing raw documents when `is_pretraining=True`.
        beta1: AdamW optimizer beta1 coefficient (momentum).
        beta2: AdamW optimizer beta2 coefficient (RMSprop).
        eps: AdamW optimizer epsilon for numerical stability.
        weight_decay: AdamW optimizer weight decay coefficient.
        nproc_per_node: Number of processes (GPUs) per node for distributed training
        nnodes: Total number of nodes for distributed training
        node_rank: Rank of this node (0 to nnodes-1) for distributed training
        rdzv_id: Unique job ID for rendezvous in distributed training
        rdzv_endpoint: Master node endpoint for multi-node training
        master_addr: Master node address for distributed training
        master_port: Master node port for distributed training
        wandb_project: Weights & Biases project name
        wandb_entity: Weights & Biases team/entity name
        wandb_run_name: Weights & Biases run name
        tensorboard_log_dir: Directory for TensorBoard logs
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment_name: MLflow experiment name
        mlflow_run_name: MLflow run name
        **kwargs: Additional parameters passed to the backend
    
    Returns:
        Training result from the backend
    """
    from . import create_algorithm
    
    algorithm = create_algorithm('sft', backend)
    return algorithm.train(
        model_path=model_path,
        data_path=data_path,
        ckpt_output_dir=ckpt_output_dir,
        num_epochs=num_epochs,
        effective_batch_size=effective_batch_size,
        learning_rate=learning_rate,
        max_seq_len=max_seq_len,
        max_tokens_per_gpu=max_tokens_per_gpu,
        data_output_dir=data_output_dir,
        save_samples=save_samples,
        warmup_steps=warmup_steps,
        accelerate_full_state_at_epoch=accelerate_full_state_at_epoch,
        checkpoint_at_epoch=checkpoint_at_epoch,
        is_pretraining=is_pretraining,
        block_size=block_size,
        document_column_name=document_column_name,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        weight_decay=weight_decay,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
        node_rank=node_rank,
        rdzv_id=rdzv_id,
        rdzv_endpoint=rdzv_endpoint,
        master_addr=master_addr,
        master_port=master_port,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_run_name=wandb_run_name,
        tensorboard_log_dir=tensorboard_log_dir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_run_name=mlflow_run_name,
        **kwargs
    )

